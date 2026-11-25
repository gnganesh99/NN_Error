"""Active learning on the Random IV data (topography patches -> IV curves).

Mirrors notebooks/STM/stm_ms_acqfn_script.py (and reuses its functions), adapted to this dataset:
- 2001 IV curves measured at random points of a 128x128 grid (only these points are used).
- The IV positions are shifted by --offset (default (-12, +7) px, see obs.md) before patch extraction.
- Image: plane (tilt) removed, then global z-score. Spectra: Savitzky-Golay smoothed, full 256-step sweep.
- Patches reaching the image border are padded (image 0, mask 0). The validity-mask channel is used for
  both ss and ms here, because border padding exists even without multiscale augmentation.
- ms: 'pad' augmentation at --scales plus the full WS patch. ss: only the full WS patch.

Example:
    python iv_acqfn_script.py --scale-type ms --acquisition-method distance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
STM_DIR = SCRIPT_DIR.parent / "STM"
for p in (SCRIPT_DIR, STM_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Importing the STM script puts src/ on the path and registers the nnerror package; its main() is not run.
from stm_ms_acqfn_script import (  # noqa: E402
    CustomDecoder,
    CustomDecoder_mask,
    Norm_G,
    find_row_indices,
    get_model,
    reduce_errors_and_coords,
    save_h5_ragged,
    set_random_seed,
    spectra_errors,
)
from nnerror.training_functions import distance_acq_fn, predict_spectra, train_model  # noqa: E402

from helper_fns import (  # noqa: E402
    apply_offset,
    augment_with_mask,
    flatten_image,
    get_padded_subimages,
    load_random_iv,
    smooth_spectra,
)

# Defaults (override with flags)
WS = 32
scales = "4,8,12,16,20,24,28"  # ms only; the full WS patch is always included
offset = "-12,7"
n_epochs_im2spec = 100
n_epochs_error = 100
error_type = "L1"
model_type = "im2spec"
use_swa = True
latent_dim = 5
last_swa_epochs = 0.1
acquisition_iterations = 50
num_new_points = 5
acquisition_method = "distance"
train_size = 0.1
n_batches = 6
lr = 1e-3
patience = 10
seed = 42
scale_type = "ms"
mask_channel = True

data_folder = SCRIPT_DIR / "content"
map_file = "V3_IV_map_1d2um_0002.h5"
spec_file = "V3_IV_spec_1d2um_0003.h5"
output_folder = SCRIPT_DIR / "data" / "active_learning_results"


def int_list(text):
    return [int(v) for v in str(text).split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Random IV multiscale active learning; saves output_dict to h5.")
    parser.add_argument("--WS", type=int, default=WS, help="patch size (largest scale)")
    parser.add_argument("--scales", default=scales, help="comma-separated smaller scales for ms (each < WS)")
    parser.add_argument("--offset", default=offset, help="'drow,dcol' added to the recorded IV positions")
    parser.add_argument("--n-epochs-im2spec", type=int, default=n_epochs_im2spec)
    parser.add_argument("--n-epochs-error", type=int, default=n_epochs_error)
    parser.add_argument("--error-type", choices=("L1", "L2", "cos"), default=error_type)
    parser.add_argument("--model-type", choices=("im2spec",), default=model_type)
    parser.add_argument("--use-swa", action=argparse.BooleanOptionalAction, default=use_swa)
    parser.add_argument("--latent-dim", type=int, default=latent_dim)
    parser.add_argument("--last-swa-epochs", type=float, default=last_swa_epochs)
    parser.add_argument("--acquisition-iterations", type=int, default=acquisition_iterations)
    parser.add_argument("--num-new-points", type=int, default=num_new_points)
    parser.add_argument("--acquisition-method", choices=("distance", "random"), default=acquisition_method)
    parser.add_argument("--train-size", type=float, default=train_size)
    parser.add_argument("--n-batches", type=int, default=n_batches)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--scale-type", choices=("ss", "ms"), default=scale_type)
    parser.add_argument("--mask-channel", action=argparse.BooleanOptionalAction, default=mask_channel,
                        help="add the validity-mask channel (border padding and, for ms, scale padding)")
    parser.add_argument("--data-folder", type=Path, default=data_folder)
    parser.add_argument("--map-file", default=map_file)
    parser.add_argument("--spec-file", default=spec_file)
    parser.add_argument("--output-folder", type=Path, default=output_folder)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def build_output_file(args, off):
    if args.output_file is not None:
        return args.output_file
    filename = (
        f"al_{args.model_type}_{args.error_type}_WS{args.WS}_latent{args.latent_dim}_appendpad_"
        f"{args.acquisition_iterations}iters_{args.num_new_points}pts_{args.acquisition_method}acq_"
        f"{args.scale_type}_normgauss{'_mask' if args.mask_channel else ''}_off{off[0]}_{off[1]}.h5"
    )
    return args.output_folder / filename


def rows_of(indices, n_points, n_scales):
    """Rows of the augmented 'all points' arrays for these point indices, in augment_with_mask order
    (one block per scale, points in the given order within each block)."""
    indices = np.asarray(indices, dtype=int).reshape(-1)
    return (np.arange(n_scales)[:, None] * n_points + indices[None, :]).ravel()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    off = tuple(int_list(args.offset))
    if len(off) != 2:
        raise ValueError("--offset must be 'drow,dcol'")
    aug_scales = int_list(args.scales) if args.scale_type == "ms" else []
    if any(s >= args.WS or s < 2 for s in aug_scales):
        raise ValueError(f"--scales must be between 2 and WS-1, got {aug_scales}")
    all_scales = aug_scales + [args.WS]
    output_file = build_output_file(args, off)

    print(f"Using device: {device}")
    print(f"scale_type={args.scale_type}, scales={all_scales}, offset={off}, mask_channel={args.mask_channel}")

    # ---- data ----
    data = load_random_iv(args.data_folder / args.map_file, args.data_folder / args.spec_file)
    coords = apply_offset(data["coords"], off)
    spectra_all = smooth_spectra(data["iv"]).astype(np.float32)
    image = Norm_G(data=flatten_image(data["topo"])).normalize()
    patches, border_masks = get_padded_subimages(image, coords, args.WS)

    N, S = len(coords), len(all_scales)
    X_all, Y_all, C_all = augment_with_mask(np.arange(N), patches, border_masks, spectra_all, coords, aug_scales)
    if not args.mask_channel:
        X_all = X_all[:, 0]  # (M, H, W); im2spec adds the channel axis itself
    print(f"points {N}, augmented rows {len(X_all)} ({S} scale(s)), input {X_all.shape[1:]}, spectra {spectra_all.shape[1]}")

    original_indices = np.arange(N)
    indices_train, indices_test = train_test_split(original_indices, train_size=args.train_size, random_state=args.seed)

    # Spectra: global z-score fitted once on the initial training set (as in STM); errors are in real units.
    spec_norm = Norm_G(data=spectra_all[indices_train], axis=None)
    print(f"Spectra Norm_G: mean={float(spec_norm.mean):.4g}, std={float(spec_norm.std):.4g}")

    output_dict = {
        "reduced_errors": [], "reduced_errors_im2spec": [], "reduced_coords": [],
        "train_indices": [], "test_indices": [], "im2spec_test_mismatches": [],
        "pred_error_test": [], "predicted_errors_all": [],
        # constant over iterations (stored once)
        "point_coords": [coords], "recorded_coords": [data["coords"]], "scales": [np.array(all_scales)],
    }
    print(f"train {len(indices_train)}, test {len(indices_test)}; saving to {output_file}")

    for iteration in range(args.acquisition_iterations):
        if len(indices_test) == 0:
            print("No test indices remain. Stopping.")
            break
        print(f"\nActive Learning Iteration {iteration + 1}/{args.acquisition_iterations}")

        r_train, r_test = rows_of(indices_train, N, S), rows_of(indices_test, N, S)
        X_train, Y_train = X_all[r_train], Y_all[r_train]
        X_test, Y_test = X_all[r_test], Y_all[r_test]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(spec_norm.normalize(Y_train)))
        val_ds = TensorDataset(torch.tensor(X_test), torch.tensor(spec_norm.normalize(Y_test)))

        imspec_model = get_model(args.model_type, (args.WS, args.WS), spectra_all.shape[1], latent_dim=args.latent_dim,
                                 device=device, in_channels=2 if args.mask_channel else 1)
        imspec_model, _, _ = train_model(
            imspec_model, train_ds, n_batches=args.n_batches, lr=args.lr, patience=args.patience,
            n_epochs=args.n_epochs_im2spec, partial_train=False, val_dataset=val_ds,
            use_swa=args.use_swa, last_swa_epochs=args.last_swa_epochs)

        # Error model on the frozen encoder; target = z-scored log(error) of the training rows.
        err_train = spectra_errors(imspec_model, X_train, Y_train, spec_norm=spec_norm, error_type=args.error_type)
        error_norm = Norm_G(data=err_train, axis=None, log=True)
        error_ds = TensorDataset(torch.tensor(X_train),
                                 torch.tensor(error_norm.normalize(err_train).astype(np.float32).reshape(-1, 1)))

        with torch.no_grad():
            probe = torch.tensor(X_train[:1], dtype=torch.float32, device=device)
            if probe.ndim == 3:
                probe = probe.unsqueeze(1)
            encoder_latent_dim = imspec_model.encoder(probe).shape[1]
        decoder_cls = CustomDecoder_mask if args.mask_channel else CustomDecoder
        error_model = decoder_cls(encoder=imspec_model.encoder, embed_dim=encoder_latent_dim, target_size=1).to(device)
        error_model, _, _ = train_model(
            error_model, error_ds, n_batches=args.n_batches, lr=args.lr, patience=args.patience,
            n_epochs=args.n_epochs_error, partial_train=True, val_dataset=None,
            use_swa=args.use_swa, last_swa_epochs=args.last_swa_epochs)

        # Errors on all rows (all points x all scales); test rows are a subset.
        true_err_all = spectra_errors(imspec_model, X_all, Y_all, spec_norm=spec_norm, error_type=args.error_type).ravel()
        pred_err_all = predict_spectra(error_model, X_all, ensemble=False)
        pred_err_all = np.clip(error_norm.denormalize(pred_err_all), 0, None).ravel()

        output_dict["im2spec_test_mismatches"].append(true_err_all[r_test])
        output_dict["pred_error_test"].append(pred_err_all[r_test])

        reduced_errors, reduced_coords = reduce_errors_and_coords(pred_err_all, C_all, n_patches=N)
        reduced_errors_im2spec = true_err_all[find_row_indices(C_all, reduced_coords)]

        output_dict["reduced_errors"].append(reduced_errors)
        output_dict["reduced_errors_im2spec"].append(reduced_errors_im2spec)
        output_dict["reduced_coords"].append(reduced_coords)
        output_dict["train_indices"].append(indices_train.copy())
        output_dict["test_indices"].append(indices_test.copy())
        output_dict["predicted_errors_all"].append(pred_err_all)

        n_new = min(args.num_new_points, len(indices_test))
        if args.acquisition_method == "distance":
            aq_ind, _ = distance_acq_fn(reduced_errors, beta=1, lambda_=1, sample_next_points=n_new,
                                        exclude_indices=indices_train.reshape(-1))
        else:
            aq_ind = np.random.choice(indices_test, size=n_new, replace=False)
        aq_ind = np.asarray(aq_ind, dtype=int).reshape(-1)
        if np.isin(aq_ind, indices_train).any():
            raise RuntimeError("Acquired a point that is already in the training set.")

        indices_train = np.concatenate([indices_train, aq_ind]).ravel()
        indices_test = indices_test[~np.isin(indices_test, aq_ind)]

        test_full = true_err_all[r_test].reshape(S, -1)[-1]   # full-scale block of the test rows
        print(f"Train {len(indices_train)}, test {len(indices_test)}; acquired {aq_ind.tolist()}")
        print(f"Mean test {args.error_type} (full scale): {test_full.mean():.6g}; "
              f"mean reduced predicted error: {reduced_errors.mean():.6g}; "
              f"mean reduced im2spec error: {reduced_errors_im2spec.mean():.6g}")

        save_h5_ragged(output_file, output_dict)
        print(f"Checkpoint saved: {output_file}")

    save_h5_ragged(output_file, output_dict)
    print(f"\nDone. Final output saved to: {output_file}")


if __name__ == "__main__":
    main()
