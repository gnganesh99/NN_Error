"""Run the STM multiscale active-learning acquisition loop.

This is a script version of the code before the "Other analysis" section in
stm_ms_acqfn1.ipynb. It skips notebook plotting/analysis and saves output_dict
automatically as an HDF5 file.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))


def find_repo_root(start: Path) -> Path:
    """Find the repository root from this script location."""
    for path in (start, *start.parents):
        if (path / "src" / "nnerror").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(SCRIPT_DIR)
SRC_ROOT = REPO_ROOT / "src"
STM_ROOT = REPO_ROOT / "notebooks" / "STM"
NNERROR_ROOT = SRC_ROOT / "nnerror"

for import_root in (SRC_ROOT, STM_ROOT):
    import_root_str = str(import_root.resolve())
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


def register_nnerror_namespace():
    """Load nnerror submodules without executing broad package __init__ imports."""
    package_paths = {
        "nnerror": NNERROR_ROOT,
        "nnerror.networks": NNERROR_ROOT / "networks",
        "nnerror.utils": NNERROR_ROOT / "utils",
    }
    for package_name, package_path in package_paths.items():
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package


register_nnerror_namespace()

from CITS_Class import CITS_Analysis  # noqa: E402
from nnerror.networks.im2spec_models import CustomDecoder, im2spec  # noqa: E402
from nnerror.training_functions import (  # noqa: E402
    distance_acq_fn,
    err_estimation,
    norm_0to1,
    predict_spectra,
    train_model,
)
from nnerror.utils.image_utils import append_multiscale_data  # noqa: E402
from nnerror.utils.im2spec_dataset import paired_images_spectra_1  # noqa: E402
from stm_utils import Sxm_Image  # noqa: E402


# Hyperparameters from the notebook. You can edit these or override most of
# them from PowerShell with flags such as --acquisition-iterations 100.
WS = 32
WS_init = 32 # to prevent augmenting with smaller windows, set WS_init = WS
d_WS = 4
n_epochs_im2spec = 100
n_epochs_error = 100
error_type = "cos"  # "L1", "L2", or "cos"
model_type = "im2spec"
append_image_type = None  # None chooses based on model_type below.
use_swa = True
latent_dim = 5
last_swa_epochs = 0.1

acquisition_iterations = 50
num_new_points = 5
acquisition_method = "distance"  # "distance" or "random"

coordinate_step = 15
train_size = 0.1
n_batches = 6
lr = 1e-3
patience = 10
seed = 42

img_sxm_name = "EnZn2As2017.sxm"
spectra_sxm_name = "EuZn2As2003.3ds"

data_folder = STM_ROOT / "data" / "large_area"
output_folder = STM_ROOT / "data" / "active_learning_results"


class Norm_0to1:
    """Normalization to the range [0, 1]."""

    def __init__(self, data=None, min_val=None, max_val=None, axis=0, eps=1e-8):
        self.data = data
        self.axis = axis
        self.eps = eps

        if data is not None:
            self.min_val = np.min(data, axis=axis, keepdims=True)
            self.max_val = np.max(data, axis=axis, keepdims=True)
        elif min_val is not None and max_val is not None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            raise ValueError("Either data or both min_val and max_val must be provided.")

        self.data_range = np.where(
            (self.max_val - self.min_val) < eps,
            eps,
            self.max_val - self.min_val,
        )

    def normalize(self, data=None):
        """Normalize data using stored min and max."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided to normalize.")
            data = self.data

        return (data - self.min_val) / self.data_range

    def denormalize(self, normalized_data):
        """Denormalize data using stored min and max."""
        return normalized_data * self.data_range + self.min_val


def get_model(model_type, in_dim, out_dim, latent_dim=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "im2spec":
        model = im2spec(feature_size=in_dim, target_size=out_dim, latent_dim=latent_dim).to(device)
    elif model_type == "FNO":
        fno_module = importlib.import_module("nnerror.networks.neuralop_im2spec")
        FNO_im2spec = fno_module.FNO_im2spec
        model = FNO_im2spec(
            target_size=out_dim,
            latent_dim=latent_dim,
            hidden_channels=32,
            n_modes=(16, 16),
            n_layers=4,
        ).to(device)
    elif model_type == "im2spec_attn":
        attn_module = importlib.import_module("nnerror.networks.attn_models")
        im2spec_attn = attn_module.im2spec_attn
        model = im2spec_attn(feature_size=in_dim, target_size=out_dim, latent_dim=latent_dim).to(device)
    elif model_type == "FNO_attn":
        attn_module = importlib.import_module("nnerror.networks.attn_models")
        FNO_im2spec_attn = attn_module.FNO_im2spec_attn
        model = FNO_im2spec_attn(
            target_size=out_dim,
            latent_dim=latent_dim,
            hidden_channels=32,
            n_modes=(16, 16),
            n_layers=4,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def default_append_image_type(model_type):
    if model_type in ("im2spec", "im2spec_attn"):
        return "pad"
    return "interpolate"


def augment_images_by_indices(
    indices,
    images,
    spectra,
    coordinates,
    WS_init,
    WS,
    d_WS,
    append_image_type=None,
):
    images = images[indices]
    spectra = spectra[indices]
    coordinates = coordinates[indices]

    if append_image_type is None:
        raise ValueError("append_image_type must be 'pad' or 'interpolate'.")

    images_aug, spectra_aug, coords_aug = append_multiscale_data(
        images,
        spectra,
        scales=np.arange(WS_init, WS, d_WS),
        coordinates=coordinates,
        append_image_type=append_image_type,
    )

    return images_aug, spectra_aug, coords_aug


def reduce_errors_and_coords(predicted_errors, scale_coordinates):
    predicted_errors = np.asarray(predicted_errors).reshape(-1)
    df_errors = pd.DataFrame(
        {
            "x": scale_coordinates[:, 0],
            "y": scale_coordinates[:, 1],
            "scale": scale_coordinates[:, 2],
            "error": predicted_errors,
        }
    )

    reduced = (
        df_errors.loc[df_errors.groupby(["x", "y"])["error"].idxmin()]
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )

    errors = reduced["error"].to_numpy()
    coords_reduced = reduced[["x", "y", "scale"]].to_numpy()

    return errors, coords_reduced


def find_row_indices(original, subset):
    row_to_index = {tuple(row): i for i, row in enumerate(original)}
    return np.array([row_to_index[tuple(row)] for row in subset])


def save_h5_ragged(filename, data_dict):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "w") as f:
        for key, value_list in data_dict.items():
            group = f.create_group(key)

            for i, value in enumerate(value_list):
                group.create_dataset(str(i), data=np.asarray(value))


def set_random_seed(seed_value):
    if seed_value is None:
        return
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run STM multiscale active-learning acquisition and save output_dict.",
    )
    parser.add_argument("--WS", type=int, default=WS)
    parser.add_argument("--WS-init", dest="WS_init", type=int, default=WS_init)
    parser.add_argument("--d-WS", dest="d_WS", type=int, default=d_WS)
    parser.add_argument("--n-epochs-im2spec", type=int, default=n_epochs_im2spec)
    parser.add_argument("--n-epochs-error", type=int, default=n_epochs_error)
    parser.add_argument("--error-type", choices=("L1", "L2", "cos"), default=error_type)
    parser.add_argument("--model-type", default=model_type)
    parser.add_argument("--append-image-type", choices=("pad", "interpolate"), default=append_image_type)
    parser.add_argument("--use-swa", dest="use_swa", action="store_true", default=use_swa)
    parser.add_argument("--no-use-swa", dest="use_swa", action="store_false")
    parser.add_argument("--latent-dim", type=int, default=latent_dim)
    parser.add_argument("--last-swa-epochs", type=float, default=last_swa_epochs)
    parser.add_argument("--acquisition-iterations", type=int, default=acquisition_iterations)
    parser.add_argument("--num-new-points", type=int, default=num_new_points)
    parser.add_argument("--acquisition-method", choices=("distance", "random"), default=acquisition_method)
    parser.add_argument("--coordinate-step", type=int, default=coordinate_step)
    parser.add_argument("--train-size", type=float, default=train_size)
    parser.add_argument("--n-batches", type=int, default=n_batches)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--data-folder", type=Path, default=data_folder)
    parser.add_argument("--output-folder", type=Path, default=output_folder)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--img-sxm-name", default=img_sxm_name)
    parser.add_argument("--spectra-sxm-name", default=spectra_sxm_name)
    return parser.parse_args()


def build_output_file(args, chosen_append_image_type):
    if args.output_file is not None:
        return args.output_file

    filename = (
        f"al_{args.model_type}_{args.error_type}_WS{args.WS}_"
        f"latent{args.latent_dim}_append{chosen_append_image_type}_"
        f"{args.acquisition_iterations}iters_{args.num_new_points}pts_"
        f"{args.acquisition_method}acq.h5"
    )
    return args.output_folder / filename


def main():

    args = parse_args()
    set_random_seed(args.seed)

    chosen_append_image_type = args.append_image_type or default_append_image_type(args.model_type)
    output_file = build_output_file(args, chosen_append_image_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {args.data_folder}")

    img_sxm = args.data_folder / args.img_sxm_name
    spectra_sxm = args.data_folder / args.spectra_sxm_name

    full_image = Sxm_Image(img_sxm).image()
    full_image = norm_0to1(full_image)

    cits_obj = CITS_Analysis(str(spectra_sxm))
    hyperspectra = cits_obj.current

    images, spectra, coordinates = paired_images_spectra_1(
        full_image,
        cits_obj,
        hyperspectra,
        window_size=args.WS,
        coordinate_step=args.coordinate_step,
        image_norm=False,
        spectra_norm=False,
    )

    print(f"images: {images.shape}, spectra: {spectra.shape}, coordinates: {coordinates.shape}")

    in_dim = images[0].shape
    out_dim = len(spectra[0])

    swap_coordinates = coordinates.copy()
    swap_coordinates = swap_coordinates[:, [1, 0]]

    images_all = images.copy()
    spectra_all = spectra.copy()
    swap_coordinates_all = swap_coordinates.copy()

    original_indices = np.arange(len(images_all))
    indices_train, indices_test = train_test_split(
        original_indices,
        train_size=args.train_size,
        random_state=args.seed,
    )

    output_dict = {
        "reduced_errors": [],
        "reduced_errors_im2spec": [],
        "reduced_coords": [],
        "train_indices": [],
        "test_indices": [],
        "im2spec_test_mismatches": [],
        "pred_error_test": [],
        "predicted_errors_all": [],
    }

    print(f"len(images): {len(images)}")
    print(f"len(indices_train): {len(indices_train)}")
    print(f"len(indices_test): {len(indices_test)}")
    print(f"Saving output_dict to: {output_file}")

    for iteration in range(args.acquisition_iterations):
        if len(indices_test) == 0:
            print("No test indices remain. Stopping acquisition loop.")
            break

        print(f"\nActive Learning Iteration {iteration + 1}/{args.acquisition_iterations}")

        current_images_train, current_spectra_train, coordinates_train = augment_images_by_indices(
            indices_train,
            images_all,
            spectra_all,
            swap_coordinates_all,
            args.WS_init,
            args.WS,
            args.d_WS,
            append_image_type=chosen_append_image_type,
        )
        images_test_aug, spectra_test_aug, coordinates_test = augment_images_by_indices(
            indices_test,
            images_all,
            spectra_all,
            swap_coordinates_all,
            args.WS_init,
            args.WS,
            args.d_WS,
            append_image_type=chosen_append_image_type,
        )

        print(
            "Current augmented train data: "
            f"{current_images_train.shape[0]}, current augmented test data: {images_test_aug.shape[0]}"
        )

        im2spec_al_training_dataset = TensorDataset(
            torch.tensor(current_images_train, dtype=torch.float32),
            torch.tensor(current_spectra_train, dtype=torch.float32),
        )

        im2spec_al_val_dataset = TensorDataset(
            torch.tensor(images_test_aug, dtype=torch.float32),
            torch.tensor(spectra_test_aug, dtype=torch.float32),
        )

        imspec_model = get_model(
            args.model_type,
            in_dim,
            out_dim,
            latent_dim=args.latent_dim,
            device=device,
        )

        imspec_model, al_im2spec_train_loss, al_im2spec_val_loss = train_model(
            imspec_model,
            im2spec_al_training_dataset,
            n_batches=args.n_batches,
            lr=args.lr,
            patience=args.patience,
            n_epochs=args.n_epochs_im2spec,
            partial_train=False,
            val_dataset=None,
            use_swa=args.use_swa,
            last_swa_epochs=args.last_swa_epochs,
        )

        error_mean_al, _, _ = err_estimation(
            imspec_model,
            current_images_train,
            current_spectra_train,
            error_type=args.error_type,
        )
        error_norm = Norm_0to1(data=error_mean_al)
        error_mean_al = error_norm.normalize(error_mean_al)
        error_targets_al = error_mean_al.astype(np.float32).reshape(-1, 1)

        model_device = next(imspec_model.parameters()).device
        with torch.no_grad():
            probe = torch.tensor(images[:1], dtype=torch.float32, device=model_device).unsqueeze(1)
            encoder_latent_dim = imspec_model.encoder(probe).shape[1]

        error_model = CustomDecoder(
            encoder=imspec_model.encoder,
            embed_dim=encoder_latent_dim,
            target_size=1,
        ).to(device)

        error_al_training_dataset = TensorDataset(
            torch.tensor(current_images_train, dtype=torch.float32),
            torch.tensor(error_targets_al, dtype=torch.float32),
        )

        error_model, al_error_train_loss, al_error_val_loss = train_model(
            error_model,
            error_al_training_dataset,
            n_batches=args.n_batches,
            lr=args.lr,
            patience=args.patience,
            n_epochs=args.n_epochs_error,
            partial_train=True,
            val_dataset=None,
            use_swa=args.use_swa,
            last_swa_epochs=args.last_swa_epochs,
        )

        error_mean_test_al, _, _ = err_estimation(
            imspec_model,
            images_test_aug,
            spectra_test_aug,
            error_type=args.error_type,
        )
        output_dict["im2spec_test_mismatches"].append(error_mean_test_al.flatten())

        predicted_errors_test = predict_spectra(error_model, images_test_aug, ensemble=False)
        predicted_errors_test = error_norm.denormalize(predicted_errors_test)
        predicted_errors_test = np.clip(predicted_errors_test, 0, None)
        output_dict["pred_error_test"].append(predicted_errors_test.flatten())

        images_all_aug, spectra_all_aug, scale_coordinates = augment_images_by_indices(
            original_indices,
            images_all,
            spectra_all,
            swap_coordinates_all,
            args.WS_init,
            args.WS,
            args.d_WS,
            append_image_type=chosen_append_image_type,
        )

        error_im2spec_al, _, _ = err_estimation(
            imspec_model,
            images_all_aug,
            spectra_all_aug,
            error_type=args.error_type,
        )
        error_im2spec_al = error_im2spec_al.flatten()

        predicted_errors_current_iter = predict_spectra(error_model, images_all_aug, ensemble=False)
        predicted_errors_current_iter = error_norm.denormalize(predicted_errors_current_iter)
        predicted_errors_current_iter = np.clip(predicted_errors_current_iter, 0, None)

        reduced_errors, reduced_coords = reduce_errors_and_coords(
            predicted_errors_current_iter,
            scale_coordinates,
        )
        reduced_errors_im2spec = error_im2spec_al[find_row_indices(scale_coordinates, reduced_coords)]

        output_dict["reduced_errors"].append(reduced_errors)
        output_dict["reduced_errors_im2spec"].append(reduced_errors_im2spec)
        output_dict["reduced_coords"].append(reduced_coords)
        output_dict["train_indices"].append(indices_train.copy())
        output_dict["test_indices"].append(indices_test.copy())
        output_dict["predicted_errors_all"].append(predicted_errors_current_iter.flatten())

        points_to_acquire = min(args.num_new_points, len(indices_test))
        if args.acquisition_method == "distance":
            aq_ind, aq_vals = distance_acq_fn(
                reduced_errors,
                beta=1,
                lambda_=1,
                sample_next_points=points_to_acquire,
                exclude_indices=indices_train.reshape(-1),
            )
        elif args.acquisition_method == "random":
            aq_ind = np.random.choice(indices_test, size=points_to_acquire, replace=False)
        else:
            raise ValueError(f"Unknown acquisition method: {args.acquisition_method}")

        aq_ind = np.asarray(aq_ind, dtype=int).reshape(-1)
        indices_train = np.concatenate([indices_train, aq_ind]).ravel()
        indices_test = indices_test[~np.isin(indices_test, aq_ind)]

        print(f"Train indices set size: {len(indices_train)}, test indices set size: {len(indices_test)}")
        print(
            "Mean reduced predicted error: "
            f"{np.mean(reduced_errors):.6g}; mean reduced im2spec error: {np.mean(reduced_errors_im2spec):.6g}"
        )

        save_h5_ragged(output_file, output_dict)
        print(f"Checkpoint saved: {output_file}")

    save_h5_ragged(output_file, output_dict)
    print(f"\nDone. Final output saved to: {output_file}")


if __name__ == "__main__":
    main()
