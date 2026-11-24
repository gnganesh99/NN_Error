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
import random

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
    norm_0to1,
    predict_spectra,
    train_model,
)
from nnerror.utils.image_utils import append_multiscale_data, edges_zeroed_image  # noqa: E402
from atomai.nets.blocks import ConvBlock  # noqa: E402
from nnerror.utils.im2spec_dataset import paired_images_spectra_1  # noqa: E402
from stm_utils import Sxm_Image  # noqa: E402



# For training reproducibility
seed = 42
 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Hyperparameters from the notebook. You can edit these or override most of
# them from PowerShell with flags such as --acquisition-iterations 100.
WS = 32
WS_init = 4
d_WS = 4
n_epochs_im2spec = 100
n_epochs_error = 100
error_type = "L1"  # "L1", "L2", or "cos"
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
scale_type = "ss" # "ms"

each_spectra_norm = False
normalization = "gauss"  # "gauss" (z-score, see design_decisions.md) or "0to1" (previous behaviour)
mask_channel = True  # add a validity-mask image channel; only used for scale_type="ms" with "pad"

img_sxm_name = "EnZn2As2017.sxm"
spectra_sxm_name = "EuZn2As2003.3ds"

data_folder = STM_ROOT / "data" / "large_area"
output_folder = STM_ROOT / "data" / "active_learning_results1"


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


class Norm_G:
    """Gaussian (z-score) normalization: (x - mean) / std.

    axis=None uses one global mean/std; an integer axis gives per-feature statistics.
    With log=True the data is mapped to log(x + log_eps) before standardizing (for
    positive, skewed quantities such as errors), and denormalize inverts both steps.
    """

    def __init__(self, data=None, mean=None, std=None, axis=None, eps=1e-8, log=False, log_eps=None):
        self.data = data
        self.axis = axis
        self.eps = eps
        self.log = log

        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            if log and log_eps is None:
                # Small offset relative to the data scale so near-zero values don't blow up the log.
                scale = float(np.mean(np.clip(data, 0, None)))
                log_eps = 1e-3 * scale if scale > 0 else 1e-8
            self.log_eps = log_eps
            transformed = self._forward(data)
            keepdims = axis is not None
            self.mean = np.mean(transformed, axis=axis, keepdims=keepdims)
            self.std = np.std(transformed, axis=axis, keepdims=keepdims)
        elif mean is not None and std is not None:
            self.log_eps = log_eps if log_eps is not None else 1e-8
            self.mean = mean
            self.std = std
        else:
            raise ValueError("Either data or both mean and std must be provided.")

        self.std = np.maximum(self.std, eps)

    def _forward(self, data):
        if self.log:
            return np.log(np.clip(data, 0, None) + self.log_eps)
        return data

    def normalize(self, data=None):
        """Standardize data using the stored mean and std."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided to normalize.")
            data = self.data
        data = np.asarray(data, dtype=np.float64)
        return ((self._forward(data) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, normalized_data):
        """Map standardized values back to the original units."""
        data = np.asarray(normalized_data, dtype=np.float64) * self.std + self.mean
        if self.log:
            data = np.clip(np.exp(data) - self.log_eps, 0, None)
        return data.astype(np.float32)


def spectra_errors(model, images, spectra, spec_norm=None, error_type="L1"):
    """Per-sample im2spec error in the original spectra units.

    The model predicts normalized spectra; predictions are denormalized with
    `spec_norm` before comparing them with the original-unit `spectra`, so the
    recorded errors are comparable across normalization choices and iterations.
    """
    n = len(images)
    pred = np.asarray(predict_spectra(model, images, ensemble=False), dtype=np.float64).reshape(n, -1)
    if spec_norm is not None:
        pred = spec_norm.denormalize(pred).astype(np.float64)
    spectra = np.asarray(spectra, dtype=np.float64).reshape(n, -1)

    if error_type == "L1":
        error = np.mean(np.abs(pred - spectra), axis=-1)
    elif error_type == "L2":
        error = np.mean((pred - spectra) ** 2, axis=-1)
    elif error_type == "cos":
        dot = np.sum(pred * spectra, axis=-1)
        norms = np.linalg.norm(pred, axis=-1) * np.linalg.norm(spectra, axis=-1)
        error = 1 - dot / np.maximum(norms, 1e-8)
    else:
        raise ValueError("Invalid error_type. Valid: L1, L2, cos")

    return error.astype(np.float32)


class im2spec_mask(im2spec):
    """im2spec taking a 2-channel input (B, 2, H, W): padded image + validity mask.

    Identical to im2spec except the first conv block accepts `in_channels` inputs and
    forward() does not add a channel axis (the input already has one).
    """

    def __init__(self, *args, in_channels=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.enc_conv = ConvBlock(
            ndim=2, nb_layers=3,
            input_channels=in_channels, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True, dropout_=0.5,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CustomDecoder_mask(CustomDecoder):
    """CustomDecoder for multi-channel inputs (no extra channel axis added)."""

    def forward(self, x):
        return self.decoder(self.encoder(x))


_MASK_CACHE = {}


def scale_mask(scale, WS):
    """1 for pixels kept by edges_zeroed_image at this scale, 0 for padded pixels."""
    key = (int(scale), int(WS))
    if key not in _MASK_CACHE:
        ones = np.ones((WS, WS), dtype=np.float32)
        _MASK_CACHE[key] = ones if scale >= WS else edges_zeroed_image(ones, int(scale))
    return _MASK_CACHE[key]


def add_mask_channel(images_aug, coords_aug):
    """Stack (M, H, W) images with their scale masks into (M, 2, H, W).

    The mask is built from each row's scale (coords_aug[:, 2]), not from image == 0,
    because real pixels can be exactly 0 (0to1 norm) or the mean value (gauss norm).
    """
    WS = images_aug.shape[-1]
    masks = np.stack([scale_mask(scale, WS) for scale in coords_aug[:, 2]])
    return np.stack([images_aug.astype(np.float32), masks], axis=1)


def resolve_mask_channel(args, append_image_type):
    """Keep the mask channel only for im2spec + ms + pad; otherwise fall back to 1 channel."""
    if not args.mask_channel:
        return
    reasons = []
    if args.scale_type != "ms":
        reasons.append(f"scale_type={args.scale_type} (no padding)")
    if append_image_type != "pad":
        reasons.append(f"append_image_type={append_image_type}")
    if args.model_type != "im2spec":
        reasons.append(f"model_type={args.model_type} (mask only implemented for im2spec)")
    if reasons:
        print(f"Mask channel disabled ({', '.join(reasons)}); using single-channel images.")
        args.mask_channel = False
    else:
        print("Mask channel enabled: model input is (image, validity mask).")


def _change_WS_init_for_scale_type(args):
    if args.scale_type == "ss":  # No augmentation for single-scale
        args.WS_init = args.WS


def get_model(model_type, in_dim, out_dim, latent_dim=3, device=None, in_channels=1):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if in_channels != 1:
        if model_type != "im2spec":
            raise ValueError(f"in_channels={in_channels} is only supported for im2spec.")
        return im2spec_mask(
            feature_size=in_dim, target_size=out_dim, latent_dim=latent_dim, in_channels=in_channels
        ).to(device)

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
    mask_channel=False,
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

    if mask_channel:
        images_aug = add_mask_channel(images_aug, coords_aug)

    return images_aug, spectra_aug, coords_aug


def reduce_errors_and_coords(predicted_errors, scale_coordinates, n_patches):
    """Minimum predicted error over scales for each patch, returned in patch-index order.

    `scale_coordinates` comes from `append_multiscale_data` on all patches, which stacks
    one block of `n_patches` rows per scale, so row r belongs to patch `r % n_patches`.
    Position i of the returned arrays is patch i, so acquisition indices taken from
    them can be used directly as indices into `images_all` / `train_indices`.
    """
    predicted_errors = np.asarray(predicted_errors).reshape(-1)
    if len(predicted_errors) != len(scale_coordinates) or len(predicted_errors) % n_patches != 0:
        raise ValueError(
            f"Expected a multiple of {n_patches} rows, got {len(predicted_errors)} errors "
            f"and {len(scale_coordinates)} coordinates."
        )

    df_errors = pd.DataFrame(
        {
            "x": scale_coordinates[:, 0],
            "y": scale_coordinates[:, 1],
            "scale": scale_coordinates[:, 2],
            "error": predicted_errors,
            "patch": np.arange(len(predicted_errors)) % n_patches,
        }
    )

    reduced = (
        df_errors.loc[df_errors.groupby("patch")["error"].idxmin()]
        .sort_values("patch")
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
    #parser.add_argument("--no-use-swa", dest="use_swa", action="store_false")
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
    parser.add_argument("--each-spectra-norm", action='store_true', default=each_spectra_norm)
    parser.add_argument("--scale-type", choices=("ss", "ms"), default=scale_type)
    parser.add_argument("--norm", choices=("gauss", "0to1"), default=normalization)
    parser.add_argument(
        "--mask-channel", action=argparse.BooleanOptionalAction, default=mask_channel,
        help="Add a validity-mask image channel (only for --scale-type ms with pad; otherwise ignored).",
    )
    return parser.parse_args()


def build_output_file(args, chosen_append_image_type):
    if args.output_file is not None:
        return args.output_file

    filename = (
        f"al_{args.model_type}_{args.error_type}_WS{args.WS}_"
        f"latent{args.latent_dim}_append{chosen_append_image_type}_"
        f"{args.acquisition_iterations}iters_{args.num_new_points}pts_"
        f"{args.acquisition_method}acq_{args.scale_type}_norm{args.norm}"
        f"{'_mask' if args.mask_channel else ''}.h5"
    )
    return args.output_folder / filename


def main():
    args = parse_args()
    set_random_seed(args.seed)

    chosen_append_image_type = args.append_image_type or default_append_image_type(args.model_type)
    _change_WS_init_for_scale_type(args)
    resolve_mask_channel(args, chosen_append_image_type)
    output_file = build_output_file(args, chosen_append_image_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {args.data_folder}")

    img_sxm = args.data_folder / args.img_sxm_name
    spectra_sxm = args.data_folder / args.spectra_sxm_name

    full_image = Sxm_Image(img_sxm).image()
    if args.norm == "gauss":
        # Global z-score over the whole image (the image is known everywhere, so no leakage).
        full_image = Norm_G(data=full_image, axis=None).normalize()
    else:
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
        spectra_norm=args.each_spectra_norm,
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

    # spectra_all stays in the original (0-1) units. In "gauss" mode the model trains on
    # globally z-scored spectra, fitted once on the initial training set and kept fixed,
    # and every recorded error is computed after denormalizing back to original units.
    if args.norm == "gauss":
        spec_norm = Norm_G(data=spectra_all[indices_train], axis=None)
        print(f"Spectra Norm_G: mean={float(spec_norm.mean):.4g}, std={float(spec_norm.std):.4g}")
    else:
        spec_norm = None

    def to_model_spectra(spectra_values):
        return spectra_values if spec_norm is None else spec_norm.normalize(spectra_values)

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
            mask_channel=args.mask_channel,
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
            mask_channel=args.mask_channel,
        )

        print(
            "Current augmented train data: "
            f"{current_images_train.shape[0]}, current augmented test data: {images_test_aug.shape[0]}"
        )

        im2spec_al_training_dataset = TensorDataset(
            torch.tensor(current_images_train, dtype=torch.float32),
            torch.tensor(to_model_spectra(current_spectra_train), dtype=torch.float32),
        )

        im2spec_al_val_dataset = TensorDataset(
            torch.tensor(images_test_aug, dtype=torch.float32),
            torch.tensor(to_model_spectra(spectra_test_aug), dtype=torch.float32),
        )

        imspec_model = get_model(
            args.model_type,
            in_dim,
            out_dim,
            latent_dim=args.latent_dim,
            device=device,
            in_channels=2 if args.mask_channel else 1,
        )

        imspec_model, al_im2spec_train_loss, al_im2spec_val_loss = train_model(
            imspec_model,
            im2spec_al_training_dataset,
            n_batches=args.n_batches,
            lr=args.lr,
            patience=args.patience,
            n_epochs=args.n_epochs_im2spec,
            partial_train=False,
            val_dataset=im2spec_al_val_dataset,
            use_swa=args.use_swa,
            last_swa_epochs=args.last_swa_epochs,
        )

        error_mean_al = spectra_errors(
            imspec_model,
            current_images_train,
            current_spectra_train,
            spec_norm=spec_norm,
            error_type=args.error_type,
        )
        if args.norm == "gauss":
            # Errors are positive and skewed: z-score log(error); denormalize maps back to real errors.
            error_norm = Norm_G(data=error_mean_al, axis=None, log=True)
        else:
            error_norm = Norm_0to1(data=error_mean_al)
        error_mean_al = error_norm.normalize(error_mean_al)
        error_targets_al = error_mean_al.astype(np.float32).reshape(-1, 1)

        model_device = next(imspec_model.parameters()).device
        with torch.no_grad():
            probe = torch.tensor(current_images_train[:1], dtype=torch.float32, device=model_device)
            if probe.ndim == 3:  # single-channel (B, H, W) -> (B, 1, H, W)
                probe = probe.unsqueeze(1)
            encoder_latent_dim = imspec_model.encoder(probe).shape[1]

        decoder_cls = CustomDecoder_mask if args.mask_channel else CustomDecoder
        error_model = decoder_cls(
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

        error_mean_test_al = spectra_errors(
            imspec_model,
            images_test_aug,
            spectra_test_aug,
            spec_norm=spec_norm,
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
            mask_channel=args.mask_channel,
        )

        error_im2spec_al = spectra_errors(
            imspec_model,
            images_all_aug,
            spectra_all_aug,
            spec_norm=spec_norm,
            error_type=args.error_type,
        )
        error_im2spec_al = error_im2spec_al.flatten()

        predicted_errors_current_iter = predict_spectra(error_model, images_all_aug, ensemble=False)
        predicted_errors_current_iter = error_norm.denormalize(predicted_errors_current_iter)
        predicted_errors_current_iter = np.clip(predicted_errors_current_iter, 0, None)

        reduced_errors, reduced_coords = reduce_errors_and_coords(
            predicted_errors_current_iter,
            scale_coordinates,
            n_patches=len(original_indices),
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
