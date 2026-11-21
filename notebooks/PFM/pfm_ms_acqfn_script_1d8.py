#!/usr/bin/env python3
"""
Notebook-style BEPS active learning script:
- Loads FORC_BEPS_1d8um_0010.h5
- Prepares image/spectra pairs
- Runs iterative active learning loop
- Saves results to HDF5
- Plots acquired points
"""

import sys
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from nnerror.networks import FNO_im2spec, CustomDecoder, im2spec, im2spec_attn, FNO_im2spec_attn
from nnerror.training_functions import err_estimation, train_model, predict_spectra, norm_0to1, predict_posterior, append_training_set
from nnerror.plot_functions import plot_only_training_loss, plot_training_loss, \
    plot_error_prediction, plot_spectra, plot_error_prediction_3d, plot_scale_slider, plot_latent_space, cluster_latent_space, plot_latent_distribution, plot_scale_specific_error_and_acquisition_maps
from nnerror.utils import append_multiscale_data, edges_zeroed_image, interpolated_center_crop, paired_images_spectra_1
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--acquisition_iterations", type=int, default=5)
parser.add_argument("--num_new_points", type=int, default=50)
parser.add_argument("--model_type", type=str, default="im2spec")
parser.add_argument("--latent_dim", type=int, default=5)
parser.add_argument("--n_batches", type=int, default=6)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n_epochs_im2spec", type=int, default=50)
parser.add_argument("--n_epochs_error", type=int, default=50)
parser.add_argument("--use_swa", action="store_true")
parser.add_argument("--last_swa_epochs", type=float, default=0.1)
parser.add_argument("--error_type", type=str, default="L1")
parser.add_argument("--acquisition_method", type=str, default="distance")
parser.add_argument("--WS_init", type=int, default=4)
parser.add_argument("--WS", type=int, default=24)
parser.add_argument("--d_WS", type=int, default=4)
args = parser.parse_args()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
class Norm_0to1:
    def __init__(self, data):
        self.min_val = np.min(data)
        self.max_val = np.max(data)
        self.data_range = max(self.max_val - self.min_val, 1e-8)
    def normalize(self, data):
        return (data - self.min_val) / self.data_range
    def denormalize(self, normalized_data):
        return normalized_data * self.data_range + self.min_val

def save_h5_ragged(filename, data_dict):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as f:
        for key, value_list in data_dict.items():
            group = f.create_group(key)
            for i, value in enumerate(value_list):
                group.create_dataset(str(i), data=np.asarray(value))

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset (assume already downloaded)
    beps_path = REPO_ROOT / "data" / "BEPS" / "FORC_BEPS_1d8um_0010.h5"
    h5_f = h5py.File(str(beps_path), "r")
    sho_mat = h5_f["Measurement_000/Channel_000/Raw_Data-SHO_Fit_000/Guess"]
    spec_val = h5_f["Measurement_000/Channel_000/Raw_Data-SHO_Fit_000/Spectroscopic_Values"]
    pos_inds = h5_f["Measurement_000/Channel_000/Position_Indices"]

    pos_dim_sizes = [np.max(pos_inds[:, 0]) + 1, np.max(pos_inds[:, 1]) + 1]
    sho_mat_ndim = sho_mat[:].reshape(pos_dim_sizes[0], pos_dim_sizes[1], -1)

    amp_mat_ndim = sho_mat_ndim["Amplitude [V]"]
    phase_mat_ndim = sho_mat_ndim["Phase [rad]"]

    min_, max_ = np.min(phase_mat_ndim), np.max(phase_mat_ndim)
    pha_correct = np.where(phase_mat_ndim > 1.5, phase_mat_ndim + (min_ - max_), phase_mat_ndim)

    full_spec_len = amp_mat_ndim.shape[2]
    spec_len = full_spec_len // 2
    amp_off = np.zeros((pos_dim_sizes[0], pos_dim_sizes[1], spec_len))
    pha_off = np.zeros_like(amp_off)
    for i in range(spec_len):
        amp_off[:, :, i] = amp_mat_ndim[:, :, 2*i+1]
        pha_off[:, :, i] = pha_correct[:, :, 2*i+1]
    pola_off = amp_off * np.cos(pha_off)

    topo = np.mean(amp_mat_ndim, axis=2)

    # Build dataset
    struc_img = amp_off.mean(axis=2)
    window_size = args.WS
    half = window_size // 2
    images, spectra, coords = [], [], []
    for x in range(half, struc_img.shape[0]-half, 1):
        for y in range(half, struc_img.shape[1]-half, 1):
            patch = struc_img[x-half:x+half, y-half:y+half]
            spectrum = pola_off[x, y, :]
            images.append(patch)
            spectra.append(spectrum)
            coords.append([x, y])
    images_all = (np.array(images) - np.min(images)) / (np.max(images) - np.min(images))
    spectra_all = np.array(spectra)
    coords_all = np.array(coords)[:, [1, 0]]

    indices = np.arange(len(images_all))
    indices_train, indices_test = train_test_split(indices, train_size=0.1, random_state=42)
    original_indices = indices.copy()
    swap_coordinates_all = coords_all.copy()

    output_dict = {
        "train_indices": [indices_train.copy()],
        "test_indices": [indices_test.copy()],
        "reduced_coords": [],
        "reduced_errors": [],
        "reduced_errors_im2spec": [],
        "predicted_errors_all": [],
        "pred_error_test": [],
        "im2spec_test_mismatches": [],
    }

    chosen_append_image_type = "pad"

    # -------------------------
    # Active Learning Loop
    # -------------------------
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

        im2spec_al_training_dataset = TensorDataset(
            torch.tensor(current_images_train, dtype=torch.float32),
            torch.tensor(current_spectra_train, dtype=torch.float32),
        )
        im2spec_al_val_dataset = TensorDataset(
            torch.tensor(images_test_aug, dtype=torch.float32),
            torch.tensor(spectra_test_aug, dtype=torch.float32),
        )

        imspec_model = im2spec(feature_size=images_all[0].shape,
                               target_size=len(spectra_all[0]),
                               latent_dim=args.latent_dim).to(device)

        imspec_model, _, _ = train_model(
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

        error_mean_al, _, _ = err_estimation(
            imspec_model,
            current_images_train,
            current_spectra_train,
            error_type=args.error_type,
        )
        error_norm = Norm_0to1(data=error_mean_al)
        error_targets_al = error_norm.normalize(error_mean_al).astype(np.float32).reshape(-1, 1)

        with torch.no_grad():
            probe = torch.tensor(images_all[:1], dtype=torch.float32, device=device).unsqueeze(1)
            encoder_latent_dim = imspec_model.encoder(probe).shape[1]

        error_model = CustomDecoder(
            encoder=imspec_model.encoder,
            embed_dim=encoder_latent_dim,
            latent_dim=args.latent_dim,
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

        # Evaluate errors on test set
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

        # Predict errors on all points
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

        # Acquisition step
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

    # -------------------------
    # Save results
    # -------------------------
    output_folder = REPO_ROOT / "data" / "beps_active_learning"
    output_file = output_folder / "beps_results.h5"
    save_h5_ragged(output_file, output_dict)
    print(f"\nDone. Final output saved to: {output_file}")

    # -------------------------
    # Plot acquired points
    # -------------------------
    initial_train_indices = output_dict["train_indices"][0]
    final_train_indices = output_dict["train_indices"][-1]
    acquired_indices = np.setdiff1d(final_train_indices, initial_train_indices)
    reduced_coords = output_dict["reduced_coords"][-1]
    acquired_coords = reduced_coords[np.isin(reduced_coords, coords_all[acquired_indices]).all(axis=1)]

    plt.figure(figsize=(7, 7))
    plt.imshow(topo, cmap="gray", origin="lower")
    plt.scatter(
        acquired_coords[:, 1],
        acquired_coords[:, 0],
        c="red",
        s=40,
        edgecolors="white",
        linewidths=0.8,
        label="Acquired points"
    )
    plt.legend()
    plt.title("Active Learning Acquired Points")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
