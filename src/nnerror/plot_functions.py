import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_error_prediction(error_mean, aq_fn, coordinates, ind, expt_name = 'test_expt', iter_nb = 0, save_folder = r"data", to_save = False):
    """
    Plot 2D maps of predicted error and acquisition values.

    Args:
        error_mean: Flat array of error values. Length must form a square map.
        aq_fn: Flat array of acquisition values matching `error_mean`.
        coordinates: Candidate coordinates. Kept for API consistency; map
            positions are inferred from square-grid ordering.
        ind: Candidate index or indices to highlight on the error map.
        expt_name: Subfolder name used when saving.
        iter_nb: Iteration number included in the saved filename.
        save_folder: Root folder for saved figures.
        to_save: If True, save the figure before showing it.
    """

    n_dim = int(len(error_mean)**0.5)
    error_mean = error_mean.reshape(n_dim, n_dim)

    aq_fn = aq_fn.reshape(n_dim, n_dim)

    x = np.linspace(0, n_dim-1, n_dim)

    X = np.asarray(np.meshgrid(x, x)).T.reshape(-1, 2)



    fig, ax = plt.subplots(1, 2, figsize = (8, 3))

    a = ax[0].imshow(error_mean, origin = 'lower')
    ax[0].set_title('Error Map')
    ax[0].scatter(X[ind, 1], X[ind, 0], c = 'r', s = 15)  #sample along the index of the other axis. this is counter-intutive!!!

    b = ax[1].imshow(aq_fn, vmin =0, origin = 'lower')
    ax[1].set_title('Acquisition Function')

    fig.colorbar(a, ax=ax[0], fraction=0.05, pad=0.04)
    fig.colorbar(b, ax=ax[1], fraction=0.05, pad=0.04)


    if to_save:
        img_name = "errormap_iter"+str(iter_nb)+'.jpg'

        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)

        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')

    plt.show()




def plot_error_prediction_3d(
    error_mean,
    aq_fn,
    coordinates,
    ind,
    expt_name="test_expt",
    iter_nb=0,
    save_folder=r"data",
    to_save=False,
    alpha_scale=True,
    colormap="viridis",
):
    """
    Plot 3D scatter maps of predicted error and acquisition values.

    Args:
        error_mean: Flat array of error values.
        aq_fn: Flat array of acquisition values.
        coordinates: Either `(nx, ny, nz, 3)` gridded coordinates or `(N, 3)`
            flat coordinates. `error_mean` and `aq_fn` must match the flattened
            coordinate order.
        ind: Candidate index or indices selected for acquisition. Currently
            retained for API consistency.
        expt_name: Subfolder name used when saving.
        iter_nb: Iteration number included in the saved filename.
        save_folder: Root folder for saved figures.
        to_save: If True, save the figure before showing it.
        alpha_scale: If True, scale point alpha by normalized values.
        colormap: Matplotlib colormap name.

    Raises:
        ValueError: If coordinate dimensions or flattened lengths are invalid.
    """

    coordinates = np.asarray(coordinates)

    # ---------------------------------------------------------------
    # 1. Get per-axis sizes from `coordinates` -- no cubic assumption.
    # ---------------------------------------------------------------
    if coordinates.ndim == 4:
        nx, ny, nz, _ = coordinates.shape
        coords_flat = coordinates.reshape(-1, 3)
    elif coordinates.ndim == 2:
        nx = len(np.unique(coordinates[:, 0]))
        ny = len(np.unique(coordinates[:, 1]))
        nz = len(np.unique(coordinates[:, 2]))
        coords_flat = coordinates
    else:
        raise ValueError(
            f"coordinates must be 2D (N, 3) or 4D (nx, ny, nz, 3), "
            f"got shape {coordinates.shape}"
        )

    n_total = nx * ny * nz
    if len(error_mean) != n_total or len(aq_fn) != n_total:
        raise ValueError(
            f"Length mismatch: coordinates imply {n_total} points "
            f"(nx={nx}, ny={ny}, nz={nz}), but got "
            f"error_mean of length {len(error_mean)} and "
            f"aq_fn of length {len(aq_fn)}."
        )

    err_flat = np.asarray(error_mean).ravel()
    aq_flat  = np.asarray(aq_fn).ravel()

    xs = coords_flat[:, 0]
    ys = coords_flat[:, 1]
    zs = coords_flat[:, 2]

    # ---------------------------------------------------------------
    # 2. Helper: build per-point RGBA so alpha is baked into the color.
    #    3D scatter chokes when you pass `alpha=` as an array, so we
    #    sidestep that path entirely.
    # ---------------------------------------------------------------
    def values_to_rgba(values, cmap_name="viridis", vmin=None, vmax=None,
                       min_alpha=0.05, max_alpha=0.9, use_alpha=True):
        """Convert scalar values to RGBA colors and a normalization object."""
        v = values.astype(float)
        lo = v.min() if vmin is None else vmin
        hi = v.max() if vmax is None else vmax
        rng = hi - lo
        norm = (v - lo) / rng if rng > 0 else np.zeros_like(v)
        rgba = plt.get_cmap(cmap_name)(norm)
        if use_alpha:
            rgba[:, -1] = min_alpha + norm * (max_alpha - min_alpha)
        return rgba, Normalize(vmin=lo, vmax=hi)

    err_rgba, err_norm = values_to_rgba(err_flat, cmap_name=colormap, use_alpha=alpha_scale)
    aq_rgba,  aq_norm  = values_to_rgba(aq_flat, cmap_name=colormap, vmin=0, use_alpha=alpha_scale)

    # ---------------------------------------------------------------
    # 3. Two 3D scatter subplots.
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    # --- Error map ---
    ax0.scatter(xs, ys, zs, c=err_rgba, s=20, edgecolors="none")

    # highlight the next selected point
    # ax0.scatter(
    #     xs[ind], ys[ind], zs[ind],
    #     c="red", s=80, edgecolors="black", linewidths=1.0,
    #     depthshade=False,
    # )
    ax0.set_title("Error Map")
    ax0.set_xlabel("x"); ax0.set_ylabel("y"); ax0.set_zlabel("z")
    fig.colorbar(ScalarMappable(norm=err_norm, cmap=colormap),
                 ax=ax0, fraction=0.04, pad=0.08)

    # --- Acquisition function ---
    ax1.scatter(xs, ys, zs, c=aq_rgba, s=20, edgecolors="none")
    ax1.set_title("Acquisition Function")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    fig.colorbar(ScalarMappable(norm=aq_norm, cmap=colormap),
                 ax=ax1, fraction=0.04, pad=0.08)

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 4. Optional save.
    # ---------------------------------------------------------------
    if to_save:
        img_name = "errormap_iter" + str(iter_nb) + ".jpg"
        out_dir = os.path.join(save_folder, expt_name)
        os.makedirs(out_dir, exist_ok=True)
        img_path = os.path.join(out_dir, img_name)
        plt.savefig(img_path, bbox_inches="tight")

    plt.show()



def plot_training_loss(im2spec_train_loss, im2spec_val_loss, error_train_loss, error_val_loss):
    """
    Plot im2spec and error-model training/validation loss histories.

    Args:
        im2spec_train_loss: List of per-ensemble-member im2spec train losses.
        im2spec_val_loss: List of per-ensemble-member im2spec validation losses.
        error_train_loss: Error-model train loss history.
        error_val_loss: Error-model validation loss history.
    """
    fig, ax = plt.subplots(1,4, figsize = (18,3))

    n_models = len(im2spec_train_loss)

    for i in range(n_models):
        ax[0].semilogy(im2spec_train_loss[i], label = 'im2spec Training loss')
        ax[1].semilogy(im2spec_val_loss[i], label = 'im2spec Validation loss')



    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Epoch loss")
    ax[0].set_title("im2spec training")

    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Epoch loss")
    ax[1].set_title("im2spec Validation")

    ax[2].semilogy(error_train_loss, label = 'error Training loss')
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Epoch loss")
    ax[2].set_title("error model training")

    ax[3].semilogy(error_val_loss, label = 'Error Validation loss')
    ax[3].set_xlabel("Epochs")
    ax[3].set_ylabel("Epoch loss")
    ax[3].set_title("Error model Validation")


    plt.show()




def plot_only_training_loss(im2spec_train_loss, im2spec_val_loss):
    """
    Plot only im2spec training and validation loss histories.

    Args:
        im2spec_train_loss: List of per-ensemble-member training loss histories.
        im2spec_val_loss: List of per-ensemble-member validation loss histories.
    """
    fig, ax = plt.subplots(1,2, figsize = (10,3))

    n_models = len(im2spec_train_loss)

    for i in range(n_models):
        ax[0].semilogy(im2spec_train_loss[i], label = 'im2spec Training loss')
        ax[1].semilogy(im2spec_val_loss[i], label = 'im2spec Validation loss')



    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Epoch loss")
    ax[0].set_title("im2spec training")

    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Epoch loss")
    ax[1].set_title("im2spec Validation")


    plt.show()


def plot_embedding(spectra, spectra_train, xdata = None):
    """
    Plot posterior spectra and the mean spectrum of the training set.

    Args:
        spectra: Posterior spectra array of shape `(N, S)`.
        spectra_train: Training spectra array used to compute the mean curve.
        xdata: Optional x-axis values. If None, sample indices are used.
    """

    n = spectra.shape[0]
    dim = spectra.shape[1]

    if xdata is not None:
        dims = xdata
    else:
        dims = np.arange(dim)

    mean_spectrum = spectra_train.mean(axis = 0)


    fig, ax = plt.subplots(1,2, figsize = (10,3))

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, n))
    for i in range(n):

        ax[0].scatter(dims, spectra[i], color = colors[i])

    ax[0].set_title("Posterior predictions")


    ax[1].plot(dims, mean_spectrum)
    ax[1].set_title("Posterior_training_mean")

    plt.show()

def plot_latent_space(embeddings, trained_embeddings = None, expt_name = 'test_expt', iter_nb = 0,
                      save_folder = r"data", to_save = False, lat_order = [0, 1, 2]):
    """
    Plot a 3D view of latent embeddings.

    Args:
        embeddings: Latent array of shape `(N, D)`.
        trained_embeddings: Optional subset to overlay and color by exploration
            order.
        expt_name: Subfolder name used when saving.
        iter_nb: Iteration number included in the saved filename.
        save_folder: Root folder for saved figures.
        to_save: If True, save the figure before showing it.
        lat_order: Three latent dimensions to plot on x, y, and z axes.
    """

    n = embeddings.shape[0]
    dim = embeddings.shape[1]

    l1 = embeddings[:, lat_order[0]]
    l2 = embeddings[:, lat_order[1]]
    l3 = embeddings[:, lat_order[2]]



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l1, l2, l3, alpha=0.2)  # edgecolors='#1f77b4


    ax.set_title('Latent space')

#     # Remove axis numbers (ticks)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

    if trained_embeddings is not None:
        n_train = trained_embeddings.shape[0]
        scalar = np.arange(n_train)

        l1_t = trained_embeddings[:, lat_order[0]]
        l2_t = trained_embeddings[:, lat_order[1]]
        l3_t = trained_embeddings[:, lat_order[2]]

        a = ax.scatter(l1_t, l2_t, l3_t, c = scalar, s = 30, edgecolors= 'r', alpha = 1,  cmap = 'Reds')

        cbar = plt.colorbar(a)
        cbar.set_label(r"Exploration step", size = 12)


    if to_save:
        img_name = "latentspace_iter"+str(iter_nb)+'.jpg'

        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)

        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')


    plt.show()


def plot_latent_distribution(embeddings, expt_name = 'test_expt', iter_nb = 0, save_folder = r"data", to_save = False):
    """
    Plot the first three latent dimensions as square spatial maps.

    Args:
        embeddings: Latent array of shape `(N, D)`, where `N` must form a
            square grid and at least three latent dimensions are available.
        expt_name: Subfolder name used when saving.
        iter_nb: Iteration number included in the saved filename.
        save_folder: Root folder for saved figures.
        to_save: If True, save the figure before showing it.
    """


    n = embeddings.shape[0]
    n_dim = int(n**0.5)

    l1 = embeddings[:, 0].reshape(n_dim, n_dim)
    l2 = embeddings[:, 1].reshape(n_dim, n_dim)
    l3 = embeddings[:, 2].reshape(n_dim, n_dim)



    fig, ax = plt.subplots(1, 3, figsize = (15, 4))

    ax[0].imshow(l1, origin = 'lower')
    ax[0].set_title('L1 distribution')

    ax[1].imshow(l2, origin = 'lower')
    ax[1].set_title('L2 distribution')

    ax[2].imshow(l3, origin = 'lower')
    ax[2].set_title('L3 distribution')


    if to_save:
        img_name = "latent_distrbn_iter"+str(iter_nb)+'.jpg'

        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)

        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')

    plt.show()

def cluster_latent_space(embeddings, lat_order = [0, 1, 2],  n_clusters = 2):
    """
    Cluster latent embeddings with K-means and plot cluster assignments.

    Args:
        embeddings: Latent array of shape `(N, D)`.
        lat_order: Three latent dimensions to plot on x, y, and z axes.
        n_clusters: Number of K-means clusters.
    """

    n = embeddings.shape[0]
    dim = embeddings.shape[1]

    l1 = embeddings[:, lat_order[0]]
    l2 = embeddings[:, lat_order[1]]
    l3 = embeddings[:, lat_order[2]]




    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)  # Cluster labels

    print(labels.shape)

    # Get cluster centers
    centroids = kmeans.cluster_centers_


    colors = ['red', 'blue', 'green', 'orange']



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(n_clusters):
        ax.scatter(embeddings[labels == i,lat_order[0]], embeddings[labels == i, lat_order[1]], embeddings[labels == i, lat_order[2]],
               color=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

    ax.set_title('Latent space')

#     # Remove axis numbers (ticks)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])



    plt.show()


    n = embeddings.shape[0]
    n_dim = int(n**0.5)

    x = np.linspace(0, n_dim-1, n_dim)

    X = np.asarray(np.meshgrid(x, x)).T.reshape(-1, 2)

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    for i in range(len(X)):
        ax.scatter(X[i, 1], X[i, 0], c = colors[labels[i]], alpha=0.8)

    plt.show()


def plot_spectra(pred_spectra, orig_spectrum, error_val, expt_name = 'test_expt', iter_nb = 0, count = 0,
                 save_folder = r"data", to_save = False, xdata = None):
    """
    Plot one target spectrum with one or more predicted spectra.

    Args:
        pred_spectra: Iterable of predicted spectra to overlay.
        orig_spectrum: Target/original spectrum.
        error_val: Error value displayed in predicted-spectrum labels.
        expt_name: Subfolder name used when saving.
        iter_nb: Iteration number included in the saved filename.
        count: Sample count included in the saved filename.
        save_folder: Root folder for saved figures.
        to_save: If True, save the figure before showing it.
        xdata: Optional x-axis values. If None, sample indices are used.
    """


    fig, ax = plt.subplots(1, figsize = (4,4))

    if xdata is not None:
        x = xdata
    else:
        x = np.arange(len(orig_spectrum))

    #ax.plot(x, orig_spectrum, color= 'black')
    ax.plot(x, orig_spectrum, 'o-', color='black', alpha = 0.6, label='original_spectrum')


    for spectrum in pred_spectra:
        ax.plot(x, spectrum, linewidth = 3, label = f'Predicted\nL1-error: {error_val:.4f}')
    ax.legend(fontsize=10)

    if to_save:
        img_name = "spectrum_iter"+str(iter_nb)+'_sample'+str(count)+'.jpg'

        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)

        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')

    plt.show()

def plot_scale_slider(coordinates, error_mean, aq_fn, ind=None, colormap="viridis"):
    """
    Create an interactive slider for multiscale error/acquisition maps.

    Args:
        coordinates: Array of shape `(N, 3)` with columns `(x, y, scale)`.
        error_mean: Flat array of predicted error values, shape `(N,)`.
        aq_fn: Flat array of acquisition values, shape `(N,)`.
        ind: Optional selected candidate index or indices. Retained for API
            consistency; current plotting does not highlight it.
        colormap: Matplotlib colormap name.
    """

    from ipywidgets import IntSlider, interact

    scales = np.unique(coordinates[:, 2])
    scales = np.array(sorted(scales), dtype=float)

    err_vmin, err_vmax = error_mean.min(), error_mean.max()
    aq_vmin, aq_vmax = 0, aq_fn.max()

    def _draw(scale):
        mask = np.isclose(coordinates[:, 2], scale)

        xs, ys = coordinates[mask, 0], coordinates[mask, 1]
        err = error_mean[mask]
        aq = aq_fn[mask]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))

        s0 = ax0.scatter(
            xs,
            ys,
            c=err,
            cmap=colormap,
            s=30,
            vmin=err_vmin,
            vmax=err_vmax,
        )
        ax0.set_title(f"Error Map (scale={scale:g})")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig.colorbar(s0, ax=ax0, fraction=0.04)

        s1 = ax1.scatter(
            xs,
            ys,
            c=aq,
            cmap=colormap,
            s=30,
            vmin=aq_vmin,
            vmax=aq_vmax,
        )
        ax1.set_title(f"Acquisition (scale={scale:g})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(s1, ax=ax1, fraction=0.04)

        plt.tight_layout()
        plt.show()

    def _draw_by_index(scale_index):
        scale = scales[int(scale_index)]
        _draw(scale)

    return interact(
        _draw_by_index,
        scale_index=IntSlider(
            value=0,
            min=0,
            max=len(scales) - 1,
            step=1,
            description="Scale",
            continuous_update=False,
        ),
    )


def visualize_attention(
    model,
    images,
    spectra,
    full_image,
    coordinates,
    index=0,
    device=None,
    colormap="viridis",
    figsize=(20, 4),
):
    """
    Visualize one image-to-spectrum sample and its attention matrix.

    Args:
        model: Trained attention model exposing `predict` and `get_attn_weights`.
        images: Image patches with shape `(N, H, W)`.
        spectra: Target spectra with shape `(N, S)`.
        full_image: Full 2D image used for spatial context.
        coordinates: Array with at least `(x, y)` columns. A third column is
            shown as scale when present.
        index: Dataset index to visualize.
        device: Optional torch device for model inference. If omitted, the
            model parameter device is used.
        colormap: Matplotlib colormap for the attention matrix.
        figsize: Matplotlib figure size.

    Returns:
        Tuple `(fig, ax, attn_matrix)` for further customization.
    """
    import torch

    if not hasattr(model, "get_attn_weights"):
        raise AttributeError("Model must expose a get_attn_weights() method.")

    images = np.asarray(images)
    spectra = np.asarray(spectra)
    coordinates = np.asarray(coordinates)

    index = int(np.clip(index, 0, len(images) - 1))

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        sample_image = torch.tensor(
            images[index:index + 1], dtype=torch.float32, device=device
        )
        pred_spectrum = model.predict(sample_image).detach().cpu().squeeze().numpy()
        attn_matrix = model.get_attn_weights()

    if attn_matrix is None:
        raise RuntimeError(
            "No attention weights found. Run the model on an input before visualizing."
        )

    if hasattr(attn_matrix, "detach"):
        attn_matrix = attn_matrix.detach().cpu().numpy()
    else:
        attn_matrix = np.asarray(attn_matrix)

    if attn_matrix.ndim == 4:
        attn_matrix = attn_matrix[0].mean(axis=0)
    elif attn_matrix.ndim == 3:
        attn_matrix = attn_matrix[0]

    coord = coordinates[index]
    x_coord, y_coord = coord[0], coord[1]
    scale = coord[2] if len(coord) > 2 else None

    fig, ax = plt.subplots(1, 4, figsize=figsize)

    ax[0].imshow(full_image, origin="lower")
    ax[0].scatter(x_coord, y_coord, c="red", s=40)
    title = f"Dataset index {index}"
    if scale is not None:
        title += f" | scale={scale:g}"
    ax[0].set_title(title)

    patch_im = ax[1].imshow(images[index], origin="lower")
    ax[1].set_title("Patch image")
    fig.colorbar(patch_im, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].plot(spectra[index], label="Target")
    ax[2].plot(pred_spectrum, label="Prediction")
    ax[2].set_title("Spectra")
    ax[2].legend()

    attn_im = ax[3].imshow(attn_matrix, cmap=colormap, aspect="auto")
    ax[3].set_title("Attention matrix")
    ax[3].set_xlabel("Key token")
    ax[3].set_ylabel("Query token")
    fig.colorbar(attn_im, ax=ax[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return fig, ax, attn_matrix


# def plot_scale_slider(coordinates, error_mean, aq_fn, ind=None, colormap="viridis"):
#     """
#     Create an interactive slider for multiscale error/acquisition maps.

#     Args:
#         coordinates: Array of shape `(N, 3)` with columns `(x, y, scale)`.
#         error_mean: Flat array of predicted error values, shape `(N,)`.
#         aq_fn: Flat array of acquisition values, shape `(N,)`.
#         ind: Optional selected candidate index or indices. Retained for API
#             consistency; current plotting does not highlight it.
#         colormap: Matplotlib colormap name.
#     """
#     scales = np.unique(coordinates[:, 2])

#     # Fix color limits to the full-dataset range so colors are comparable across scales.
#     err_vmin, err_vmax = error_mean.min(), error_mean.max()
#     aq_vmin, aq_vmax   = 0, aq_fn.max()

#     def _draw(scale):
#         """Draw error and acquisition maps for one selected scale."""
#         mask = coordinates[:, 2] == scale
#         xs, ys = coordinates[mask, 0], coordinates[mask, 1]
#         err = error_mean[mask]
#         aq  = aq_fn[mask]

#         fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
#         s0 = ax0.scatter(xs, ys, c=err, cmap=colormap, s=30,
#                          vmin=err_vmin, vmax=err_vmax)
#         ax0.set_title(f"Error Map (scale={scale:g})")
#         ax0.set_xlabel("x"); ax0.set_ylabel("y")
#         fig.colorbar(s0, ax=ax0, fraction=0.04)

#         s1 = ax1.scatter(xs, ys, c=aq, cmap=colormap, s=30,
#                          vmin=aq_vmin, vmax=aq_vmax)
#         ax1.set_title(f"Acquisition (scale={scale:g})")
#         ax1.set_xlabel("x"); ax1.set_ylabel("y")
#         fig.colorbar(s1, ax=ax1, fraction=0.04)

#         plt.tight_layout()
#         plt.show()

#     # Use SelectionSlider since scales are discrete
#     from ipywidgets import SelectionSlider, interact
#     interact(_draw, scale=SelectionSlider(options=[float(s) for s in scales],
#                                           description="Scale"))


def plot_scale_specific_error_and_acquisition_maps(scale_coordinates, predicted_errors, aq_fn):
    """
    Plots error and acquisition functions for each unique scale.

    Args:
        scale_coordinates (np.ndarray): Array containing x, y, and scale coordinates.
        predicted_errors (np.ndarray): Predicted error values corresponding to scale_coordinates.
        aq_fn (np.ndarray): Acquisition function values corresponding to scale_coordinates.
    """

    unique_scales = np.unique(scale_coordinates[:, 2])

    # min/max for color scaling
    global_error_min = predicted_errors.min()
    global_error_max = predicted_errors.max()
    global_aq_fn_min = aq_fn.min()
    global_aq_fn_max = aq_fn.max()

    # min/max for X and Y coordinates across scales
    # Add a small padding to prevent dots from being cut off
    padding = 10 # Adjust padding value as needed
    global_x_min = scale_coordinates[:, 0].min() - padding
    global_x_max = scale_coordinates[:, 0].max() + padding
    global_y_min = scale_coordinates[:, 1].min() - padding
    global_y_max = scale_coordinates[:, 1].max() + padding

    # loop through scales
    for s in unique_scales:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Error and Acquisition Functions for Scale: {int(s)}', fontsize=16)

        scale_mask = scale_coordinates[:, 2] == s
        coords_for_scale = scale_coordinates[scale_mask, :2] # x, y coordinates
        errors_for_scale = predicted_errors[scale_mask]
        aq_fn_for_scale = aq_fn[scale_mask]

        # error plot
        sc1 = axes[0].scatter(coords_for_scale[:, 0], coords_for_scale[:, 1],
                              c=errors_for_scale, cmap='jet', s=50, marker='o', edgecolors='none',
                              vmin=global_error_min, vmax=global_error_max)
        axes[0].set_title('Error Function')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('X')
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].set_xlim(global_x_min, global_x_max)
        axes[0].set_ylim(global_y_min, global_y_max)
        fig.colorbar(sc1)

        # aqn function plot
        sc2 = axes[1].scatter(coords_for_scale[:, 0], coords_for_scale[:, 1],
                              c=aq_fn_for_scale, cmap='jet', s=50, marker='o', edgecolors='none',
                              vmin=global_aq_fn_min, vmax=global_aq_fn_max)
        axes[1].set_title('Acquisition Function')
        axes[1].set_xlabel('Y')
        axes[1].set_ylabel('X')
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].set_xlim(global_x_min, global_x_max)
        axes[1].set_ylim(global_y_min, global_y_max)
        fig.colorbar(sc2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
        plt.show()
