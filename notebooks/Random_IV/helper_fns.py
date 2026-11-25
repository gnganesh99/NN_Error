"""Helper functions for the Random_IV notebook (data extraction and checks).

`get_grid_coords` and `get_subimages` follow `engagegp.utils` (https://github.com/yongtaoliu/ENGAGEGP)
so the notebook does not need the `engagegp` package.
"""
import numpy as np


def get_grid_coords(img, step=1):
    """Grid of (row, col) coordinates for a 2D image, in row-major order.

    Args:
        img: 2D (or 3D, channels last) numpy array
        step: distance between grid points

    Returns:
        N x 2 int array of (row, col) coordinates. With step=1, row k is pixel
        k of `img.ravel()`, i.e. flat index k = row * width + col.
    """
    h, w = img.shape[:2]
    rows, cols = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing="ij")
    return np.stack([rows.ravel(), cols.ravel()], axis=1)


def get_subimages(img, coordinates, window_size):
    """Square patches centred at the given (row, col) coordinates.

    Same behaviour as `engagegp.utils.get_subimages`: a patch spans
    [c - w//2, c + w//2) in each direction, and coordinates whose patch would
    cross the image border are skipped.

    Args:
        img: 2D (h, w) or 3D (h, w, c) numpy array
        coordinates: N x 2 array of (row, col) coordinates
        window_size: size of the square window

    Returns:
        subimages: (M, window_size, window_size, channels) array
        valid_coords: (M, 2) coordinates of the extracted patches
        valid_indices: (M,) indices into `coordinates` of the extracted patches.
            Use these to select the matching spectra when M < N.
    """
    if img.ndim == 2:
        img = img[..., None]
    h, w, _ = img.shape
    half = window_size // 2

    subimages, valid_coords, valid_indices = [], [], []
    for idx, (y, x) in enumerate(np.asarray(coordinates, dtype=int)):
        if y - half >= 0 and y + half <= h and x - half >= 0 and x + half <= w:
            patch = img[y - half:y + half, x - half:x + half, :]
            if patch.shape[0] == window_size and patch.shape[1] == window_size:
                subimages.append(patch)
                valid_coords.append([y, x])
                valid_indices.append(idx)

    if len(valid_indices) < len(coordinates):
        print(f"get_subimages: {len(coordinates) - len(valid_indices)} of {len(coordinates)} "
              f"coordinates skipped (window {window_size} crosses the image border)")
    return np.array(subimages), np.array(valid_coords), np.array(valid_indices, dtype=int)


def remove_plane(z, rows, cols):
    """Subtract a least-squares plane a*row + b*col + c from values z."""
    A = np.c_[rows, cols, np.ones_like(rows)].astype(float)
    coef, *_ = np.linalg.lstsq(A, z.astype(float), rcond=None)
    return z - A @ coef


def estimate_position_shift(topo_map, coords, point_heights, max_shift=15):
    """Estimate a (row, col) offset between the image and the spectroscopy points.

    Compares the map topography at each point with the height recorded during the
    spectrum at that point (both with a fitted plane removed), for every integer shift
    up to `max_shift` pixels. A best shift far from (0, 0) means the sample/tip drifted
    between the image and the spectroscopy measurement.

    Args:
        topo_map: (H, W) topography image
        coords: (N, 2) (row, col) positions of the spectra on the image grid
        point_heights: (N,) mean height recorded during each spectrum
        max_shift: largest shift tried in each direction (pixels)

    Returns:
        dict with "best_shift" (drow, dcol), "best_corr", "corr_at_zero",
        and "corr_map" ((2*max_shift+1)^2 array of correlations, index [drow+max_shift, dcol+max_shift]).
    """
    H, W = topo_map.shape
    rr, cc = np.mgrid[:H, :W]
    topo_flat = remove_plane(topo_map.ravel(), rr.ravel(), cc.ravel()).reshape(H, W)
    rows, cols = np.asarray(coords, dtype=int).T
    heights = remove_plane(np.asarray(point_heights, dtype=float), rows, cols)

    shifts = np.arange(-max_shift, max_shift + 1)
    corr = np.full((len(shifts), len(shifts)), np.nan)
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            r, c = rows + dy, cols + dx
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            corr[i, j] = np.corrcoef(topo_flat[r[ok], c[ok]], heights[ok])[0, 1]

    i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
    return {"best_shift": (int(shifts[i]), int(shifts[j])), "best_corr": float(corr[i, j]),
            "corr_at_zero": float(corr[max_shift, max_shift]), "corr_map": corr}


# ---------------------------------------------------------------------------
# Data loading and offset-aware patch extraction (used by iv_ms_im2spec_error.ipynb)
# ---------------------------------------------------------------------------

def load_random_iv(map_file, spec_file, grid_shape=(128, 128)):
    """Load the topography image and the IV curves measured at random grid points.

    Args:
        map_file: h5 file of the image scan (BELineData); Channel_001 = height
        spec_file: h5 file of the IV grid (BEPSData); Channel_002 = current, Channel_001 = height during IV
        grid_shape: (rows, cols) of the position grid

    Returns:
        dict with
            "topo": (rows, cols) height image from the map file
            "iv": (N, S) raw IV curves at the measured points
            "v_step": (S,) DC voltage of each step
            "points_measured": (N,) flat grid indices of the measured points (row * cols + col)
            "coords": (N, 2) recorded (row, col) of the measured points
            "iv_height": (N,) mean height recorded during each IV curve
    """
    import h5py

    with h5py.File(map_file, "r") as f:
        topo = f["Measurement_000/Channel_001/Raw_Data"]["r"].reshape(grid_shape).astype(np.float64)

    with h5py.File(spec_file, "r") as f:
        sho = f["Measurement_000/Channel_000/Raw_Data-SHO_Fit_000"]
        # Measured points have a nonzero SHO quality factor (same rule as usid_to_pr in RandomIV.ipynb).
        points_measured = np.where(sho["Guess"]["Quality Factor"][:, 1] != 0)[0]
        v_step = sho["Spectroscopic_Values"][0, :].astype(np.float64)
        iv = f["Measurement_000/Channel_002/Raw_Data"]["r"][points_measured].astype(np.float64)
        iv_height = f["Measurement_000/Channel_001/Raw_Data"]["r"][points_measured].mean(1).astype(np.float64)

    coords = np.stack(np.unravel_index(points_measured, grid_shape), axis=1)
    return {"topo": topo, "iv": iv, "v_step": v_step, "points_measured": points_measured,
            "coords": coords, "iv_height": iv_height}


def smooth_spectra(spectra, window_length=55, polyorder=5):
    """Savitzky-Golay smoothing of each spectrum (same settings as RandomIV.ipynb)."""
    from scipy.signal import savgol_filter
    return savgol_filter(np.asarray(spectra, dtype=np.float64), window_length, polyorder, axis=-1)


def flatten_image(img):
    """Remove the least-squares plane (sample tilt) from a 2D image."""
    H, W = img.shape
    rr, cc = np.mgrid[:H, :W]
    return remove_plane(img.ravel(), rr.ravel(), cc.ravel()).reshape(H, W)


def apply_offset(coords, offset):
    """Image-grid (row, col) of each point: recorded coords + (drow, dcol)."""
    return np.asarray(coords, dtype=int) + np.asarray(offset, dtype=int)


def get_padded_subimages(img, coordinates, window_size, pad_value=0.0):
    """Square patches centred at (row, col) coordinates, padding the image border.

    Same patch convention as `get_subimages` (rows/cols c - w//2 to c + w//2 - 1), but
    no coordinate is skipped: the image is padded by `window_size // 2` pixels with
    `pad_value`, and a validity mask marks which patch pixels lie inside the image.

    Returns:
        patches: (N, window_size, window_size) image patches
        masks: (N, window_size, window_size) 1 = pixel inside the image, 0 = padding
    """
    coordinates = np.asarray(coordinates, dtype=int)
    H, W = img.shape
    half = window_size // 2
    if (coordinates < 0).any() or (coordinates[:, 0] >= H).any() or (coordinates[:, 1] >= W).any():
        raise ValueError("Some coordinates are outside the image.")

    img_p = np.pad(img, half, mode="constant", constant_values=pad_value)
    valid_p = np.pad(np.ones_like(img, dtype=np.float32), half, mode="constant", constant_values=0)

    patches = np.empty((len(coordinates), window_size, window_size), dtype=np.float32)
    masks = np.empty_like(patches)
    for k, (y, x) in enumerate(coordinates):
        # (y, x) in the padded image is (y + half, x + half); the window starts half before it.
        patches[k] = img_p[y:y + window_size, x:x + window_size]
        masks[k] = valid_p[y:y + window_size, x:x + window_size]

    n_border = int((masks.min(axis=(1, 2)) == 0).sum())
    if n_border:
        print(f"get_padded_subimages: {n_border} of {len(coordinates)} patches reach the image border (padded, mask = 0)")
    return patches, masks


def augment_with_mask(indices, patches, masks, spectra, coords, scales):
    """Multiscale 'pad' augmentation with a validity-mask channel.

    The image patches and the border masks go through the same `append_multiscale_data`
    ('pad' mode), so the augmented mask is 0 both where the scale zeroes the patch edges
    and where the patch lies outside the image.

    Args:
        indices: indices of the points to use
        patches, masks: (N, W, W) image patches and border masks
        spectra: (N, S)
        coords: (N, 2) (row, col) image-grid coordinates
        scales: scales smaller than W; the full patch (scale W) is appended last

    Returns:
        images_aug (M, 2, W, W), spectra_aug (M, S), coords_aug (M, 3) as (row, col, scale).
        Rows are stacked in blocks of len(indices), one block per scale, so row r is point indices[r % len(indices)].
    """
    from nnerror.utils.image_utils import append_multiscale_data

    indices = np.asarray(indices)
    img_aug, spec_aug, coords_aug = append_multiscale_data(
        patches[indices], spectra[indices], scales=list(scales), coordinates=coords[indices], append_image_type="pad")
    mask_aug, _, coords_mask = append_multiscale_data(
        masks[indices], spectra[indices], scales=list(scales), coordinates=coords[indices], append_image_type="pad")
    assert np.array_equal(coords_aug, coords_mask)
    return np.stack([img_aug, mask_aug], axis=1).astype(np.float32), spec_aug, coords_aug


# ---------------------------------------------------------------------------
# Interactive 3D error map (plotly.js from a CDN; no Python plotly package needed)
# ---------------------------------------------------------------------------

def error_3d_html(coords, scales, error_maps, highlight=None, title="3D error map", plotly_version="2.35.2"):
    """Standalone HTML page with an interactive 3D scatter of errors per (col, row, scale).

    Args:
        coords: (N, 2) (row, col) of the points
        scales: list of the S scales (z axis)
        error_maps: dict name -> (S, N) array of errors; a dropdown switches between them
        highlight: optional dict name -> (indices, scale_per_index or None, color, symbol) drawn on top
            (scale None = a vertical line through all scales)
        title: plot title

    Returns:
        HTML string (open in a browser, or show in a notebook with `show_html_iframe`).
    """
    import json

    coords = np.asarray(coords)
    S, N = len(scales), len(coords)
    xs = np.tile(coords[:, 1], S).tolist()
    ys = np.tile(coords[:, 0], S).tolist()
    zs = np.repeat(scales, N).tolist()
    point = np.tile(np.arange(N), S)

    traces, names = [], list(error_maps)
    for k, name in enumerate(names):
        e = np.asarray(error_maps[name]).reshape(S, N).ravel()
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": name, "visible": k == 0,
            "x": xs, "y": ys, "z": zs,
            "marker": {"size": 3, "color": e.tolist(), "colorscale": "Viridis", "opacity": 0.85,
                       "cmin": float(np.percentile(e, 1)), "cmax": float(np.percentile(e, 99)),
                       "colorbar": {"title": {"text": name}}},
            "text": [f"point {p}<br>row {y}, col {x}<br>scale {z}<br>{name}: {v:.3g}"
                     for p, x, y, z, v in zip(point, xs, ys, zs, e)],
            "hoverinfo": "text",
        })
    n_maps = len(traces)

    for name, (idx, sc, color, symbol) in (highlight or {}).items():
        idx = np.asarray(idx, dtype=int)
        if sc is None:   # vertical line through all scales for each point
            x, y, z = [], [], []
            for i in idx:
                x += [int(coords[i, 1])] * 2 + [None]
                y += [int(coords[i, 0])] * 2 + [None]
                z += [min(scales), max(scales), None]
            traces.append({"type": "scatter3d", "mode": "lines", "name": name, "x": x, "y": y, "z": z,
                           "line": {"color": color, "width": 6}, "hoverinfo": "name"})
        else:
            traces.append({"type": "scatter3d", "mode": "markers", "name": name,
                           "x": coords[idx, 1].tolist(), "y": coords[idx, 0].tolist(), "z": np.asarray(sc).tolist(),
                           "marker": {"size": 6, "color": color, "symbol": symbol},
                           "text": [f"point {i}" for i in idx], "hoverinfo": "text+name"})

    buttons = [{"label": name, "method": "update",
                "args": [{"visible": [j == k for j in range(n_maps)] + [True] * (len(traces) - n_maps)}]}
               for k, name in enumerate(names)]
    layout = {
        "title": {"text": title}, "height": 700, "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
        "scene": {"xaxis": {"title": {"text": "col (px)"}}, "yaxis": {"title": {"text": "row (px)"}},
                  "zaxis": {"title": {"text": "scale (px)"}, "tickvals": list(scales)},
                  "aspectmode": "manual", "aspectratio": {"x": 1, "y": 1, "z": 0.6}},
        "updatemenus": [{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.0, "xanchor": "left"}] if len(names) > 1 else [],
        "legend": {"x": 0.8, "y": 0.95},
    }
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{plotly_version}/plotly.min.js"></script></head>
<body style="margin:0;background:#fff"><div id="plot" style="width:100%;height:700px"></div>
<script>Plotly.newPlot("plot", {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});</script>
</body></html>"""


def show_html_iframe(html, height=720):
    """Show a standalone HTML page inside a notebook output (an iframe keeps its scripts isolated)."""
    import html as _html
    from IPython.display import HTML, display
    display(HTML(f'<iframe srcdoc="{_html.escape(html, quote=True)}" style="width:100%;height:{height}px;border:none"></iframe>'))
