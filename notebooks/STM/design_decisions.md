# Design decisions — STM active learning (`stm_ms_acqfn_script.py`)

## Data normalization (Gaussian / z-score, `Norm_G`)

Selected with `--norm gauss` (default). `--norm 0to1` keeps the previous min-max behaviour.
The normalization is part of the output filename (`..._normgauss.h5` / `..._norm0to1.h5`).

| Quantity | Normalization | Fitted on |
|---|---|---|
| Images | Global z-score (one mean/std over the full image) | Full image, once. The image is known everywhere, so this is not leakage. |
| Spectra | Global z-score (one mean/std over all channels) | Initial training set, fixed for all iterations, so error units stay comparable across iterations. |
| Errors (error-model target) | Global z-score of `log(error + eps)` | Current training errors, every iteration. |

**Why global and not per-channel for spectra:** the per-channel std varies about 3000× across the
121 bias channels (≈1e-4 near zero bias, ≈0.19 at the high-bias end). Per-channel z-scoring would
amplify the near-zero-bias channels, which are mostly noise, by about 1000×, and they would dominate
the loss and the L1/L2 errors.

**Why log for errors:** errors are positive and right-skewed. A log makes them closer to Gaussian, and
`exp` on denormalization guarantees positive predicted errors.

**Recorded values are in real units:** im2spec predictions are denormalized before computing errors
(`spectra_errors`), and error-model predictions are denormalized with the same `Norm_G`. All errors
saved in the `.h5` file are therefore in the original spectra units (the 0–1 scaling from
`paired_images_spectra_1`) and can be compared with earlier `0to1` runs.

**Side effects to keep in mind:**
- With `pad`, the zeroed patch edges now equal the image mean instead of the darkest value.
- Training targets have std 1 instead of about 0.05, so loss values are larger and not comparable with old runs; `lr` may need retuning.
- Z-scored spectra are centred, so the model is trained on centred spectra. The `cos` error itself is still computed in real units.

## Validity-mask channel for padded images (`--mask-channel`)

On by default (`--no-mask-channel` turns it off). With `pad` augmentation the image edges are set to 0,
which in `gauss` mode is the image mean, so on its own the model cannot tell "padded" from "average height".
A second input channel carries the mask: 1 = valid pixel, 0 = padded pixel. Input shape becomes `(B, 2, H, W)`.

- **Only used for `--scale-type ms` with `pad` and `im2spec`.** In any other case the script prints a
  message and falls back to single-channel images (`ss` has no padding, and `interpolate` has no
  invalid pixels).
- **The mask is built from each row's scale** (`scale_coordinates[:, 2]`, via `edges_zeroed_image` on
  an all-ones image), not from `image == 0`, because real pixels can be exactly 0 or the mean value.
- **The mask is not normalized** (it stays 0/1).
- Implemented in the script with `im2spec_mask` and `CustomDecoder_mask`: the first conv block takes 2
  input channels, and `forward` does not add a channel axis. The library models are unchanged.
- Runs that use the mask get `_mask` in the output filename.
