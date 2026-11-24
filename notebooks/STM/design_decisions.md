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

## Drop the leftover training batch (`drop_last`) — 2026-11-24

**Problem:** `train_model` sets the batch size to `len(train) // n_batches` (`n_batches=6`) and used
to keep the leftover samples as a small extra batch. In `ms` the training set grows by 40 samples per
iteration, so the leftover cycled 0 → 4 → 2 samples (period 3). That small batch corrupted the
BatchNorm statistics, both in the last training step of every epoch and in SWA's `update_bn`, which
weights every batch equally. The result was two error levels:

| `ms` random run (mean test error) | no leftover | leftover 2 | leftover 4 |
|---|---|---|---|
| before | 0.0077 | 0.0109 | 0.0104 |

In `ss` (leftover 0–5), the error spikes (about 0.024) were all iterations with a leftover of 1.
The same bug biased the best-scale choice toward scale 4 (28–30% of patches, against 3% in clean iterations).

**Fix:** in `src/nnerror/training_functions.py`, `train_model`:

```diff
     train_batch_size = max(1, len(train_dataset)//n_batches)
     val_batch_size = max(1, len(val_dataset)//n_batches)
 
-    tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
-    val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)
- 
     #if batchsize provided disregard n_batches
     if batchsize is not None:
-        tr_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
-        val_dataloader = DataLoader(val_dataset, batch_size = batchsize, shuffle = True)
- 
+        train_batch_size = batchsize
+        val_batch_size = batchsize
+
+    # Drop the leftover training batch: a final batch of only a few samples gives noisy
+    # BatchNorm statistics, both during training and in SWA's update_bn (which weights every
+    # batch equally). Kept only if dropping it would leave no batches at all.
+    tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True,
+                               drop_last = len(train_dataset) >= train_batch_size)
+    val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)
```

- Only the training loader drops its leftover batch; the validation loader is unchanged.
- At most `batch_size - 1` samples (5 or fewer here) are skipped per epoch, and shuffling makes them different each epoch.
- Check (6 iterations, `ms` random, 100 epochs): mean test error before 0.0076 / 0.0106 / 0.0110 / 0.0081 / 0.0115 / 0.0108,
  after 0.0076 / 0.0078 / 0.0086 / 0.0079 / 0.0082 / 0.0082.
- Not changed: `train_model_ensemble` and `train_error_ensemble` have the same leftover-batch pattern but are not used by this script.
- Results produced before this fix (for example `data/active_learning_results1`, 2026-11-24) were trained with the leftover batch.
