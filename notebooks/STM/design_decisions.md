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

## Position-aware FNO readout (`readout="grid4"`) — 2026-11-24

**Problem:** `FNOEncoder` reduced the FNO feature map with a global average (`AdaptiveAvgPool2d((1, 1))`).
Each spectrum belongs to the patch centre, but after a global average the encoder knows only *which*
features a patch contains, not *where* they are relative to the centre. In a check (initial 10% split,
`ms` + `interpolate` training, 922 full-scale test patches, 100 epochs), `FNO` and `FNO_attn` barely
beat predicting the mean training spectrum, and their predictions hardly varied between patches:

| model | test L1 | train loss | prediction spread / target spread |
|---|---|---|---|
| mean-spectrum baseline | 0.0118 | – | 0 |
| im2spec | 0.0078 | 0.15 | 0.83 |
| im2spec_attn | 0.0075 | 0.33 | 0.66 |
| FNO (global) | 0.0103 | 0.62 | 0.29 |
| FNO_attn (global) | 0.0125 | 0.63 | 0.24 |

Training FNO for longer (400 epochs) did not help (train loss stayed about 0.63, test L1 0.0098).
The error model reuses this frozen encoder, so FNO's distance acquisition was close to random.

**Sandbox comparison of readouts** (same backbone and decoder, 2 seeds; spread = prediction std / target std):

| readout | test L1, 100 epochs | seed range | train loss | spread |
|---|---|---|---|---|
| `global` (original) | 0.0107 | 0.0095–0.0119 | 0.61 | 0.21 |
| **`grid4`** (4×4 average pool, flattened) | **0.0086** | 0.0082–0.0090 | 0.44 | 0.48 |
| `center` (mean of central 4×4 pixels) | 0.0092 | 0.0088–0.0097 | 0.30 | 0.53 |
| `center+glob` (both, concatenated) | 0.0094 | 0.0088–0.0100 | 0.36 | 0.45 |

At 300 epochs (early stopping disabled): `grid4` 0.0092 (train loss 0.03), `center` 0.0104 (train loss 0.04).

**Observations:**
- The global average was the main limitation: every position-aware readout helps. `grid4` is best,
  with about 20% lower error than `global`, a third of the seed-to-seed variation, and predictions
  that vary between patches more than twice as much.
- The remaining gap to im2spec is not under-training: longer training overfits the roughly 100 training
  patches (×8 scales) and test error rises. With this little data the CNN generalizes better.
- `center` alone is worse than `grid4`: the spectrum depends on the surroundings, not only the centre pixels.
- Not yet checked: whether `grid4` improves FNO's active-learning (error-model) behaviour in a full run.

**Change:**
- `FNOEncoder(readout="global" | "grid4", grid_size=4)` in `src/nnerror/networks/neuralop_im2spec.py`.
  `"grid4"` pools to a 4×4 map, flattens it (`hidden_channels * 16` features) and feeds that to the same
  MLP head. It is still independent of the input size.
- `FNO_im2spec` and `FNO_im2spec_attn` (`attn_models.py`) pass `readout` through.
- **The library default stays `"global"`**, so existing notebooks and earlier results reproduce.
- The script uses `--fno-readout grid4` by default (`--fno-readout global` gives the original). FNO and
  FNO_attn output files get `_readoutgrid4` or `_readoutglobal` in their names.
- The sandbox script used for the comparison was not added to the repository.

**Check through the script** (`ms`, 1 iteration, 100 epochs; mean reduced im2spec error):
FNO `grid4` 0.0080, FNO `global` 0.0102, im2spec 0.0072. **FNO_attn `grid4` stays at about 0.0117 (the
mean-spectrum level).** At first this looked like a `LayerNorm` problem in `Attn_Block`. The next section
shows that it is not: the block can learn to pass values through, and the failure is an encoder collapse.

## Attention block (`Attn_Block`) over the latent vector — investigation, 2026-11-24 (no code changed)

**Tested on the STM dataset only** (the CITS image/spectra pair used by `stm_ms_acqfn_script.py`;
32×32 patches, `coordinate_step=15`, initial 10% split with `random_state=42`, `ms` training augmentation,
922 full-scale test patches, `gauss` normalization, 100 epochs, SWA, early stopping off, seeds 42 and 7).
The conclusions may not carry over to other datasets.

**What the block does:** each of the 5 latent values becomes a token via `Linear(1, 32)` plus a learned
positional embedding, then goes through 4-head self-attention, `LayerNorm(token + attention)` and `Linear(32, 1)`,
so it returns 5 values to the decoder.

**1. Can the block pass its input through?** Yes. Trained on its own to reproduce its 5 inputs (`y = x`),
the current block reaches R² ≥ 0.995 for input std 0.1, 1 and 10. The `LayerNorm` magnitude loss noted above
only applies at initialization and is not the cause of the poor results.

**2. Block variants inside the models** (test L1, mean of 2 seeds; mean-spectrum baseline 0.0118):

| model | current block | `x + block(x)` | pre-norm block | no attention |
|---|---|---|---|---|
| FNO_attn (`grid4`) | 0.0101 (0.0085–0.0118) | 0.0109 | 0.0118 (collapsed) | **0.0085** (0.0084–0.0086) |
| im2spec_attn | 0.0082 | 0.0078 | 0.0078 | **0.0078** |

**3. Why FNO_attn fails: the encoder collapses.**

| FNO_attn run | test L1 | latent std across patches |
|---|---|---|
| current block, seed 42 | 0.0118 | 0.001 |
| pre-norm block | 0.0118 | 0.000 |
| current block, seed 7 | 0.0085 | 33 |
| no attention | 0.0084 | 13 |

In the failed runs the FNO encoder outputs the same latent vector for every patch, so the model predicts the
mean spectrum. The decoder is not dead (40–60% of its first-layer units are active). With the attention block
in the path, training sometimes settles early on the mean-spectrum solution and the encoder never recovers.
Whether this happens depends on the seed. A skip connection around the block does not prevent it.

**Conclusions (STM dataset):**
- Attention over the 5 latent values gives no benefit for either model. For im2spec it makes no difference;
  for FNO it adds a seed-dependent risk of total collapse.
- FNO without attention (`grid4` readout) is the best and most stable FNO setup (0.0085).
- If attention is wanted, a more meaningful design would attend over spatial positions, for example the 16
  cells of the `grid4` map, each a 32-channel token, instead of over latent scalars. Not tried.

**Not changed:** `Attn_Block`, `im2spec_attn` and `FNO_im2spec_attn` are unchanged. The sandbox scripts were
not added to the repository.

## Summary: im2spec vs FNO (STM dataset, 10% training split) — 2026-11-24

Mean test L1 on the 922 full-scale test patches (lower is better):

| check | im2spec | FNO `grid4` (no attention) | FNO `global` (original) | mean-spectrum baseline |
|---|---|---|---|---|
| Sandbox, 100 epochs, seeds 42 and 7 | **0.0078** | 0.0085–0.0086 | 0.0103–0.0107 | 0.0118 |
| Through the script, `ms`, 1 iteration | **0.0072** | 0.0080 | 0.0102 | – |

- **im2spec is better than FNO at this training size.** The `grid4` readout closes most of the gap to
  im2spec, but not all of it.
- FNO overfits with longer training (300 epochs: `grid4` 0.0092), so the gap is not due to under-training.
- Attention does not improve either model (see the previous section).

**Limitation:** every comparison used only the initial 10% split (`train_size=0.1`, `random_state=42`;
about 100 patches, about 800 rows after `ms` augmentation), one split and 2 seeds. Larger training sizes and
full active-learning runs were not compared. FNO may close the gap as the training set grows during active
learning; this is untested.
