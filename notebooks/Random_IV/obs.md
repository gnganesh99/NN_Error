# Observations — Random_IV (`RandomIV.ipynb`)

## Data files

Both files are BEPS-style h5 files written by the Cypher AFM (`be_analyzer`, DAQ `V3 020222 01`) on the same
128×128 position grid (1.2 µm scan, about 9.4 nm per pixel).

| | Background 2D image | IV points (spectra + height during IV) |
|---|---|---|
| **File** | `content/V3_IV_map_1d2um_0002.h5` | `content/V3_IV_spec_1d2um_0003.h5` |
| **Data type** | `BELineData`, an ordinary line-by-line image scan | `BEPSData`, IV curves at grid points (`DC modulation mode`) |
| **Recorded** | 7/20/2022, 1:06:25 PM | 7/20/2022, 1:15:05 PM |
| **Height dataset** | `Measurement_000/Channel_001/Raw_Data['r']`, size (16384, 1), reshaped to 128×128 → `topo` | `Measurement_000/Channel_001/Raw_Data['r']`, size (16384, 256) → `IV_height` |
| **Label in the file** | `channel_type: 'topo'` | `channel_type: 'topography'` |
| **Used as** | height at each pixel | height recorded at each of the 256 voltage steps of an IV curve, averaged into one value per point, at the 2001 measured points only |
| **Value range** | −2.978 to −2.428 | point averages −2.856 to −2.486 (unmeasured points are 0) |

Other channels:

| Channel | Map file (label: `'topo'`) | Spectra file |
|---|---|---|
| `Channel_002` | current map, −0.008 to −0.0007 | `'current'`, the IV curves, −0.042 to −0.0007 |
| `Channel_003` | deflection, 0.86–1.05 (around the 1 V setpoint) | `'auxillary'`, 0 to 0.95 |

**Caveats on the labels:**
- In the map file, Channels 001, 002 and 003 are all labelled `'topo'`. The names "Height / Current / Deflection"
  come from the notebook, not the file.
- The file stores no units for any channel.
- The two `Channel_001`s have the same raw range (about −2.4 to −2.9), and the other channels match between the
  files, which supports both `Channel_001`s being height.

## Data integrity (checked 2026-11-24)

- 2001 measured points, all unique. "Measured" means the SHO-fit quality factor is not 0, and these are exactly the
  pixels with a nonzero IV curve. All other pixels are 0.
- 256 voltage steps per IV curve. No NaNs in the image or the IV data.
- The voltage sweeps 0 → −2 V (index 128) → 0. The divide-by-zero warning in the turn-on cell comes from
  `np.gradient` at the turning point and is harmless.
- Coordinates match the spectrum order: flat index = row × 128 + col (USID position indices, X fastest).
- Measured points lie in rows 15–113 and cols 15–113, so **30 is the largest patch size with no skipped points**
  (32 would drop 80 points). The save cell crops the patches to 28×28 (`img[:, 1:-1, 1:-1]`).

## Position offset between the image and the IV points

The height recorded during each IV curve should match the map topography at the same pixel (both compared with a
fitted plane removed):

| IV point positions | correlation with map height |
|---|---|
| as recorded | 0.39 |
| shifted by (−12 rows, +7 cols) | **0.82** |

- The shift is about −112 nm by +66 nm. The correlation peaks are broad, so the estimate is good to about ±1–2 px.
- The same grains line up visually after the shift. See the "Position offset" section at the end of the notebook.
- **Likely cause: drift.** The spectra measurement started about 9 minutes after the image and took over an hour:
  each of the 2001 points has a 1 s settle time, a 1 s transit time and 256 steps of about 1 ms, so about 2.3 s per
  point. A drift of this size over that time is realistic for an AFM.
- A single fixed shift is only an approximation. The file has no per-point timestamps or measurement order, so
  whether the drift was steady or changed during the run cannot be checked.
- **Effect if not corrected:** each spectrum is paired with an image patch about 14 px from where it was measured,
  half of a 30 px patch.
- **Effect if corrected:** the shifted points reach 3 px from the bottom edge and 8 px from the right edge, so only
  1640 of 2001 points fit a 30×30 patch without padding.
- Status: `RandomIV.ipynb` is unchanged (recorded positions). `iv_ms_im2spec_error.ipynb` assumes the
  (−12, +7) offset (`OFFSET` parameter).

## First im2spec + error-model run (`iv_ms_im2spec_error.ipynb`, 2026-11-24)

Setup: 2001 measured points only; offset (−12, +7); Savitzky-Golay smoothed spectra (55, 5), full 256-step sweep;
image with the plane removed, then global z-score; 16×16 patches, border padded (image 0, mask 0); `pad` multiscale at
scales 8, 10, 12, 16 with a mask channel; fixed 10% train split (200 points, 800 rows), seed 42; im2spec + error model
as in the STM pipeline (100 epochs each, SWA, no active-learning loop).

| | test L1 |
|---|---|
| mean-spectrum baseline | 0.00052 |
| im2spec at scale 8 / 10 / 12 / 16 | 0.00052 / 0.00053 / 0.00054 / 0.00054 |

- **im2spec does not beat the mean-spectrum baseline.** Training loss keeps falling, but the loss on the test pool
  is lowest at about epoch 10 and then rises: the model overfits.
- **The error model has no predictive power:** correlation between predicted and true error on the test rows is −0.01.
  The distance acquisition is therefore effectively random. It picked points 1163, 1141, 1112, 1298 and 1284, in two
  tight clusters around image (row, col) (58, 98) and (66, 67).
- Best scale (minimum predicted error): 8 for 708 points, 10 for 234, 12 for 236 and 16 for 823.

## Why the model likely fails: the IV response is barely spatially correlated

Checked directly on the data, using the current at −2 V (step 128) of each smoothed IV curve:

- **Nearest neighbours barely agree.** Most points have a neighbour 1 px (about 9 nm) away, yet the −2 V current of a
  point and its nearest neighbour correlate only **0.10**. The IV curves change much more from point to point than the
  topography does, so a model based on image patches cannot predict them well.
- **Topography is a weak predictor.** Correlation of the −2 V current with the local height (window 1–15 px):
  −0.15 with the offset, −0.04 without. The offset helps, but the signal is small. The height recorded during the
  IV itself correlates −0.26.
- A likely cause is the tip or sample changing during the hour-long IV measurement, the same time window as the drift.
- **Image row 0 is a scan-start artifact:** its mean (plane removed) is −0.111 against −0.054 for row 1 (image std 0.047).
  Patches near the bottom edge include it; it could be masked like the padding.

## Border padding vs scale (with the (−12, +7) offset)

At scale `s`, `pad` keeps rows/cols `c − s/2 .. c + s/2 − 1` around the point, so the image border only matters at that scale
if the point is closer than `s/2` px to the bottom/left edge (or `s/2 − 1` px to the top/right edge). With the offset the
points span rows 3–101 and cols 22–120, so in practice only the **bottom edge** (rows 3–15) matters for scales up to 32.

| scale | points with padding | padded share of the kept region: mean over those points / worst | points > 25% / > 50% padded | kept region includes image row 0 |
|---|---|---|---|---|
| 4 | 0 (0%) | – | 0 / 0 | 0 |
| 8 | 24 (1.2%) | 12.5% / 12.5% | 0 / 0 | 44 |
| 12 | 59 (2.9%) | 17.9% / 25.0% | 0 / 0 | 79 |
| 16 | 105 (5.2%) | 18.5% / 31.2% | 24 / 0 | 127 |
| 20 | 179 (8.9%) | 17.9% / 41.5% | 44 / 0 | 159 |
| 24 | 253 (12.6%) | 18.3% / 47.9% | 60 / 0 | 198 |
| 28 | 323 (16.1%) | 19.2% / 52.3% | 83 / 1 | 241 |
| 32 | 396 (19.8%) | 19.9% / 55.5% | 112 / 3 | 278 |

- Even at scale 32, no point is mostly padding: the worst is 55% padded and only 3 points exceed 50%.
- Every point still has clean small scales (4 is never padded; 8 at most 12.5%), and the scale reduction takes the
  minimum over scales, so a larger maximum scale adds context without removing clean options for edge points.
- Without the offset (rows/cols 15–113) padding appears only at scale 32: 80 points, at most 6% padded.
- Row 0 (the artifact) is inside the kept region for more points than the padding itself, so masking it would matter
  more than the padding as the scale grows.
