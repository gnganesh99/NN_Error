# Notebooks

This folder contains example workflows for applying `nnerror` to microscopy image-to-spectroscopy learning, neural-network error prediction, and active learning.

The notebooks assume the package source is available from the repository `src/` directory and use helper scripts colocated with each microscopy workflow.

## Folder Layout

- `PFM/`: BEPS/PFM workflows, BEPS helper functions, and BEPS sample data.
- `STM/`: STM/CITS workflows, STM helper functions, and STM sample data.

## PFM/BEPS Analysis

### `PFM/test_im2spec_error.ipynb`

Basic BEPS image-to-spectrum example using the standard im2spec model. It demonstrates loading BEPS data, extracting image/spectrum pairs, training an im2spec model, estimating prediction error, and visualizing spectra and error maps.

### `PFM/test_fnoim2spec_error.ipynb`

BEPS image-to-spectrum example using the FNO-based im2spec model. It follows the same general single-model error-estimation workflow while replacing the im2spec model architecture with `FNO_im2spec`.

### `PFM/ensemble_im2spec_training.ipynb`

BEPS active-learning workflow based on ensemble im2spec training. It trains an ensemble of im2spec models, selects a best-performing model member, trains a neural-network error model, predicts error across candidate patches, and uses acquisition functions to choose new points.

### `PFM/multiscale_im2spec_error.ipynb`

BEPS multiscale workflow. It augments image/spectrum pairs across multiple spatial scales, trains im2spec and error models on the augmented data, and uses `(x, y, scale)` coordinates to inspect predicted error and acquisition behavior across scale.

## STM Analysis

### `STM/stm_multiscale_im2spec_error.ipynb`

STM multiscale image-to-spectrum workflow using `.sxm` morphology data and `.3ds` CITS spectroscopy data. It pairs STM image patches with spectra, augments the data across spatial scales, trains im2spec and error models, and evaluates active-learning acquisition behavior.

## Helper Scripts

- `PFM/BEPS_functions.py`: BEPS `.npz` loading and image/spectrum pair extraction.
- `STM/stm_utils.py`: STM `.sxm` loading and image correction helpers.
- `STM/CITS_Class.py`: CITS `.3ds` loading and spatial/voltage lookup helpers.

PyTorch datasets and image/spectrum pairing helpers are now provided by
`nnerror.utils.im2spec_dataset`.

## Data

Example data files are stored with their corresponding workflow folders. The BEPS examples use `notebooks/PFM/data/BEPS_data/PTO_BEPS_0p85um.npz`; the STM example uses the SXM and 3DS files in `notebooks/STM/data/large_area/`.
