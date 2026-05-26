# NN_Error

Neural-network error estimation tools for image-to-spectroscopy (`im2spec`) learning in microscopy.

This package supports workflows where a neural network first learns a structure-property correlation from image patches to local spectra, then a second neural-network error model estimates where the im2spec model is likely to be unreliable. Those predicted errors can be converted into acquisition functions for active learning, allowing the next measurements or labels to be selected from regions that are informative for improving the model.

## Overall Workflow

The central workflow has three stages:

1. **Image-to-spectrum correlation**

   An `im2spec` model is trained on paired image patches and spectra. The image patch is the structural input, and the spectrum is the target local response. The package includes convolutional im2spec models, FNO-based image-to-spectrum models, and ensemble wrappers.

2. **Neural-network error prediction**

   After the im2spec model predicts spectra, the package computes spectral error against known spectra. A separate error model is then trained to predict that error directly from image information or learned embeddings. This gives a model-based estimate of uncertainty or expected reconstruction error at unmeasured candidate locations.

3. **Active learning through acquisition functions**

   The predicted error can be transformed into acquisition values. Acquisition functions rank candidate image patches or spatial points so that the next points can be selected for labeling or measurement. This supports curiosity-driven exploration: sampling locations where the model expects high error, high information value, or a chosen balance between exploration and exploitation.

## Usage Modalities

### 1. Im2spec Correlation and Error Prediction

Use this mode when you want a single image-to-spectrum model and a downstream error predictor.

Typical steps:

- Prepare paired image patches and spectra.
- Train an im2spec model with `train_model`.
- Predict spectra with `predict_spectra`.
- Estimate spectral errors with `err_estimation`.
- Train an error model using the error targets.
- Use `plot_error_prediction`, `plot_spectra`, and related plotting helpers to inspect predicted errors and spectra.

This is the simplest workflow for testing whether local image structure contains enough information to reconstruct local spectra and whether the learned error model captures weak regions of the im2spec model.

### 2. Ensemble Model Training

Use this mode when you want multiple im2spec models trained together and ranked by validation behavior.

Typical steps:

- Construct an `ensemble_im2spec` model from multiple im2spec architectures.
- Train with `train_model_ensemble`.
- Optionally use stochastic weight averaging through the `swa` and `swa_epoch` arguments.
- Select the best member with `sort_model_idx`.
- Train either a single decoder-style error model or an ensemble error model.

The ensemble workflow helps compare model members, improve robustness, and generate model-specific error targets. The ensemble training functions can use a separate `val_dataset`; if none is provided, the training dataset is split internally into train and validation subsets.

### 3. Multiscale Analysis

Use this mode when the relevant image context may depend on spatial scale.

The multiscale utilities create augmented versions of each image patch using different central-region scales:

- `append_multiscale_data(..., append_image_type="pad")` keeps the image size fixed and zeros out image borders outside the selected central region. Recommended usage when using convolutional layers.
- `append_multiscale_data(..., append_image_type="interpolate")` crops the selected central region and resizes it back to the original image size. Recommended usage when using neuraloperators.

The chosen scale is appended to each coordinate vector, producing `(x, y, scale)` coordinates. This allows the same im2spec/error workflow to be applied across spatial context sizes. The plotting helper `plot_scale_slider` provides interactive scale-wise views of predicted error and acquisition values.

## Package Structure

```
NN_Error/
├── src/
│   └── nnerror/
│       ├── __init__.py
│       ├── training_functions.py   # training loops, error estimation, prediction, acquisition functions
│       ├── plot_functions.py       # loss, spectra, latent space, error map, and scale-slider plots
│       ├── networks/
│       │   ├── __init__.py
│       │   ├── im2spec_models.py   # convolutional im2spec and ensemble wrapper models
│       │   ├── neuralop_im2spec.py # FNO-based image-to-spectrum models
│       │   └── nn_combiners.py     # encoder and decoder combiner modules
│       └── utils/
│           ├── __init__.py
│           ├── image_utils.py      # multiscale image augmentation helpers
│           └── im2spec_dataset.py  # PyTorch datasets and image/spectrum pairing helpers
├── notebooks/                      # example workflows grouped by microscopy modality
│   ├── README.md
│   ├── PFM/
│   │   ├── BEPS_functions.py       # BEPS loading and image/spectrum pair extraction
│   │   ├── test_im2spec_error.ipynb
│   │   ├── test_fnoim2spec_error.ipynb
│   │   ├── ensemble_im2spec_training.ipynb
│   │   ├── multiscale_im2spec_error.ipynb
│   │   └── data/
│   │       └── BEPS_data/          # sample BEPS dataset
│   └── STM/
│       ├── stm_utils.py            # STM image loading and correction helpers
│       ├── CITS_Class.py           # CITS spectroscopy loading helpers
│       ├── stm_multiscale_im2spec_error.ipynb
│       └── data/
│           └── large_area/         # sample STM SXM and 3DS files
├── tests/
├── requirements.txt
└── LICENSE
```

## Credits

BEPS data: Yongtao Liu, ORNL

STM data: Dejia Kong (ORNL, PNNL), Zheng Gai (ORNL).

Feedback and inputs: Rama Vasudevan, ORNL

## Reference

Vatsavai, Aditya, Ganesh Narasimha, Yongtao Liu, Jawad Chowdhury, Jan-Chi Yang, Hiroshi Funakubo, Maxim Ziatdinov, and Rama Vasudevan. "Curiosity driven exploration to optimize structure-property learning in microscopy." *Digital Discovery* 4, no. 8 (2025): 2188-2197.
