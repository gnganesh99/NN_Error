"""Smoke tests: verify all public symbols can be imported without error."""

import pytest


def test_top_level_import():
    import nnerror


def test_submodule_imports():
    from nnerror import networks, plot_functions, training_functions, utils


def test_network_classes():
    from nnerror.networks import (
        im2spec,
        im2spec_2,
        im2spec_3,
        im2spec_4,
        im2spec_5,
        ensemble_im2spec,
        Swa_Ensemble,
        Encoder_Wrapper,
        error_model,
        ensemble_error_model,
        DecoderModule,
        CustomDecoder,
    )


def test_fno_classes():
    from nnerror.networks import (
        FNOEncoder,
        FNO_im2spec,
        FNOEncoderWrapper,
    )


def test_combiner_classes():
    from nnerror.networks import (
        CombinerCustomDecoder,
        CombinerDecoderModule,
        CombinerEncoderWrapper,
        CombinerSwaEnsemble,
        CombinerEnsembleErrorModel,
        CombinerErrorModel,
    )


def test_training_functions():
    from nnerror import (
        ELBOLoss,
        EarlyStopping,
        EarlyStopping_ensemble,
        EarlyStopping_ensemble_swatrigger,
        acquisition_fn,
        append_training_set,
        calc_distance,
        distance_acq_fn,
        distance_distribution,
        err_estimation,
        error_dataset,
        l1_regularization,
        norm_0to1,
        predict_embedding,
        predict_error,
        predict_posterior,
        predict_spectra,
        predict_vae_embedding,
        sort_model_idx,
        train_error_ensemble,
        train_model,
        train_model_ensemble,
        vae_loss_mse,
    )


def test_plot_functions():
    from nnerror import (
        cluster_latent_space,
        plot_embedding,
        plot_error_prediction,
        plot_latent_distribution,
        plot_latent_space,
        plot_only_training_loss,
        plot_scale_slider,
        plot_spectra,
        plot_training_loss,
    )


def test_utils():
    from nnerror import (
        append_multiscale_data,
        edges_zeroed_image,
        interpolated_center_crop,
    )
