"""Instantiation and forward-pass smoke tests for nnerror network models."""

import numpy as np
import pytest
import torch

from nnerror.networks import (
    im2spec,
    im2spec_2,
    im2spec_3,
    error_model,
    ensemble_im2spec,
    FNO_im2spec,
)

FEATURE_SIZE = (8, 8)
TARGET_SIZE = 32
LATENT_DIM = 16
BATCH = 4


@pytest.fixture
def dummy_image_batch():
    # Models call x.unsqueeze(1) internally, so input is (batch, H, W)
    return torch.randn(BATCH, *FEATURE_SIZE)


class TestIm2SpecInstantiation:

    def test_im2spec_creates(self):
        model = im2spec(FEATURE_SIZE, TARGET_SIZE, LATENT_DIM)
        assert model is not None

    def test_im2spec_forward(self, dummy_image_batch):
        model = im2spec(FEATURE_SIZE, TARGET_SIZE, LATENT_DIM)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        assert out.shape == (BATCH, TARGET_SIZE)

    def test_im2spec_2_creates(self):
        model = im2spec_2(FEATURE_SIZE, TARGET_SIZE, LATENT_DIM)
        assert model is not None

    def test_im2spec_3_creates(self):
        model = im2spec_3(FEATURE_SIZE, TARGET_SIZE, LATENT_DIM)
        assert model is not None


class TestErrorModelInstantiation:

    def test_error_model_creates(self):
        model = error_model(FEATURE_SIZE)  # target_size defaults to 1 (scalar error)
        assert model is not None

    def test_error_model_forward(self, dummy_image_batch):
        model = error_model(FEATURE_SIZE)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_batch)
        assert out.shape == (BATCH, 1)


class TestEnsembleIm2SpecInstantiation:

    def test_ensemble_creates(self):
        ens = ensemble_im2spec(FEATURE_SIZE, TARGET_SIZE, force_latent_dim=LATENT_DIM)
        assert ens is not None
