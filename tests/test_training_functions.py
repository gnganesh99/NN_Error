"""Unit tests for stateless helpers in nnerror.training_functions."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nnerror import norm_0to1, l1_regularization, ELBOLoss, EarlyStopping


class TestNorm0to1:

    def test_min_is_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = norm_0to1(arr)
        assert out.min() == pytest.approx(0.0)

    def test_max_is_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = norm_0to1(arr)
        assert out.max() == pytest.approx(1.0)

    def test_preserves_shape(self):
        arr = np.random.rand(5, 10)
        out = norm_0to1(arr)
        assert out.shape == arr.shape

    def test_constant_array(self):
        arr = np.ones((4, 4))
        out = norm_0to1(arr)
        # all-same values produce nan from 0/0 — just check it returns an array
        assert out.shape == arr.shape


class TestL1Regularization:

    def test_returns_scalar_tensor(self):
        model = nn.Linear(4, 4)
        penalty = l1_regularization(model, l1_lambda=1e-4)
        assert penalty.ndim == 0

    def test_penalty_is_positive(self):
        model = nn.Linear(4, 4)
        penalty = l1_regularization(model)
        assert penalty.item() > 0

    def test_zero_weights_give_zero_penalty(self):
        model = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(model.weight)
        penalty = l1_regularization(model)
        assert penalty.item() == pytest.approx(0.0)

    def test_lambda_scales_penalty(self):
        model = nn.Linear(4, 4)
        p1 = l1_regularization(model, l1_lambda=1.0)
        p2 = l1_regularization(model, l1_lambda=2.0)
        assert p2.item() == pytest.approx(2.0 * p1.item())


class TestELBOLoss:

    def _dummy_inputs(self, batch=4, size=16):
        pred = torch.randn(batch, size)
        target = torch.randn(batch, size)
        mu = torch.randn(batch, 8)
        logvar = torch.randn(batch, 8)
        return pred, target, mu, logvar

    def test_returns_scalar(self):
        loss_fn = ELBOLoss()
        pred, target, mu, logvar = self._dummy_inputs()
        loss = loss_fn((pred, mu, logvar), target)
        assert loss.ndim == 0

    def test_beta_zero_equals_recon_loss(self):
        recon_fn = nn.MSELoss()
        loss_fn = ELBOLoss(recon_loss_fn=recon_fn, beta_elbo=0.0)
        pred, target, mu, logvar = self._dummy_inputs()
        elbo = loss_fn((pred, mu, logvar), target)
        recon = recon_fn(pred, target)
        assert elbo.item() == pytest.approx(recon.item(), rel=1e-5)


class TestEarlyStopping:

    def test_no_stop_when_improving(self):
        # skip_epochs=0 so stopping logic is active from epoch 0
        es = EarlyStopping(skip_epochs=0, patience=3)
        for epoch, val_loss in enumerate([1.0, 0.9, 0.8, 0.7]):
            result = es(val_loss, epoch)
        assert result is False

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(skip_epochs=0, patience=3)
        es(1.0, 0)
        result = False
        for epoch in range(1, 5):
            result = es(1.0, epoch)  # no improvement
        assert result is True
