"""Unit tests for nnerror.utils image utility functions."""

import numpy as np
import pytest

from nnerror.utils import edges_zeroed_image, interpolated_center_crop, append_multiscale_data


# ---------------------------------------------------------------------------
# edges_zeroed_image
# ---------------------------------------------------------------------------

class TestEdgesZeroedImage:

    def test_output_shape_preserved(self):
        img = np.ones((16, 16))
        out = edges_zeroed_image(img, consider_pixel=8)
        assert out.shape == img.shape

    def test_border_is_zeroed(self):
        img = np.ones((16, 16))
        out = edges_zeroed_image(img, consider_pixel=8)
        # corners should be zero
        assert out[0, 0] == 0.0
        assert out[-1, -1] == 0.0

    def test_center_is_preserved(self):
        img = np.ones((16, 16))
        out = edges_zeroed_image(img, consider_pixel=8)
        assert out[8, 8] == 1.0

    def test_full_image_returned_when_consider_pixel_large(self):
        img = np.random.rand(16, 16)
        out = edges_zeroed_image(img, consider_pixel=20)
        np.testing.assert_array_equal(out, img)

    def test_raises_for_non_2d(self):
        with pytest.raises(ValueError):
            edges_zeroed_image(np.ones((4, 4, 4)), consider_pixel=2)

    def test_raises_for_consider_pixel_less_than_2(self):
        with pytest.raises(ValueError):
            edges_zeroed_image(np.ones((16, 16)), consider_pixel=1)


# ---------------------------------------------------------------------------
# interpolated_center_crop
# ---------------------------------------------------------------------------

class TestInterpolatedCenterCrop:

    def test_output_shape_default(self):
        img = np.random.rand(16, 16)
        out = interpolated_center_crop(img, consider_pixel=8)
        assert out.shape == (16, 16)

    def test_output_shape_custom_width(self):
        img = np.random.rand(16, 16)
        out = interpolated_center_crop(img, consider_pixel=8, image_width=32)
        assert out.shape == (32, 32)

    def test_full_image_when_consider_pixel_large(self):
        img = np.ones((16, 16))
        out = interpolated_center_crop(img, consider_pixel=20)
        assert out.shape == (16, 16)

    def test_raises_for_non_2d(self):
        with pytest.raises(ValueError):
            interpolated_center_crop(np.ones((4, 4, 4)), consider_pixel=2)

    def test_raises_for_consider_pixel_less_than_2(self):
        with pytest.raises(ValueError):
            interpolated_center_crop(np.ones((16, 16)), consider_pixel=1)

    def test_raises_for_image_width_less_than_1(self):
        with pytest.raises(ValueError):
            interpolated_center_crop(np.ones((16, 16)), consider_pixel=8, image_width=0)


# ---------------------------------------------------------------------------
# append_multiscale_data
# ---------------------------------------------------------------------------

class TestAppendMultiscaleData:

    def _make_data(self, n=10, h=16, w=16, s=32):
        images = np.random.rand(n, h, w).astype(np.float32)
        spectra = np.random.rand(n, s).astype(np.float32)
        coords = np.random.rand(n, 2).astype(np.float32)
        return images, spectra, coords

    def test_output_shapes_pad(self):
        images, spectra, coords = self._make_data(n=10)
        scales = [6, 10]
        aug_imgs, aug_spec, aug_coords = append_multiscale_data(
            images, spectra, scales, coords, include_ori_set=True, append_image_type="pad"
        )
        # 2 scales + original = 3 * 10 samples
        assert aug_imgs.shape == (30, 16, 16)
        assert aug_spec.shape == (30, 32)
        assert aug_coords.shape == (30, 3)

    def test_output_shapes_interpolate(self):
        images, spectra, coords = self._make_data(n=5)
        scales = [8]
        aug_imgs, aug_spec, aug_coords = append_multiscale_data(
            images, spectra, scales, coords, include_ori_set=False, append_image_type="interpolate"
        )
        assert aug_imgs.shape == (5, 16, 16)
        assert aug_spec.shape == (5, 32)
        assert aug_coords.shape == (5, 3)

    def test_scale_appended_to_coordinates(self):
        images, spectra, coords = self._make_data(n=4)
        scales = [6]
        _, _, aug_coords = append_multiscale_data(
            images, spectra, scales, coords, include_ori_set=False
        )
        # third column should equal the scale value (rounded up to even: 6 stays 6)
        assert np.all(aug_coords[:, 2] == 6)

    def test_raises_for_invalid_append_type(self):
        images, spectra, coords = self._make_data(n=4)
        with pytest.raises(ValueError):
            append_multiscale_data(images, spectra, [6], coords, append_image_type="unknown")

    def test_raises_when_scale_too_large(self):
        images, spectra, coords = self._make_data(n=4)
        with pytest.raises(ValueError):
            append_multiscale_data(images, spectra, [20], coords)
