"""
Unit tests for the mosaic_generator package.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mosaic_generator.image_processor import (
    resize_image,
    extract_grid_cells,
    compute_cell_avg_colors,
    reconstruct_image,
)
from mosaic_generator.metrics import compute_mse, compute_ssim
from mosaic_generator.utils import validate_image, validate_grid_size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_image():
    """Return a random 64×64 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def solid_image():
    """Return a solid-color 64×64 RGB image."""
    return np.full((64, 64, 3), [128, 64, 32], dtype=np.uint8)


# ---------------------------------------------------------------------------
# image_processor tests
# ---------------------------------------------------------------------------

class TestResizeImage:
    def test_output_is_square_canvas(self, random_image):
        grid_cells, tile_px = 4, 16
        result = resize_image(random_image, grid_cells=grid_cells, tile_px=tile_px)
        assert result.shape == (grid_cells * tile_px, grid_cells * tile_px, 3)

    def test_accepts_non_square_input(self):
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = resize_image(img, grid_cells=4, tile_px=8)
        assert result.shape == (32, 32, 3)

    def test_output_dtype_is_uint8(self, random_image):
        result = resize_image(random_image, grid_cells=4, tile_px=8)
        assert result.dtype == np.uint8


class TestExtractGridCells:
    def test_output_shape(self):
        grid_cells, tile_px = 4, 8
        img = np.zeros((grid_cells * tile_px, grid_cells * tile_px, 3), dtype=np.uint8)
        cells = extract_grid_cells(img, grid_cells, tile_px)
        assert cells.shape == (grid_cells * grid_cells, tile_px, tile_px, 3)

    def test_solid_image_all_same(self):
        grid_cells, tile_px = 4, 8
        img = np.full((grid_cells * tile_px, grid_cells * tile_px, 3), 200, dtype=np.uint8)
        cells = extract_grid_cells(img, grid_cells, tile_px)
        assert np.all(cells == 200)

    def test_invalid_image_raises(self):
        with pytest.raises(TypeError):
            extract_grid_cells([[1, 2], [3, 4]], 2, 2)


class TestComputeCellAvgColors:
    def test_shape(self):
        cells = np.zeros((16, 8, 8, 3), dtype=np.uint8)
        avg = compute_cell_avg_colors(cells)
        assert avg.shape == (16, 3)

    def test_solid_cells_exact_mean(self):
        cells = np.full((4, 8, 8, 3), 100, dtype=np.uint8)
        avg = compute_cell_avg_colors(cells)
        assert np.allclose(avg, 100.0)


class TestReconstructImage:
    def test_roundtrip(self):
        """extract_grid_cells then reconstruct_image should give back the original."""
        grid_cells, tile_px = 4, 8
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (grid_cells * tile_px, grid_cells * tile_px, 3), dtype=np.uint8)
        cells = extract_grid_cells(img, grid_cells, tile_px)
        reconstructed = reconstruct_image(cells, grid_cells, tile_px)
        assert np.array_equal(img, reconstructed)

    def test_output_shape(self):
        grid_cells, tile_px = 4, 8
        tiles = np.zeros((grid_cells * grid_cells, tile_px, tile_px, 3), dtype=np.uint8)
        result = reconstruct_image(tiles, grid_cells, tile_px)
        assert result.shape == (grid_cells * tile_px, grid_cells * tile_px, 3)


# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mse_identical_images_is_zero(self, random_image):
        assert compute_mse(random_image, random_image) == pytest.approx(0.0)

    def test_mse_different_images_positive(self, random_image):
        other = np.zeros_like(random_image)
        assert compute_mse(random_image, other) > 0

    def test_ssim_identical_is_one(self, random_image):
        assert compute_ssim(random_image, random_image) == pytest.approx(1.0, abs=1e-4)

    def test_ssim_range(self, random_image):
        other = np.random.randint(0, 256, random_image.shape, dtype=np.uint8)
        score = compute_ssim(random_image, other)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# utils tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_image_ok(self, random_image):
        validate_image(random_image)  # Should not raise

    def test_validate_image_wrong_channels(self):
        with pytest.raises(ValueError):
            validate_image(np.zeros((10, 10, 4), dtype=np.uint8))

    def test_validate_image_not_array(self):
        with pytest.raises(TypeError):
            validate_image("not an array")

    def test_validate_grid_size_ok(self, random_image):
        validate_grid_size(random_image, 4)  # Should not raise

    def test_validate_grid_size_zero(self, random_image):
        with pytest.raises(ValueError):
            validate_grid_size(random_image, 0)

    def test_validate_grid_size_too_large(self):
        small_img = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_grid_size(small_img, 8)



