# Builds the mosaic: matches each grid cell to the closest tile, then stitches them together.

import time
import numpy as np
from typing import Tuple, Dict, Any

from .tile_manager import TileManager
from .image_processor import (
    resize_image,
    extract_grid_cells,
    compute_cell_avg_colors,
    compute_canvas_avg_colors,
    reconstruct_image,
)
from .metrics import compute_all_metrics
from .utils import validate_image, validate_grid_size


class MosaicBuilder:
    """Replaces each grid cell in an image with the closest matching tile."""

    def __init__(
        self,
        tile_manager: TileManager,
        grid_size: Tuple[int, int] = (32, 32),
        tile_px: int = 32,
    ):
        """Sets up the builder and caches tile data for fast access."""
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        self.tile_manager = tile_manager
        self.grid_size = grid_size
        self.tile_px = tile_px

        self._tiles = tile_manager.get_all_tiles()         # (T, t, t, 3)
        self._tile_colors = tile_manager.get_avg_colors()  # (T, 3)
        self._tile_sq = (self._tile_colors * self._tile_colors).sum(axis=1)  # (T,)

    def create_mosaic(self, image: np.ndarray) -> np.ndarray:
        """Takes an input image and returns the mosaic version."""
        validate_image(image)
        grid_cells = self.grid_size[0]

        canvas = resize_image(image, grid_cells=grid_cells, tile_px=self.tile_px)
        cell_colors = compute_canvas_avg_colors(canvas, grid_cells, self.tile_px)
        indices = self._match_tiles_vectorized(cell_colors)
        matched = self._tiles[indices]
        return reconstruct_image(matched, grid_cells, self.tile_px)

    def create_mosaic_timed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Same as create_mosaic but also returns elapsed time in seconds."""
        t0 = time.perf_counter()
        mosaic = self.create_mosaic(image)
        return mosaic, time.perf_counter() - t0

    def compute_similarity(
        self, original: np.ndarray, mosaic: np.ndarray
    ) -> Dict[str, float]:
        """Returns MSE and SSIM scores comparing the original to the mosaic."""
        import cv2
        h, w = mosaic.shape[:2]
        orig_resized = cv2.resize(original, (w, h), interpolation=3)
        return compute_all_metrics(orig_resized, mosaic)

    def _match_tiles_vectorized(self, cell_colors: np.ndarray) -> np.ndarray:
        """Finds the closest tile for every cell using vectorized L2 distance.

        Uses: ||a - b||² = ||a||² + ||b||² - 2(a · b)
        """
        cell_sq = (cell_colors * cell_colors).sum(axis=1, keepdims=True)  # (N, 1)
        cross = cell_colors @ self._tile_colors.T                          # (N, T)
        dists = cell_sq + self._tile_sq[None, :] - 2.0 * cross            # (N, T)
        return np.argmin(dists, axis=1)
