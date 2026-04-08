import cv2
import numpy as np
from pathlib import Path

from .config import DEFAULT_IMAGE_SIZE, RESIZE_INTERPOLATION
from .utils import validate_image, validate_grid_size


def load_image(path: str) -> np.ndarray:
    """Loads an image from disk and returns it as an RGB array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    target_size: int = DEFAULT_IMAGE_SIZE,
    grid_cells: int = 32,
    tile_px: int = 32,
) -> np.ndarray:
    """Resizes and center-crops the image to fit the grid exactly (grid_cells * tile_px)."""
    canvas = grid_cells * tile_px
    h, w = image.shape[:2]

    # Scale so the short side fills the canvas, then crop the center
    scale = canvas / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=RESIZE_INTERPOLATION)

    y0 = (new_h - canvas) // 2
    x0 = (new_w - canvas) // 2
    return resized[y0: y0 + canvas, x0: x0 + canvas]


def extract_grid_cells(
    image: np.ndarray, grid_cells: int, tile_px: int
) -> np.ndarray:
    """Splits the image into a grid and returns all cells as one array. No loops used."""
    validate_image(image)
    validate_grid_size(image, grid_cells)

    g, t = grid_cells, tile_px
    # Reshape the image into a grid then reorder axes to get individual cells
    cells = (
        image
        .reshape(g, t, g, t, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, t, t, 3)
    )
    return cells


def compute_cell_avg_colors(cells: np.ndarray) -> np.ndarray:
    """Returns the average RGB color of each cell. Output shape: (N, 3)."""
    return cells.reshape(len(cells), -1, 3).mean(axis=1, dtype=np.float32)


def compute_canvas_avg_colors(canvas: np.ndarray, grid_cells: int, tile_px: int) -> np.ndarray:
    """
    Returns the average color of every grid cell by shrinking the image with OpenCV.
    Faster than extract_grid_cells + compute_cell_avg_colors.
    """
    # Shrinking with INTER_AREA makes each output pixel the average of one tile block
    small = cv2.resize(canvas, (grid_cells, grid_cells), interpolation=cv2.INTER_AREA)
    return small.reshape(-1, 3).astype(np.float32)


def reconstruct_image(
    tile_batch: np.ndarray, grid_cells: int, tile_px: int
) -> np.ndarray:
    """Stitches a batch of tiles back together into one full image."""
    g, t = grid_cells, tile_px
    tile_grid = tile_batch.reshape(g, g, t, t, 3)
    # Join tiles row by row, then stack all rows
    rows = [np.concatenate(tile_grid[i], axis=1) for i in range(g)]
    return np.concatenate(rows, axis=0)
