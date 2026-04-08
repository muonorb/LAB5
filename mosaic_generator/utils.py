import time
import functools
import numpy as np


def timeit(func):
    """Wraps a function to print how long it took and return (result, seconds)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timeit] {func.__name__} took {elapsed:.4f}s")
        return result, elapsed
    return wrapper


def validate_grid_size(image: np.ndarray, grid_cells: int) -> None:
    """Checks the grid fits inside the image — raises ValueError if not."""
    if grid_cells <= 0:
        raise ValueError(f"grid_cells must be positive, got {grid_cells}")
    h, w = image.shape[:2]
    if h < grid_cells or w < grid_cells:
        raise ValueError(
            f"Image ({h}×{w}) is too small for a {grid_cells}×{grid_cells} grid. "
            f"Each cell would be less than 1 pixel."
        )


def validate_image(image: np.ndarray) -> None:
    """Checks the input is a valid RGB numpy array — raises an error if not."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image of shape (H, W, 3), got shape {image.shape}"
        )


def clamp(value: float, low: float, high: float) -> float:
    """Keeps a number between low and high."""
    return max(low, min(high, value))
