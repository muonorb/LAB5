# Loads tile images from disk and stores them in memory.

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class TileManager:
    """Loads and stores tile images, and pre-computes their average colors."""

    def __init__(self, tile_directory: str, tile_px: int = 32):
        """Sets up the manager. Does not load tiles yet."""
        self.tile_directory = Path(tile_directory)
        self.tile_px = tile_px
        self._tiles: Optional[np.ndarray] = None       # all tiles, shape (N, tile_px, tile_px, 3)
        self._avg_colors: Optional[np.ndarray] = None  # average color per tile, shape (N, 3)
        self._names: List[str] = []                    # filenames of loaded tiles

    def load_tiles(self) -> int:
        """Reads all tile images from the folder. Returns how many were loaded."""
        if not self.tile_directory.exists():
            raise FileNotFoundError(f"Tile directory not found: {self.tile_directory}")

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths = sorted(p for p in self.tile_directory.iterdir() if p.suffix.lower() in exts)

        if not paths:
            raise ValueError(f"No tile images found in: {self.tile_directory}")

        t = self.tile_px
        loaded = []
        names = []
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)
            loaded.append(img)
            names.append(path.name)

        if not loaded:
            raise ValueError("All tile files failed to load.")

        self._tiles = np.stack(loaded, axis=0)  # (N, t, t, 3)
        self._names = names
        self._avg_colors = (
            self._tiles.reshape(len(loaded), -1, 3)
            .mean(axis=1)
            .astype(np.float32)
        )
        return len(loaded)

    def get_avg_colors(self) -> np.ndarray:
        """Returns average color per tile. Shape: (N, 3), float32."""
        self._require_loaded()
        return self._avg_colors

    def get_tile(self, index: int) -> np.ndarray:
        """Returns one tile image by index. Shape: (tile_px, tile_px, 3)."""
        self._require_loaded()
        return self._tiles[index]

    def get_all_tiles(self) -> np.ndarray:
        """Returns all tile images. Shape: (N, tile_px, tile_px, 3)."""
        self._require_loaded()
        return self._tiles

    def num_tiles(self) -> int:
        """Returns how many tiles are loaded."""
        if self._tiles is None:
            return 0
        return len(self._tiles)

    def tile_names(self) -> List[str]:
        """Returns filenames of all loaded tiles."""
        return list(self._names)

    def _require_loaded(self) -> None:
        # Raises an error if tiles haven't been loaded yet
        if self._tiles is None:
            raise RuntimeError("Tiles not loaded. Call load_tiles() first.")
