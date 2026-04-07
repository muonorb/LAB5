# Generates 87 colored tile images and saves them to the tiles/ folder.
# Run once before starting the app: python generate_tiles.py

import os
import numpy as np
from pathlib import Path
import cv2

TILE_DIR = Path(__file__).parent / "tiles"
TILE_SIZE = 32  # each tile is 32x32 pixels


def make_solid_tile(r: int, g: int, b: int) -> np.ndarray:
    """Creates a solid-color tile filled with the given RGB color."""
    return np.full((TILE_SIZE, TILE_SIZE, 3), [b, g, r], dtype=np.uint8)  # OpenCV uses BGR


def make_gradient_tile(r1, g1, b1, r2, g2, b2) -> np.ndarray:
    """Creates a tile that fades horizontally from one color to another."""
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    for x in range(TILE_SIZE):
        t = x / (TILE_SIZE - 1)
        tile[:, x] = [int(b1 + t * (b2 - b1)), int(g1 + t * (g2 - g1)), int(r1 + t * (r2 - r1))]
    return tile


def main():
    TILE_DIR.mkdir(exist_ok=True)
    tiles = []

    # 16 grayscale shades from black to white
    for i in range(16):
        v = int(i * 255 / 15)
        tiles.append((f"gray_{v:03d}", make_solid_tile(v, v, v)))

    # 12 hues × 3 saturation levels = 36 colorful tiles
    for hue_deg in range(0, 360, 30):
        h = hue_deg / 2  # OpenCV hue range is 0–180
        for s in [128, 200, 255]:
            hsv = np.full((TILE_SIZE, TILE_SIZE, 3), [h, s, 220], dtype=np.uint8)
            tiles.append((f"hue_{hue_deg:03d}_s{s}", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))

    # 8 pastel tiles
    for hue_deg in range(0, 360, 45):
        h = hue_deg / 2
        hsv = np.full((TILE_SIZE, TILE_SIZE, 3), [h, 80, 240], dtype=np.uint8)
        tiles.append((f"pastel_{hue_deg:03d}", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))

    # 8 dark tiles
    for hue_deg in range(0, 360, 45):
        h = hue_deg / 2
        hsv = np.full((TILE_SIZE, TILE_SIZE, 3), [h, 180, 100], dtype=np.uint8)
        tiles.append((f"dark_{hue_deg:03d}", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))

    # 9 earth and skin tones
    earth = [
        (210, 180, 140), (160, 120, 90), (100, 70, 50),
        (230, 200, 170), (180, 140, 100), (120, 90, 60),
        (245, 220, 190), (200, 160, 120), (70, 50, 35),
    ]
    for i, (r, g, b) in enumerate(earth):
        tiles.append((f"earth_{i:02d}", make_solid_tile(r, g, b)))

    # 6 neon colors
    neons = [
        (255, 0, 128), (0, 255, 128), (128, 0, 255),
        (255, 128, 0), (0, 128, 255), (128, 255, 0),
    ]
    for i, (r, g, b) in enumerate(neons):
        tiles.append((f"neon_{i:02d}", make_solid_tile(r, g, b)))

    # 4 gradient tiles
    gradient_pairs = [
        ((255, 0, 0), (0, 0, 255)),
        ((0, 255, 0), (255, 255, 0)),
        ((0, 0, 0), (255, 255, 255)),
        ((255, 128, 0), (0, 128, 255)),
    ]
    for i, ((r1, g1, b1), (r2, g2, b2)) in enumerate(gradient_pairs):
        tiles.append((f"grad_{i:02d}", make_gradient_tile(r1, g1, b1, r2, g2, b2)))

    written = 0
    for name, tile_bgr in tiles:
        cv2.imwrite(str(TILE_DIR / f"{name}.png"), tile_bgr)
        written += 1

    print(f"Generated {written} tiles in {TILE_DIR}/")


if __name__ == "__main__":
    main()
