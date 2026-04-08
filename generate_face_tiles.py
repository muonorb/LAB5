# Generates face tile images from the Olivetti Faces dataset (scikit-learn).
# 400 faces of 40 people, each resized to 32x32 px and saved as PNG.
# Run once before using Face Images mode: python generate_face_tiles.py

import numpy as np
import cv2
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces

FACE_TILE_DIR = Path(__file__).parent / "face_tiles"
TILE_SIZE = 32


def main():
    FACE_TILE_DIR.mkdir(exist_ok=True)
    print("Fetching Olivetti Faces dataset (downloads once, ~1 MB)...")
    dataset = fetch_olivetti_faces(shuffle=False)
    images = dataset.images      # (400, 64, 64), float64 in [0.0, 1.0], grayscale
    targets = dataset.target     # subject IDs 0–39, 10 images each

    # Assign each of the 40 subjects a unique hue evenly spread across the color wheel.
    # 40 subjects × 9° = 360°, so every subject gets a distinct vivid color.
    NUM_SUBJECTS = 40
    hues = {s: int(s * 180 / NUM_SUBJECTS) for s in range(NUM_SUBJECTS)}  # OpenCV hue: 0-180

    written = 0
    for i, (face, subject) in enumerate(zip(images, targets)):
        # Convert float [0,1] → uint8 [0,255] grayscale
        face_u8 = (face * 255).astype(np.uint8)

        # Resize to 32x32 while still grayscale (INTER_AREA preserves detail)
        face_small = cv2.resize(face_u8, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)

        # Colorize: build an HSV image where H=subject hue, S=200 (vivid), V=face luminance.
        # This preserves all the face texture/shading but in full color.
        hsv = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        hsv[:, :, 0] = hues[subject]   # hue  — unique per person
        hsv[:, :, 1] = 200             # saturation — vivid color
        hsv[:, :, 2] = face_small      # value — original face brightness

        face_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        filename = FACE_TILE_DIR / f"face_s{subject:02d}_{i:03d}.png"
        cv2.imwrite(str(filename), face_bgr)
        written += 1

    print(f"Saved {written} face tiles to {FACE_TILE_DIR}/")
    print("You can now select 'Face Images' mode in the app.")


if __name__ == "__main__":
    main()
