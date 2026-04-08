# Gradio web interface for the Image Mosaic Generator.
# Run with: python app.py  or  gradio app.py

import time
import numpy as np
import cv2
import gradio as gr

from mosaic_generator import MosaicBuilder, TileManager, resize_image
from mosaic_generator.config import DEFAULT_TILE_DIR, TILE_SIZE

TILE_PX = 32
FACE_TILE_DIR = "face_tiles"

_tile_manager: TileManager = None
_face_manager: TileManager = None


def get_tile_manager() -> TileManager:
    """Returns the color tile manager, loading tiles from disk on the first call only."""
    global _tile_manager
    if _tile_manager is None:
        tm = TileManager(tile_directory=DEFAULT_TILE_DIR, tile_px=TILE_PX)
        n = tm.load_tiles()
        print(f"[app] Loaded {n} color tiles at {TILE_PX}px")
        _tile_manager = tm
    return _tile_manager


def get_face_manager() -> TileManager:
    """Returns the face tile manager, loading tiles from disk on the first call only."""
    global _face_manager
    if _face_manager is None:
        tm = TileManager(tile_directory=FACE_TILE_DIR, tile_px=TILE_PX)
        n = tm.load_tiles()
        print(f"[app] Loaded {n} face tiles at {TILE_PX}px")
        _face_manager = tm
    return _face_manager


def draw_grid(canvas: np.ndarray, grid_cells: int, tile_px: int) -> np.ndarray:
    """Draws white grid lines on the canvas to show cell boundaries."""
    out = canvas.copy()
    size = grid_cells * tile_px
    for i in range(0, size + 1, tile_px):
        cv2.line(out, (i, 0), (i, size - 1), (255, 255, 255), 1)
        cv2.line(out, (0, i), (size - 1, i), (255, 255, 255), 1)
    return out


def generate_mosaic(image: np.ndarray, grid_cells: int, mode: str) -> tuple:
    """
    Builds the mosaic from the uploaded image.
    Returns: cropped image, segmented image (with grid), mosaic image, stats text.
    """
    if image is None:
        return None, None, None, "Please upload an image."

    use_faces = (mode == "Face Images")

    try:
        tm = get_face_manager() if use_faces else get_tile_manager()
    except (FileNotFoundError, ValueError) as e:
        script = "generate_face_tiles.py" if use_faces else "generate_tiles.py"
        return None, None, None, (
            f"Tile loading error: {e}\n"
            f"Run `python {script}` first to generate the tile library."
        )

    try:
        # Step 1: crop/resize to the exact grid canvas
        canvas = resize_image(image, grid_cells=grid_cells, tile_px=TILE_PX)

        # Step 2: draw grid lines on the canvas to show segmentation
        segmented = draw_grid(canvas, grid_cells, TILE_PX)

        # Step 3: build the mosaic
        builder = MosaicBuilder(tm, grid_size=(grid_cells, grid_cells), tile_px=TILE_PX)
        t0 = time.perf_counter()
        mosaic = builder.create_mosaic(image)
        elapsed = time.perf_counter() - t0
        metrics = builder.compute_similarity(image, mosaic)

    except ValueError as e:
        return None, None, None, f"Error: {e}"

    canvas_px = grid_cells * TILE_PX
    info = (
        f"Mode: {mode}  |  Grid: {grid_cells}×{grid_cells} cells  |  "
        f"Output: {canvas_px}×{canvas_px}px\n"
        f"Processing time: {elapsed:.3f}s\n"
        f"MSE: {metrics['mse']:.2f}  |  "
        f"SSIM: {metrics['ssim']:.4f}  (higher SSIM = more similar)"
    )
    return canvas, segmented, mosaic, info


def build_interface() -> gr.Blocks:
    """Builds and returns the Gradio UI."""
    with gr.Blocks(title="Image Mosaic Generator") as demo:
        gr.Markdown(
            """
            # Image Mosaic Generator
            Upload an image to see it cropped, segmented into a grid, then reconstructed as a photomosaic.
            """
        )

        with gr.Row():
            # ── Left column: controls ──────────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="numpy")
                grid_size_slider = gr.Slider(
                    minimum=8, maximum=64, step=8, value=32,
                    label="Grid Cells (N×N)",
                    info="Number of tiles along each axis. More = finer detail.",
                )
                mode_radio = gr.Radio(
                    choices=["Color Tiles", "Face Images"],
                    value="Color Tiles",
                    label="Mosaic Mode",
                    info=(
                        "Color Tiles: synthetic colored squares.  "
                        "Face Images: Olivetti face photos (run generate_face_tiles.py first)."
                    ),
                )
                run_btn = gr.Button("Generate Mosaic", variant="primary")

            # ── Right column: tabbed output images + stats ────────────────
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("1. Scaled & Cropped"):
                        cropped_image = gr.Image(type="numpy", show_label=False)
                    with gr.Tab("2. Segmented Grid"):
                        segmented_image = gr.Image(type="numpy", show_label=False)
                    with gr.Tab("3. Mosaic Output"):
                        mosaic_image = gr.Image(type="numpy", show_label=False)
                info_box = gr.Textbox(label="Stats", lines=3, interactive=False)

        run_btn.click(
            fn=generate_mosaic,
            inputs=[input_image, grid_size_slider, mode_radio],
            outputs=[cropped_image, segmented_image, mosaic_image, info_box],
        )

        gr.Examples(
            examples=[
                ["sample_images/Sample1.jpeg", 32, "Color Tiles"],
                ["sample_images/Sample2.jpg", 16, "Color Tiles"],
                ["sample_images/Sample3.webp", 32, "Face Images"],
            ],
            inputs=[input_image, grid_size_slider, mode_radio],
            label="Example Images (click to load)",
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_port=7860, share=False)
else:
    # Gradio looks for a `demo` variable when run with `gradio app.py`
    demo = build_interface()
