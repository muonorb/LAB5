# Gradio web interface for the Image Mosaic Generator.
# Run with: python app.py  or  gradio app.py

import numpy as np
import gradio as gr

from mosaic_generator import MosaicBuilder, TileManager
from mosaic_generator.config import DEFAULT_TILE_DIR, TILE_SIZE, SUPPORTED_GRID_SIZES

TILE_PX = 32  # each tile is rendered at 32x32 pixels

_tile_manager: TileManager = None  # cached after first load


def get_tile_manager() -> TileManager:
    """Returns the tile manager, loading tiles from disk on the first call only."""
    global _tile_manager
    if _tile_manager is None:
        tm = TileManager(tile_directory=DEFAULT_TILE_DIR, tile_px=TILE_PX)
        n = tm.load_tiles()
        print(f"[app] Loaded {n} tiles at {TILE_PX}px")
        _tile_manager = tm
    return _tile_manager


def generate_mosaic(image: np.ndarray, grid_cells: int) -> tuple:
    """Builds the mosaic from the uploaded image and returns it with stats."""
    if image is None:
        return None, "Please upload an image."

    try:
        tm = get_tile_manager()
    except (FileNotFoundError, ValueError) as e:
        return None, f"Tile loading error: {e}\nRun `python generate_tiles.py` first."

    try:
        builder = MosaicBuilder(tm, grid_size=(grid_cells, grid_cells), tile_px=TILE_PX)
        mosaic, elapsed = builder.create_mosaic_timed(image)
        metrics = builder.compute_similarity(image, mosaic)
    except ValueError as e:
        return None, f"Error: {e}"

    canvas_px = grid_cells * TILE_PX
    info = (
        f"Grid: {grid_cells}×{grid_cells} cells  |  "
        f"Output: {canvas_px}×{canvas_px}px\n"
        f"Processing time: {elapsed:.3f}s\n"
        f"MSE: {metrics['mse']:.2f}  |  "
        f"SSIM: {metrics['ssim']:.4f}  (higher SSIM = more similar)"
    )
    return mosaic, info


def build_interface() -> gr.Blocks:
    """Builds and returns the Gradio UI."""
    with gr.Blocks(title="Image Mosaic Generator") as demo:
        gr.Markdown(
            """
            # 🎨 Image Mosaic Generator
            Upload an image and reconstruct it as a photomosaic using colored tile images.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy")
                grid_size_slider = gr.Slider(
                    minimum=8, maximum=64, step=8, value=32,
                    label="Grid Cells (N×N)",
                    info="Number of tiles along each axis. More = finer detail.",
                )
                run_btn = gr.Button("Generate Mosaic", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Mosaic Output", type="numpy")
                info_box = gr.Textbox(label="Stats", lines=4, interactive=False)

        run_btn.click(
            fn=generate_mosaic,
            inputs=[input_image, grid_size_slider],
            outputs=[output_image, info_box],
        )

        gr.Examples(
            examples=[
                ["sample_images/How-to-Draw-the-Scream.jpeg", 32],
                ["sample_images/hq720.jpg", 16],
                ["sample_images/cool_512.png", 64],
                ["sample_images/warm_512.png", 32],
            ],
            inputs=[input_image, grid_size_slider],
            label="Example Images (click to load)",
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_port=7860, share=False)
else:
    # Gradio looks for a `demo` variable when run with `gradio app.py`
    demo = build_interface()
