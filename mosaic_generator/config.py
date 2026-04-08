# Default settings for the project

TILE_SIZE = 32                      # size of each tile in pixels
DEFAULT_GRID_CELLS = 32             # default number of tiles per row/col
SUPPORTED_GRID_SIZES = [16, 32, 64] # grid sizes available in the UI
DEFAULT_IMAGE_SIZE = 512            # default image size before processing
DEFAULT_TILE_DIR = "tiles"          # folder where tile images are stored
SAMPLE_IMAGES_DIR = "sample_images" # folder where test images are stored
RESIZE_INTERPOLATION = 3            # cv2.INTER_AREA — best for shrinking images
COLOR_SPACE = "RGB"                 # color system used for tile matching
N_DOMINANT_COLORS = 1               # number of colors used per tile for matching
GRADIO_SERVER_PORT = 7860           # port the web app runs on
GRADIO_SHARE = False                # set True to get a public Gradio link
