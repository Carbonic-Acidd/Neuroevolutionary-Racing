"""
constants.py - every tunable number in the project, grouped by what it controls.
"""

# ----------------------------------------------------------------------
# Display / game loop
# ----------------------------------------------------------------------
SCREEN_SIZE = (1200, 800)
FPS = 60
FONT_SIZE = 26
IMAGES_DIR = "Images/"

# ----------------------------------------------------------------------
# Track colors
# ----------------------------------------------------------------------
GRASS = (34, 177, 76)          # this is MS Paint's default green
GRASS_TOLERANCE = 25           # how far off a color can be and still count
START_COLOR = (255, 255, 255)
START_TOLERANCE = 20

# ----------------------------------------------------------------------
# Drawing colors
# ----------------------------------------------------------------------
TEXT_COLOR = (255, 255, 255)
CORNER_COLOR_ALIVE = (255, 0, 255)
CORNER_COLOR_DEAD = (100, 100, 100)
SENSOR_COLOR = (0, 200, 255)

# ----------------------------------------------------------------------
# Car physics
# ----------------------------------------------------------------------
CAR_MAX_VEL = 5
CAR_ROTATION_VEL = 4
CAR_ACCELERATION = 0.1
CAR_BASE_START_HEADING = 180        # 0 = right, 90 = down, 180 = left, 270 = up
STILL_LIMIT_FRAMES = 90        # frames a car may sit still before it's killed

# ----------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------
SENSOR_ANGLES = [-60, -30, 0, 30, 60]   # degrees relative to where the car faces
SENSOR_MAX_RANGE = 200                  # pixels; anything further reads as "far"
SENSOR_STEP = 4                         # pixels between checks along each ray

# ----------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------
N_INPUTS = len(SENSOR_ANGLES)   # one per sensor
N_HIDDEN = 3
N_OUTPUTS = 2                   # steering, throttle

# ----------------------------------------------------------------------
# Evolution
# ----------------------------------------------------------------------
POPULATION_SIZE = 40
GENERATION_MAX_TIME = 20                           # seconds per track
GENERATION_MAX_FRAMES = FPS * GENERATION_MAX_TIME
PROGRESS_CHECK_TIME = 5                            # seconds before the "are you going anywhere?" check
PROGRESS_CHECK_FRAMES = FPS * PROGRESS_CHECK_TIME
MIN_PROGRESS_DISTANCE = 80                         # px from spawn; anything closer gets killed
SHOWCASE_COUNT = 5                                 # cars shown when you press S
TRAINING_TRACKS = ["track1.png", "track2.png", "track3.png"]
SINGLE_TRACK_BRAIN_FILE = "single_track_brain.npz" # best brain from --mode NN (one track)
ALL_TRACKS_BRAIN_FILE = "all_tracks_brain.npz"     # best brain from --mode autotest (every track)
N_SURVIVORS = 8
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.4
ATTRIBUTE_NAMES = ["hidden_weights", "hidden_biases", "output_weights", "output_biases"]
