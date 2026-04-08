"""
Global configuration for the Digital Twin Spacecraft Simulation.
All constants, paths, and tuneable parameters live here.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HIPPARCOS_CACHE_DIR = os.path.join(DATA_DIR, "hipparcos_cache")
TRAINING_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
MODEL_DIR = os.path.join(DATA_DIR, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# Create directories
for d in [DATA_DIR, HIPPARCOS_CACHE_DIR, TRAINING_IMAGES_DIR, MODEL_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Star Sensor / Camera
# ──────────────────────────────────────────────
CAMERA_FOV_DEG = 12.0                       # Field of view in degrees
CAMERA_FOV_RAD = np.radians(CAMERA_FOV_DEG)
IMAGE_SIZE = 224                             # Pixels (square), matches ResNet input
STAR_MAGNITUDE_LIMIT = 6.5                   # Cut-off; ~9,000 stars from Hipparcos
STAR_SPHERE_RADIUS = 100.0                   # Radius of the rendered starfield sphere

# Camera intrinsics (pinhole model)
FOCAL_LENGTH_PX = IMAGE_SIZE / (2.0 * np.tan(CAMERA_FOV_RAD / 2.0))

# Noise parameters for synthetic images
NOISE_READ_STD = 5.0      # Gaussian read noise (ADU)
NOISE_DARK_CURRENT = 2.0  # Mean dark current (ADU / pixel)
STAR_PSF_SIGMA = 1.5      # Gaussian PSF sigma (pixels)

# ──────────────────────────────────────────────
# Mars Trajectory (Target Orientation)
# ──────────────────────────────────────────────
# The "correct" spacecraft orientation pointing toward Mars.
# Represented as Euler angles (degrees) for readability.
MARS_TARGET_EULER_DEG = np.array([0.0, 0.0, 0.0])  # pitch, roll, yaw
MARS_TARGET_ROTATION = R.from_euler(
    'xyz', MARS_TARGET_EULER_DEG, degrees=True
)

# ──────────────────────────────────────────────
# AI / Training
# ──────────────────────────────────────────────
NUM_TRAINING_IMAGES = 10_000
TRAIN_SPLIT = 0.8          # 80% train
VAL_SPLIT = 0.1            # 10% validation
TEST_SPLIT = 0.1           # 10% test
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "star_cnn_best.pth")

# Orientation sampling range for dataset generation (degrees)
PITCH_RANGE = (-90.0, 90.0)
ROLL_RANGE = (-180.0, 180.0)
YAW_RANGE = (-180.0, 180.0)

# ──────────────────────────────────────────────
# Simulation / UI
# ──────────────────────────────────────────────
UI_UPDATE_RATE_HZ = 20          # Dashboard refresh rate
UI_UPDATE_INTERVAL_MS = int(1000 / UI_UPDATE_RATE_HZ)

# Deviation thresholds (degrees)
THRESHOLD_NOMINAL = 1.0         # < 1°  → GREEN
THRESHOLD_WARNING = 5.0         # 1-5°  → YELLOW
                                # > 5°  → RED / CRITICAL

# Auto-recovery
SLERP_STEPS = 60                # Animation frames for auto-recovery
SLERP_DURATION_S = 3.0          # Total recovery time in seconds

# ──────────────────────────────────────────────
# Device (auto-detect GPU)
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
