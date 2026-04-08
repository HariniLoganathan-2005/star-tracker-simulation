"""
Synthetic Star Image Dataset Generator

Generates 10,000 labeled star images by randomly sampling spacecraft
orientations, rendering through the Virtual Camera, and saving as
numpy arrays with Euler angle labels.
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from node_a_physics.starfield import Starfield
from node_a_physics.virtual_camera import VirtualCamera


def generate_dataset(num_images=None, output_dir=None, seed=42):
    """
    Generate synthetic star images with random orientations.

    Saves:
      - images/  : individual .npy files (224×224 float32)
      - labels.npy : (N, 3) array of [pitch, roll, yaw] in degrees
      - splits.npy : integer array — 0=train, 1=val, 2=test
    """
    if num_images is None:
        num_images = config.NUM_TRAINING_IMAGES
    if output_dir is None:
        output_dir = config.TRAINING_IMAGES_DIR

    np.random.seed(seed)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[DatasetGen] Generating {num_images} synthetic star images …")
    print(f"[DatasetGen] Output: {output_dir}")

    # Load starfield + camera
    sf = Starfield()
    cam = VirtualCamera(sf)

    labels = np.zeros((num_images, 3), dtype=np.float32)

    for i in range(num_images):
        # Random orientation within configured ranges
        pitch = np.random.uniform(*config.PITCH_RANGE)
        roll = np.random.uniform(*config.ROLL_RANGE)
        yaw = np.random.uniform(*config.YAW_RANGE)

        rot = R.from_euler('xyz', [pitch, roll, yaw], degrees=True)

        # Vary noise level for augmentation
        orig_noise_std = config.NOISE_READ_STD
        config.NOISE_READ_STD = np.random.uniform(2.0, 8.0)

        img, n_stars = cam.capture(rot, add_noise=True)

        config.NOISE_READ_STD = orig_noise_std  # restore

        # Save image
        np.save(os.path.join(images_dir, f"{i:06d}.npy"), img)
        labels[i] = [pitch, roll, yaw]

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1:>6}/{num_images}]  stars={n_stars:>3}  "
                  f"euler=({pitch:+7.2f}, {roll:+7.2f}, {yaw:+7.2f})")

    # Save labels
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    # Create train / val / test splits
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    n_train = int(num_images * config.TRAIN_SPLIT)
    n_val = int(num_images * config.VAL_SPLIT)

    splits = np.zeros(num_images, dtype=np.int32)
    splits[indices[n_train:n_train + n_val]] = 1       # val
    splits[indices[n_train + n_val:]] = 2               # test
    np.save(os.path.join(output_dir, "splits.npy"), splits)

    print(f"[DatasetGen] ✓ Done — {n_train} train, {n_val} val, "
          f"{num_images - n_train - n_val} test")

    # Save a few preview images as PNG
    preview_dir = os.path.join(output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    for j in range(min(10, num_images)):
        img = np.load(os.path.join(images_dir, f"{j:06d}.npy"))
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        pil.save(os.path.join(preview_dir, f"preview_{j:03d}.png"))

    return labels, splits


# ─────────────────────── CLI ─────────────────────── #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic star dataset")
    parser.add_argument("-n", "--num", type=int, default=config.NUM_TRAINING_IMAGES,
                        help="Number of images to generate")
    args = parser.parse_args()
    generate_dataset(num_images=args.num)
