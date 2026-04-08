"""
Virtual Star Camera — Pinhole Projection + Noise Model

Given a quaternion orientation, projects visible stars onto a 2D image
plane using a pinhole camera model with realistic Gaussian PSF and noise.
This is the bridge between the Physical Engine and the AI Perception Layer.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class VirtualCamera:
    """Simulates a star‑tracker camera mounted on the spacecraft nose."""

    def __init__(self, starfield):
        """
        Parameters
        ----------
        starfield : Starfield
            Instance carrying unit‑sphere star positions & magnitudes.
        """
        self.star_dirs = starfield.get_unit_vectors()   # (N, 3)
        self.star_mags = starfield.get_magnitudes()     # (N,)
        self.fov_rad = config.CAMERA_FOV_RAD
        self.img_size = config.IMAGE_SIZE
        self.focal = config.FOCAL_LENGTH_PX
        self.half = self.img_size / 2.0

    # ------------------------------------------------------------------ #
    def capture(self, rotation, add_noise=True):
        """
        Render a 224×224 grayscale star image for the given orientation.

        Parameters
        ----------
        rotation : scipy.spatial.transform.Rotation
            Current spacecraft orientation (body→inertial).
        add_noise : bool
            Whether to add realistic sensor noise.

        Returns
        -------
        image : np.ndarray, shape (224, 224), dtype float32, range [0, 1]
        visible_count : int
            Number of stars actually rendered.
        """
        # Rotate all star directions into camera frame
        # Camera looks along +X in body frame; sensor plane is Y‑Z
        rot_matrix = rotation.as_matrix()                       # (3, 3)
        cam_dirs = (rot_matrix.T @ self.star_dirs.T).T          # (N, 3)

        # Keep only stars in front of the camera (positive X)
        in_front = cam_dirs[:, 0] > 0
        cam_dirs = cam_dirs[in_front]
        mags = self.star_mags[in_front]

        # Angular distance from boresight (camera +X axis)
        cos_angle = cam_dirs[:, 0] / np.linalg.norm(cam_dirs, axis=1)
        within_fov = cos_angle >= np.cos(self.fov_rad / 2.0)
        cam_dirs = cam_dirs[within_fov]
        mags = mags[within_fov]

        # Pinhole projection → pixel coordinates
        px = self.focal * (cam_dirs[:, 1] / cam_dirs[:, 0]) + self.half
        py = self.focal * (cam_dirs[:, 2] / cam_dirs[:, 0]) + self.half

        # Create blank image
        image = np.zeros((self.img_size, self.img_size), dtype=np.float64)

        # Render each star as a Gaussian PSF
        sigma = config.STAR_PSF_SIGMA
        max_mag = self.star_mags.max()

        for i in range(len(px)):
            cx, cy = px[i], py[i]
            # Brightness: inverse log of magnitude (brighter = lower mag)
            brightness = 10.0 ** ((max_mag - mags[i]) / 2.5)
            brightness = np.clip(brightness, 1.0, 255.0)

            # Stamp a small Gaussian (7×7 window)
            r = int(3 * sigma) + 1
            x0 = max(int(cx) - r, 0)
            x1 = min(int(cx) + r + 1, self.img_size)
            y0 = max(int(cy) - r, 0)
            y1 = min(int(cy) + r + 1, self.img_size)

            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]
            gauss = brightness * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)
            )
            image[y0:y1, x0:x1] += gauss

        # Add sensor noise
        if add_noise:
            read_noise = np.random.normal(0, config.NOISE_READ_STD,
                                          image.shape)
            dark_noise = np.random.poisson(config.NOISE_DARK_CURRENT,
                                           image.shape).astype(np.float64)
            image = image + read_noise + dark_noise

        # Clamp and normalise to [0, 1]
        image = np.clip(image, 0, None)
        if image.max() > 0:
            image = image / image.max()

        return image.astype(np.float32), len(px)

    # ------------------------------------------------------------------ #
    def capture_pil(self, rotation, add_noise=True):
        """Return the star image as a PIL Image (uint8, grayscale)."""
        img, n = self.capture(rotation, add_noise=add_noise)
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        return pil, n


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    from starfield import Starfield

    sf = Starfield()
    cam = VirtualCamera(sf)

    # Render looking straight ahead (identity quaternion)
    rot = R.identity()
    pil_img, count = cam.capture_pil(rot, add_noise=True)
    print(f"Stars in view: {count}")
    pil_img.save(os.path.join(config.DATA_DIR, "test_capture.png"))
    pil_img.show()
