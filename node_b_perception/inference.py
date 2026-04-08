"""
Real-Time Inference — Attitude Estimation from Star Images

Loads the trained ResNet-18 checkpoint and estimates spacecraft
orientation (pitch, roll, yaw) from a single star camera image.
"""

import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from node_b_perception.star_cnn import StarCNN


class AttitudeEstimator:
    """Wraps the trained CNN for single-image attitude inference."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH

        self.device = config.DEVICE
        self.model = StarCNN(pretrained=False).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.label_scales = np.array(
                checkpoint.get("label_scales", [90.0, 180.0, 180.0]),
                dtype=np.float32
            )
            print(f"[Estimator] Loaded model from {model_path}")
            print(f"[Estimator] Trained for {checkpoint.get('epoch', '?')} epochs, "
                  f"val_loss={checkpoint.get('val_loss', '?')}")
        else:
            # No trained model — use random weights (for UI testing)
            self.label_scales = np.array([90.0, 180.0, 180.0], dtype=np.float32)
            print(f"[Estimator] ⚠ No trained model found at {model_path}")
            print(f"[Estimator]   Using untrained model — run train.py first!")

        self.model.eval()

    def estimate(self, star_image):
        """
        Estimate attitude from a 224×224 star image.

        Parameters
        ----------
        star_image : np.ndarray, shape (224, 224), float32, range [0, 1]

        Returns
        -------
        rotation : scipy.spatial.transform.Rotation
        euler_deg : np.ndarray, shape (3,) — [pitch, roll, yaw] in degrees
        """
        # Prepare input tensor: (224,224) → (1, 1, 224, 224)
        img_tensor = torch.from_numpy(star_image).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.float().to(self.device)

        with torch.no_grad():
            pred_norm = self.model(img_tensor).cpu().numpy()[0]

        # De-normalise
        euler_deg = pred_norm * self.label_scales

        # Clamp to valid ranges
        euler_deg[0] = np.clip(euler_deg[0], -90, 90)
        euler_deg[1] = np.clip(euler_deg[1], -180, 180)
        euler_deg[2] = np.clip(euler_deg[2], -180, 180)

        rotation = R.from_euler('xyz', euler_deg, degrees=True)
        return rotation, euler_deg

    def estimate_from_pil(self, pil_image):
        """Convenience: accept a PIL Image (grayscale)."""
        img = np.array(pil_image).astype(np.float32) / 255.0
        return self.estimate(img)


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    from node_a_physics.starfield import Starfield
    from node_a_physics.virtual_camera import VirtualCamera

    sf = Starfield()
    cam = VirtualCamera(sf)
    estimator = AttitudeEstimator()

    # Test with a known orientation
    true_euler = [15.0, -30.0, 45.0]
    rot = R.from_euler('xyz', true_euler, degrees=True)
    img, n_stars = cam.capture(rot, add_noise=True)

    est_rot, est_euler = estimator.estimate(img)

    print(f"\nTrue  euler: {true_euler}")
    print(f"Est.  euler: {est_euler}")
    print(f"Error (deg): {np.abs(np.array(true_euler) - est_euler)}")
