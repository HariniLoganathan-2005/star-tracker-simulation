"""
Auto-Recovery — SLERP-based Attitude Correction

Calculates the shortest rotation path from the current (deviated)
orientation back to the Mars trajectory using Spherical Linear
Interpolation.  The "Showstopper" feature for the expo.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class AutoRecovery:
    """SLERP‑based auto‑correction controller."""

    def __init__(self):
        self.is_recovering = False
        self._keyframes = None
        self._current_step = 0
        self._total_steps = config.SLERP_STEPS
        self.target = None

    def start_recovery(self, current_rotation, target_rotation):
        """
        Begin a SLERP recovery animation from current → target.

        Parameters
        ----------
        current_rotation : scipy.spatial.transform.Rotation
        target_rotation : scipy.spatial.transform.Rotation

        Returns
        -------
        bool : True if recovery was started (error was significant).
        """
        self.target = target_rotation
        # Check if error is negligible
        q_error = self.target * current_rotation.inv()
        angle = np.degrees(np.linalg.norm(q_error.as_rotvec()))

        if angle < 0.01:
            print("[AutoRecovery] Already on target — skipping.")
            return False

        # Build SLERP interpolator
        key_rots = R.concatenate([current_rotation, self.target])
        self._slerp = Slerp([0.0, 1.0], key_rots)

        # Pre-compute all keyframes with ease-in-out curve
        t_values = np.linspace(0, 1, self._total_steps)
        # Smooth ease-in-out (cubic)
        t_smooth = 3 * t_values**2 - 2 * t_values**3
        self._keyframes = [self._slerp(t) for t in t_smooth]

        self._current_step = 0
        self.is_recovering = True

        print(f"[AutoRecovery] ▶ Recovery started — "
              f"{angle:.2f}° error, {self._total_steps} steps")
        return True

    def get_next_frame(self):
        """
        Get the next interpolated orientation in the recovery.

        Returns
        -------
        rotation : Rotation or None (if recovery is complete)
        progress : float in [0, 1]
        """
        if not self.is_recovering:
            return None, 1.0

        if self._current_step >= self._total_steps:
            self.is_recovering = False
            print("[AutoRecovery] ✓ Recovery complete!")
            return self.target, 1.0

        rot = self._keyframes[self._current_step]
        progress = self._current_step / self._total_steps
        self._current_step += 1

        return rot, progress

    def cancel(self):
        """Cancel an in-progress recovery."""
        self.is_recovering = False
        self._current_step = 0
        print("[AutoRecovery] ✗ Recovery cancelled.")


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    recovery = AutoRecovery()

    # Start from a 30° pitch deviation
    start = R.from_euler('xyz', [30, -15, 10], degrees=True)
    recovery.start_recovery(start)

    while recovery.is_recovering:
        rot, progress = recovery.get_next_frame()
        if rot is not None:
            euler = rot.as_euler('xyz', degrees=True)
            bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            print(f"  [{bar}] {progress*100:5.1f}%  "
                  f"euler=({euler[0]:+6.2f}, {euler[1]:+6.2f}, {euler[2]:+6.2f})")
