"""
Error Quaternion Calculator — The Navigation Brain

Computes the orientation error between the AI-estimated attitude
and the commanded Mars trajectory, classifies severity, and
provides human-readable telemetry.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ErrorCalculator:
    """Computes and classifies orientation error."""

    # Severity levels
    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __init__(self):
        pass

    def compute(self, target_rotation, actual_rotation):
        """
        Compute the error quaternion between actual and target.

        Parameters
        ----------
        target_rotation : scipy.spatial.transform.Rotation
            The dynamically intended orientation of the spacecraft.
        actual_rotation : scipy.spatial.transform.Rotation
            The AI-estimated (or true) spacecraft orientation.

        Returns
        -------
        dict with keys:
            error_rotation  : Rotation  — the error quaternion object
            error_euler_deg : ndarray   — (pitch, roll, yaw) error in degrees
            error_angle_deg : float     — total angular error magnitude
            error_axis      : ndarray   — (3,) rotation axis of the error
            severity        : str       — NOMINAL / WARNING / CRITICAL
            color           : str       — hex color for UI display
        """
        # q_error = q_target ⊗ q_actual⁻¹
        q_error = target_rotation * actual_rotation.inv()

        # Rotation vector → magnitude is the total angle
        rotvec = q_error.as_rotvec()
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)

        # Axis (handle zero-error case)
        if angle_rad > 1e-10:
            axis = rotvec / angle_rad
        else:
            axis = np.array([1.0, 0.0, 0.0])

        # Euler decomposition of the error
        error_euler = q_error.as_euler('xyz', degrees=True)

        # Severity classification
        if angle_deg < config.THRESHOLD_NOMINAL:
            severity = self.NOMINAL
            color = "#00FF88"    # Green
        elif angle_deg < config.THRESHOLD_WARNING:
            severity = self.WARNING
            color = "#FFD700"    # Gold
        else:
            severity = self.CRITICAL
            color = "#FF3333"    # Red

        return {
            "error_rotation": q_error,
            "error_euler_deg": error_euler,
            "error_angle_deg": angle_deg,
            "error_axis": axis,
            "severity": severity,
            "color": color,
        }

    def format_telemetry(self, error_data):
        """Return a formatted multi-line telemetry string."""
        e = error_data
        lines = [
            f"┌─── ERROR TELEMETRY ───┐",
            f"│ Status: {e['severity']:>13} │",
            f"│ Total Error: {e['error_angle_deg']:>7.2f}° │",
            f"│ Pitch Err: {e['error_euler_deg'][0]:>+8.2f}° │",
            f"│ Roll  Err: {e['error_euler_deg'][1]:>+8.2f}° │",
            f"│ Yaw   Err: {e['error_euler_deg'][2]:>+8.2f}° │",
            f"└───────────────────────┘",
        ]
        return "\n".join(lines)


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    calc = ErrorCalculator()

    # Simulate a 20° pitch deviation
    actual = R.from_euler('xyz', [20, 0, 0], degrees=True)
    result = calc.compute(actual)
    print(calc.format_telemetry(result))
    print(f"\nSeverity: {result['severity']}")
    print(f"Color:    {result['color']}")
