"""
Spacecraft — Procedural 3D Mesh + Quaternion State

Builds a good‑looking spacecraft from PyVista primitives and manages
its orientation using scipy Rotation (quaternion internally).
"""

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _build_spacecraft_mesh():
    """
    Construct a spacecraft from geometric primitives:
      • Cone (nose / body)
      • Cylinder (main fuselage)
      • Two flat boxes (solar panels)
      • Small cylinder (engine nozzle)
    All centred at origin, pointing along +X.
    """

    # ── Main body (cylinder along X) ──
    body = pv.Cylinder(
        center=(0, 0, 0), direction=(1, 0, 0),
        radius=0.5, height=3.0, resolution=40
    )

    # ── Nose cone ──
    nose = pv.Cone(
        center=(2.0, 0, 0), direction=(1, 0, 0),
        height=1.2, radius=0.5, resolution=40
    )

    # ── Engine nozzle ──
    nozzle = pv.Cylinder(
        center=(-1.8, 0, 0), direction=(1, 0, 0),
        radius=0.35, height=0.6, resolution=30
    )

    # ── Solar panel LEFT ──
    panel_l = pv.Box(bounds=(-0.8, 0.8, -3.5, -0.6, -0.03, 0.03))

    # ── Solar panel RIGHT ──
    panel_r = pv.Box(bounds=(-0.8, 0.8, 0.6, 3.5, -0.03, 0.03))

    # ── Panel struts ──
    strut_l = pv.Cylinder(
        center=(0, -0.55, 0), direction=(0, -1, 0),
        radius=0.04, height=0.1, resolution=12
    )
    strut_r = pv.Cylinder(
        center=(0, 0.55, 0), direction=(0, 1, 0),
        radius=0.04, height=0.1, resolution=12
    )

    # Merge all parts
    ship = body + nose + nozzle + panel_l + panel_r + strut_l + strut_r
    ship = ship.clean()

    return ship


class Spacecraft:
    """Manages the 3D mesh and quaternion orientation of the spacecraft."""

    def __init__(self):
        self.base_mesh = _build_spacecraft_mesh()
        self._rotation = R.identity()
        self.target_rotation = config.MARS_TARGET_ROTATION

    # ── Orientation setters ────────────────────────────────────────── #
    def set_euler(self, pitch, roll, yaw, degrees=True):
        """Set orientation from Euler angles (intrinsic XYZ)."""
        self._rotation = R.from_euler('xyz', [pitch, roll, yaw],
                                      degrees=degrees)

    def set_rotation(self, rotation: R):
        """Set orientation from a scipy Rotation object."""
        self._rotation = rotation

    # ── Orientation getters ────────────────────────────────────────── #
    def get_rotation(self) -> R:
        return self._rotation

    def get_quaternion(self):
        """Return (x, y, z, w) quaternion."""
        return self._rotation.as_quat()

    def get_euler_deg(self):
        """Return (pitch, roll, yaw) in degrees."""
        return self._rotation.as_euler('xyz', degrees=True)

    def get_rotation_matrix(self):
        return self._rotation.as_matrix()

    # ── Mesh for rendering ─────────────────────────────────────────── #
    def get_transformed_mesh(self):
        """Return a copy of the mesh rotated to the current orientation."""
        mesh = self.base_mesh.copy()
        mat4 = np.eye(4)
        mat4[:3, :3] = self.get_rotation_matrix()
        mesh.transform(mat4, inplace=True)
        return mesh

    # ── Target ─────────────────────────────────────────────────────── #
    def get_target_euler_deg(self):
        return self.target_rotation.as_euler('xyz', degrees=True)


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    sc = Spacecraft()
    sc.set_euler(20, 10, -15)
    print("Euler (deg):", sc.get_euler_deg())
    print("Quaternion :", sc.get_quaternion())

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(sc.get_transformed_mesh(), color="silver",
                     show_edges=True)
    plotter.add_axes()
    plotter.show()
