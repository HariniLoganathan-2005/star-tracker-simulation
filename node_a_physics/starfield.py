"""
Starfield Generator — Hipparcos Catalog on a 3D Sphere

Downloads the Hipparcos star catalog via Skyfield, converts RA/Dec to
3D Cartesian coordinates, and produces a PyVista point cloud for rendering.
"""

import os
import numpy as np
import pyvista as pv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _ra_dec_to_cartesian(ra_rad, dec_rad, radius=1.0):
    """Convert Right Ascension / Declination (radians) to 3D Cartesian."""
    x = radius * np.cos(dec_rad) * np.cos(ra_rad)
    y = radius * np.cos(dec_rad) * np.sin(ra_rad)
    z = radius * np.sin(dec_rad)
    return np.column_stack([x, y, z])


class Starfield:
    """Loads Hipparcos catalog and exposes star data + PyVista mesh."""

    def __init__(self):
        self.star_positions = None   # (N, 3) unit‑sphere coords
        self.star_magnitudes = None  # (N,) apparent magnitudes
        self._mesh = None
        self._load_catalog()

    # ------------------------------------------------------------------ #
    def _load_catalog(self):
        """Download (first run) or load cached Hipparcos catalog."""
        from skyfield.api import Loader
        from skyfield.data import hipparcos

        # Skyfield's Loader will download once into our cache dir, then reuse
        loader = Loader(config.HIPPARCOS_CACHE_DIR)
        with loader.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)

        # Drop rows with missing RA/Dec/Magnitude
        df = df.dropna(subset=["ra_degrees", "dec_degrees", "magnitude"])

        # Filter to visible stars
        df = df[df["magnitude"] <= config.STAR_MAGNITUDE_LIMIT]
        df = df.reset_index(drop=True)

        ra_rad = np.radians(df["ra_degrees"].values)
        dec_rad = np.radians(df["dec_degrees"].values)

        self.star_positions = _ra_dec_to_cartesian(ra_rad, dec_rad, radius=1.0)
        self.star_magnitudes = df["magnitude"].values.astype(np.float32)

        print(f"[Starfield] Loaded {len(self.star_magnitudes)} stars "
              f"(mag ≤ {config.STAR_MAGNITUDE_LIMIT})")

    # ------------------------------------------------------------------ #
    def get_pyvista_mesh(self, radius=None):
        """
        Returns a PyVista PolyData point cloud scaled to `radius`.
        Brightness is encoded as point size via the 'magnitude' scalar.
        """
        if radius is None:
            radius = config.STAR_SPHERE_RADIUS

        if self._mesh is None or radius != config.STAR_SPHERE_RADIUS:
            points = self.star_positions * radius
            self._mesh = pv.PolyData(points)

            # Invert magnitude so brighter stars have higher values
            max_mag = self.star_magnitudes.max()
            brightness = max_mag - self.star_magnitudes + 0.5
            self._mesh["brightness"] = brightness

        return self._mesh

    # ------------------------------------------------------------------ #
    def get_unit_vectors(self):
        """Return (N, 3) unit‑sphere star direction vectors."""
        return self.star_positions.copy()

    def get_magnitudes(self):
        """Return (N,) apparent magnitudes."""
        return self.star_magnitudes.copy()


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    sf = Starfield()
    mesh = sf.get_pyvista_mesh()
    print(f"Mesh points: {mesh.n_points}")

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(
        mesh,
        scalars="brightness",
        point_size=3,
        render_points_as_spheres=True,
        cmap="hot",
        show_scalar_bar=False,
    )
    plotter.show()
