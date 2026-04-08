"""
Trajectory Simulator — Earth to Moon to Mars Path

Calculates the position and optimal "forward pointing" orientation
(target rotation) as the spacecraft travels along an imaginary spline.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv

class Trajectory:
    def __init__(self):
        # Scale: Arbitrary units for visualization
        self.earth_pos = np.array([-40.0, -10.0, -5.0])
        self.moon_pos = np.array([-10.0, 5.0, 2.0])
        self.mars_pos = np.array([50.0, 15.0, 10.0])

        # Generate a smooth spline path through the points
        points = np.vstack([self.earth_pos, self.moon_pos, self.mars_pos])
        
        # PyVista spline creates a series of points along the curve
        spline_poly = pv.Spline(points, 200)
        self.path_points = spline_poly.points
        self.num_points = len(self.path_points)
        
        self.t = 0.0  # Progress from 0.0 to 1.0
        self.speed = 0.0005  # Increment per tick (simulated speed)

    def advance(self):
        """Move the spacecraft forward along the line."""
        self.t += self.speed
        if self.t > 1.0:
            self.t = 0.0  # Loop back to Earth

    def get_state(self):
        """
        Returns the current 3D position and the tangent (forward array)
        which acts as the dynamic Target Orientation.
        """
        idx = int(self.t * (self.num_points - 1))
        pos = self.path_points[idx]
        
        # Calculate forward tangent vector
        next_idx = min(idx + 1, self.num_points - 1)
        if next_idx == idx:
            # If at the end, look slightly backward to get direction
            fwd = pos - self.path_points[idx - 1]
        else:
            fwd = self.path_points[next_idx] - pos
            
        fwd_norm = fwd / (np.linalg.norm(fwd) + 1e-8)
        
        # Generate a rotation describing this forward direction
        # Assume 'up' is +Z
        up = np.array([0, 0, 1])
        # If moving straight up, tweak it
        if abs(np.dot(fwd_norm, up)) > 0.99:
            up = np.array([0, 1, 0])
            
        right = np.cross(up, fwd_norm)
        right = right / np.linalg.norm(right)
        actual_up = np.cross(fwd_norm, right)
        
        # Rotation matrix aligning spacecraft body (+X) to forward tangent, (+Y) to right
        rot_mat = np.column_stack([fwd_norm, right, actual_up])
        
        target_rot = R.from_matrix(rot_mat)
        return pos, target_rot
