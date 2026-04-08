"""
Mission Control Dashboard — Main Window

Combines the PyVista 3D viewport, Control Panel, and Telemetry Panel
into a professional aerospace-themed dashboard. Runs the simulation loop
and manages the Earth-to-Mars trajectory physics.
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QApplication, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import pyvista as pv
from pyvistaqt import QtInteractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from node_a_physics.starfield import Starfield
from node_a_physics.spacecraft import Spacecraft
from node_a_physics.virtual_camera import VirtualCamera
from node_a_physics.trajectory import Trajectory
from node_b_perception.inference import AttitudeEstimator
from node_c_navigation.error_calculator import ErrorCalculator
from node_c_navigation.auto_recovery import AutoRecovery
from ui.control_panel import ControlPanel
from ui.telemetry_panel import TelemetryPanel


class MissionControlDashboard(QMainWindow):
    """The main Mission Control window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 DIGITAL TWIN — Spacecraft Navigation Simulator")
        self.setMinimumSize(1400, 800)

        # Load stylesheet
        qss_path = os.path.join(os.path.dirname(__file__), "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

        # ── Initialise Nodes ──
        print("[Dashboard] Initialising Node A — Physical Engine …")
        self.starfield = Starfield()
        self.spacecraft = Spacecraft()
        self.camera = VirtualCamera(self.starfield)
        self.trajectory = Trajectory()

        print("[Dashboard] Initialising Node B — AI Perception …")
        self.estimator = AttitudeEstimator()

        print("[Dashboard] Initialising Node C — Navigation …")
        self.error_calc = ErrorCalculator()
        self.auto_recovery = AutoRecovery()

        # ── State ──
        self.target_rotation = R.identity()
        self.current_pos = np.array([0,0,0])
        self.deviation_pitch = 0.0
        self.deviation_roll = 0.0
        self.deviation_yaw = 0.0

        # ── Build UI ──
        self._build_ui()

        # ── Simulation Timer (20 Hz) ──
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._simulation_tick)
        self.sim_timer.start(config.UI_UPDATE_INTERVAL_MS)

        print("[Dashboard] ✓ Ready — enjoy the simulation!")

    # ────────────────────────────────────────────────────────── #
    # UI Construction
    # ────────────────────────────────────────────────────────── #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── Left: 3D Viewport ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("🛰️  DIGITAL TWIN — Mars Trajectory Monitor")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        self.plotter = QtInteractor(left_panel)
        self.plotter.set_background("#050510")
        self.plotter.enable_anti_aliasing("ssaa")
        left_layout.addWidget(self.plotter, stretch=1)

        # ── Right: Controls + Telemetry ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 4, 0)

        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel)

        self.telemetry_panel = TelemetryPanel()
        right_layout.addWidget(self.telemetry_panel, stretch=1)

        # Connect signals
        self.control_panel.orientation_changed.connect(self._on_deviation_changed)
        self.control_panel.auto_recover_clicked.connect(self._on_auto_recover)
        self.control_panel.reset_clicked.connect(self._on_reset)
        self.control_panel.perturb_clicked.connect(self._on_perturb)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

    # ────────────────────────────────────────────────────────── #
    # 3D Scene
    # ────────────────────────────────────────────────────────── #
    def _update_3d_scene(self):
        """Redraw the 3D scene."""
        if not hasattr(self, "_scene_inited"):
            self._scene_inited = True
            
            # 1. Starfield background
            star_mesh = self.starfield.get_pyvista_mesh()
            self.plotter.add_mesh(
                star_mesh, scalars="brightness", point_size=2.5,
                render_points_as_spheres=True, cmap="hot", show_scalar_bar=False,
                name="stars"
            )

            # 2. Earth, Moon, Mars spheres
            earth = pv.Sphere(radius=4.0, center=self.trajectory.earth_pos)
            moon = pv.Sphere(radius=1.5, center=self.trajectory.moon_pos)
            mars = pv.Sphere(radius=3.0, center=self.trajectory.mars_pos)
            self.plotter.add_mesh(earth, color="#2b82c9", smooth_shading=True, name="earth")
            self.plotter.add_mesh(moon, color="#9ea0a3", smooth_shading=True, name="moon")
            self.plotter.add_mesh(mars, color="#c1440e", smooth_shading=True, name="mars")

            # 3. Path Spline
            spline = pv.Spline(self.trajectory.path_points, len(self.trajectory.path_points))
            self.plotter.add_mesh(spline, color="#5a7a9a", line_width=2, style="wireframe", name="spline")

        # 4. Spacecraft Mesh (Translated & Rotated)
        ship_mesh = self.spacecraft.get_transformed_mesh()
        # Translate to current path position
        ship_mesh.translate(self.current_pos, inplace=True)
        self.plotter.add_mesh(
            ship_mesh, color="#b0c4de", specular=0.8, specular_power=30,
            ambient=0.3, diffuse=0.7, show_edges=False, name="spacecraft"
        )

        # Update Camera to look at ship, slightly offset to see the path
        ship_pos = self.current_pos
        self.plotter.camera_position = [
            # Move camera up and back relative to ship to provide a good tracking view
            ship_pos + np.array([-15, -25, 10]),  
            ship_pos,     # Focal point
            (0, 0, 1),    # Up vector
        ]

        self.plotter.render()

    # ────────────────────────────────────────────────────────── #
    # Pipeline
    # ────────────────────────────────────────────────────────── #
    def _run_full_pipeline(self):
        """Run the perception → navigation pipeline."""
        
        # 1. Calculate actual rotation = deviation * target
        deviation_rot = R.from_euler('xyz', [self.deviation_pitch, self.deviation_roll, self.deviation_yaw], degrees=True)
        actual_rotation = self.target_rotation * deviation_rot
        self.spacecraft.set_rotation(actual_rotation)

        # 2. Render star camera image
        star_img, star_count = self.camera.capture(actual_rotation, add_noise=True)

        # 3. AI inference
        ai_rot, ai_euler = self.estimator.estimate(star_img)

        # 4. Compute error (AI-estimated vs dynamic target)
        error_data = self.error_calc.compute(self.target_rotation, actual_rotation)

        # 5. Update telemetry display
        target_euler = self.target_rotation.as_euler('xyz', degrees=True)
        actual_euler = actual_rotation.as_euler('xyz', degrees=True)

        self.telemetry_panel.update_target(*target_euler)
        self.telemetry_panel.update_true_orientation(*actual_euler)
        self.telemetry_panel.update_ai_orientation(*ai_euler)
        self.telemetry_panel.update_error(error_data)
        self.telemetry_panel.update_ai_accuracy(actual_euler.tolist(), ai_euler.tolist())
        self.telemetry_panel.update_camera_image(star_img, star_count)

        # 6. Update 3D scene
        self._update_3d_scene()

    # ────────────────────────────────────────────────────────── #
    # Event Handlers
    # ────────────────────────────────────────────────────────── #
    def _on_deviation_changed(self, pitch, roll, yaw):
        """Called when user moves a slider."""
        if not self.auto_recovery.is_recovering:
            self.deviation_pitch = pitch
            self.deviation_roll = roll
            self.deviation_yaw = yaw
            self._run_full_pipeline()

    def _on_auto_recover(self):
        """Start auto-recovery SLERP animation."""
        current_rot = self.spacecraft.get_rotation()
        started = self.auto_recovery.start_recovery(current_rot, self.target_rotation)
        if started:
            self.control_panel.set_recovering(True)

    def _on_reset(self):
        """Reset to origin orientation."""
        self.auto_recovery.cancel()
        self.control_panel.set_recovering(False)
        self.deviation_pitch = 0.0
        self.deviation_roll = 0.0
        self.deviation_yaw = 0.0
        self.control_panel.set_sliders(0, 0, 0)
        self._run_full_pipeline()

    def _on_perturb(self):
        """Apply a random perturbation for expo drama."""
        p = np.random.uniform(-40, 40)
        r = np.random.uniform(-60, 60)
        y = np.random.uniform(-60, 60)
        self.control_panel.set_sliders(p, r, y)
        self.deviation_pitch = p
        self.deviation_roll = r
        self.deviation_yaw = y
        self._run_full_pipeline()

    # ────────────────────────────────────────────────────────── #
    # Simulation Timer Tick
    # ────────────────────────────────────────────────────────── #
    def _simulation_tick(self):
        """Called at 20 Hz."""

        if self.auto_recovery.is_recovering:
            # During emergency recovery, trajectory movement is paused for pure correction demo
            rot, progress = self.auto_recovery.get_next_frame()
            if rot is not None:
                self.spacecraft.set_rotation(rot)
                
                # Inverse calculate what the deviation sliders SHOULD show
                # current = target * dev  => dev = target^-1 * current
                dev_rot = self.target_rotation.inv() * rot
                dev_euler = dev_rot.as_euler('xyz', degrees=True)
                
                self.deviation_pitch = dev_euler[0]
                self.deviation_roll = dev_euler[1]
                self.deviation_yaw = dev_euler[2]
                self.control_panel.set_sliders(*dev_euler)

                self._run_full_pipeline()
                self.telemetry_panel.show_recovery_progress(progress)

            if not self.auto_recovery.is_recovering:
                self.control_panel.set_recovering(False)
                self.telemetry_panel.show_recovery_progress(1.0)
                
        else:
            # Normal condition: advance trajectory
            self.trajectory.advance()
            self.current_pos, self.target_rotation = self.trajectory.get_state()
            self._run_full_pipeline()


def launch_dashboard():
    """Create and show the Mission Control window."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setFont(QFont("Consolas", 10))

    window = MissionControlDashboard()
    window.showMaximized()

    return app, window
