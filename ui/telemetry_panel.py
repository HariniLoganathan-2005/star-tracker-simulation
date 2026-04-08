"""
Telemetry Panel — Real-Time Orientation & Error Display

Shows commanded orientation, AI-detected orientation, error vector,
severity indicator, star camera preview, and AI accuracy metrics.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QGridLayout, QFrame, QProgressBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QFont


class TelemetryPanel(QGroupBox):
    """Real-time telemetry readouts for the Mission Control dashboard."""

    def __init__(self, parent=None):
        super().__init__("📡  TELEMETRY", parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Status Banner ──
        self.status_label = QLabel("NOMINAL")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: #00331a; color: #00ff88; "
            "font-size: 16px; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(self.status_label)

        # ── Orientation Grid ──
        orient_box = QGroupBox("Orientation Data")
        grid = QGridLayout(orient_box)
        grid.setSpacing(4)

        headers = ["", "Pitch", "Roll", "Yaw"]
        for i, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet("color: #5a7a9a; font-weight: bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, i)

        # Target row
        grid.addWidget(self._label("Target:", "#888"), 1, 0)
        self.target_p = self._value_label()
        self.target_r = self._value_label()
        self.target_y = self._value_label()
        grid.addWidget(self.target_p, 1, 1)
        grid.addWidget(self.target_r, 1, 2)
        grid.addWidget(self.target_y, 1, 3)

        # True (actual) orientation
        grid.addWidget(self._label("True:", "#00aaff"), 2, 0)
        self.true_p = self._value_label("#00aaff")
        self.true_r = self._value_label("#00aaff")
        self.true_y = self._value_label("#00aaff")
        grid.addWidget(self.true_p, 2, 1)
        grid.addWidget(self.true_r, 2, 2)
        grid.addWidget(self.true_y, 2, 3)

        # AI estimated orientation
        grid.addWidget(self._label("AI Det:", "#ffaa00"), 3, 0)
        self.ai_p = self._value_label("#ffaa00")
        self.ai_r = self._value_label("#ffaa00")
        self.ai_y = self._value_label("#ffaa00")
        grid.addWidget(self.ai_p, 3, 1)
        grid.addWidget(self.ai_r, 3, 2)
        grid.addWidget(self.ai_y, 3, 3)

        # Error row
        grid.addWidget(self._label("Error:", "#ff4444"), 4, 0)
        self.err_p = self._value_label("#ff4444")
        self.err_r = self._value_label("#ff4444")
        self.err_y = self._value_label("#ff4444")
        grid.addWidget(self.err_p, 4, 1)
        grid.addWidget(self.err_r, 4, 2)
        grid.addWidget(self.err_y, 4, 3)

        layout.addWidget(orient_box)

        # ── Total Error + AI Accuracy ──
        metrics_row = QHBoxLayout()

        self.total_error_label = QLabel("Total Error: 0.00°")
        self.total_error_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #00ff88; "
            "padding: 4px 8px; background-color: #0d1117; "
            "border: 1px solid #1e3a5f; border-radius: 3px;"
        )
        metrics_row.addWidget(self.total_error_label)

        self.ai_accuracy_label = QLabel("AI Accuracy: —")
        self.ai_accuracy_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffaa00; "
            "padding: 4px 8px; background-color: #0d1117; "
            "border: 1px solid #1e3a5f; border-radius: 3px;"
        )
        metrics_row.addWidget(self.ai_accuracy_label)

        layout.addLayout(metrics_row)

        # ── Star Camera Preview ──
        cam_box = QGroupBox("🔭 Star Camera View")
        cam_layout = QVBoxLayout(cam_box)
        self.camera_display = QLabel()
        self.camera_display.setFixedSize(200, 200)
        self.camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_display.setStyleSheet(
            "background-color: #000; border: 2px solid #1e3a5f; "
            "border-radius: 4px;"
        )
        cam_layout.addWidget(self.camera_display,
                             alignment=Qt.AlignmentFlag.AlignCenter)

        self.star_count_label = QLabel("Stars in view: —")
        self.star_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.star_count_label.setStyleSheet("color: #5a7a9a;")
        cam_layout.addWidget(self.star_count_label)

        layout.addWidget(cam_box)

        # ── Recovery Progress ──
        self.recovery_bar = QProgressBar()
        self.recovery_bar.setVisible(False)
        self.recovery_bar.setTextVisible(True)
        self.recovery_bar.setFormat("Recovery: %p%")
        layout.addWidget(self.recovery_bar)

    # ── Helper factories ─────────────────────────────────────── #
    def _label(self, text, color="#888"):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        return lbl

    def _value_label(self, color="#00ff88"):
        lbl = QLabel("  0.00°")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 13px; "
            f"background-color: #0d1117; border: 1px solid #1e3a5f; "
            f"border-radius: 2px; padding: 2px 4px;"
        )
        return lbl

    # ── Update methods ────────────────────────────────────────── #
    def update_target(self, pitch, roll, yaw):
        self.target_p.setText(f"{pitch:+7.2f}°")
        self.target_r.setText(f"{roll:+7.2f}°")
        self.target_y.setText(f"{yaw:+7.2f}°")

    def update_true_orientation(self, pitch, roll, yaw):
        self.true_p.setText(f"{pitch:+7.2f}°")
        self.true_r.setText(f"{roll:+7.2f}°")
        self.true_y.setText(f"{yaw:+7.2f}°")

    def update_ai_orientation(self, pitch, roll, yaw):
        self.ai_p.setText(f"{pitch:+7.2f}°")
        self.ai_r.setText(f"{roll:+7.2f}°")
        self.ai_y.setText(f"{yaw:+7.2f}°")

    def update_error(self, error_data):
        """Update all error-related fields from ErrorCalculator output."""
        e = error_data["error_euler_deg"]
        self.err_p.setText(f"{e[0]:+7.2f}°")
        self.err_r.setText(f"{e[1]:+7.2f}°")
        self.err_y.setText(f"{e[2]:+7.2f}°")

        total = error_data["error_angle_deg"]
        self.total_error_label.setText(f"Total Error: {total:.2f}°")

        severity = error_data["severity"]
        color = error_data["color"]

        # Status banner
        if severity == "NOMINAL":
            bg = "#00331a"
        elif severity == "WARNING":
            bg = "#332800"
        else:
            bg = "#330000"

        self.status_label.setText(f"⬤  {severity}")
        self.status_label.setStyleSheet(
            f"background-color: {bg}; color: {color}; "
            f"font-size: 16px; font-weight: bold; "
            f"border-radius: 4px; padding: 6px;"
        )
        self.total_error_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color}; "
            f"padding: 4px 8px; background-color: #0d1117; "
            f"border: 1px solid #1e3a5f; border-radius: 3px;"
        )

    def update_ai_accuracy(self, true_euler, ai_euler):
        """Show the |AI - true| error."""
        if ai_euler is None:
            self.ai_accuracy_label.setText("AI Accuracy: —")
            return
        diff = np.abs(np.array(true_euler) - np.array(ai_euler))
        mean_err = np.mean(diff)
        self.ai_accuracy_label.setText(f"AI Error: {mean_err:.2f}°")

    def update_camera_image(self, star_image, star_count=0):
        """
        Display the star camera image.

        Parameters
        ----------
        star_image : np.ndarray (224, 224) float32 [0, 1]
        star_count : int
        """
        img_uint8 = (star_image * 255).astype(np.uint8)
        h, w = img_uint8.shape
        qt_img = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            200, 200, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_display.setPixmap(pixmap)
        self.star_count_label.setText(f"Stars in view: {star_count}")

    def show_recovery_progress(self, progress):
        """Show/update recovery progress bar."""
        if progress < 1.0:
            self.recovery_bar.setVisible(True)
            self.recovery_bar.setValue(int(progress * 100))
        else:
            self.recovery_bar.setVisible(False)
