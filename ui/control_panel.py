"""
Control Panel — Pitch / Roll / Yaw Sliders + Action Buttons

Provides the interactive "pilot" controls for the Mission Control dashboard.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QGroupBox, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QGroupBox):
    """Orientation sliders and action buttons."""

    # Signals emitted on user interaction
    orientation_changed = pyqtSignal(float, float, float)  # pitch, roll, yaw
    auto_recover_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    perturb_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("🎮  SPACECRAFT CONTROL", parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Sliders ──
        self.pitch_slider, pitch_row = self._make_slider(
            "Pitch", -90, 90, 0, "°"
        )
        self.roll_slider, roll_row = self._make_slider(
            "Roll ", -180, 180, 0, "°"
        )
        self.yaw_slider, yaw_row = self._make_slider(
            "Yaw  ", -180, 180, 0, "°"
        )

        layout.addLayout(pitch_row)
        layout.addLayout(roll_row)
        layout.addLayout(yaw_row)

        # Connect slider signals
        self.pitch_slider.valueChanged.connect(self._on_slider_changed)
        self.roll_slider.valueChanged.connect(self._on_slider_changed)
        self.yaw_slider.valueChanged.connect(self._on_slider_changed)

        # ── Buttons ──
        layout.addSpacing(10)

        self.recover_btn = QPushButton("🔄  AUTO-RECOVER")
        self.recover_btn.setObjectName("recover_btn")
        self.recover_btn.clicked.connect(self.auto_recover_clicked.emit)
        layout.addWidget(self.recover_btn)

        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("⟳  Reset to Origin")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self.reset_btn)

        self.perturb_btn = QPushButton("🎲  Random Perturbation")
        self.perturb_btn.clicked.connect(self.perturb_clicked.emit)
        btn_row.addWidget(self.perturb_btn)

        layout.addLayout(btn_row)

    def _make_slider(self, label_text, min_val, max_val, default, suffix=""):
        """Create a labeled slider with value display."""
        row = QHBoxLayout()

        label = QLabel(f"{label_text}:")
        label.setFixedWidth(50)
        label.setStyleSheet("font-weight: bold; color: #00d4ff;")
        row.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val * 10, max_val * 10)  # 0.1° precision
        slider.setValue(default * 10)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval((max_val - min_val) * 10 // 6)
        row.addWidget(slider)

        value_label = QLabel(f"{default:+6.1f}{suffix}")
        value_label.setFixedWidth(70)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        value_label.setStyleSheet("color: #00ff88; font-family: Consolas;")
        row.addWidget(value_label)

        # Update the value label when slider moves
        slider.valueChanged.connect(
            lambda v, lbl=value_label, s=suffix: lbl.setText(
                f"{v/10:+6.1f}{s}"
            )
        )

        return slider, row

    def _on_slider_changed(self, _):
        """Emit current slider values as (pitch, roll, yaw) in degrees."""
        p = self.pitch_slider.value() / 10.0
        r = self.roll_slider.value() / 10.0
        y = self.yaw_slider.value() / 10.0
        self.orientation_changed.emit(p, r, y)

    def _on_reset(self):
        """Reset all sliders to zero."""
        self.pitch_slider.setValue(0)
        self.roll_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.reset_clicked.emit()

    def set_sliders(self, pitch, roll, yaw):
        """Programmatically set slider positions (e.g. during auto-recovery)."""
        self.pitch_slider.blockSignals(True)
        self.roll_slider.blockSignals(True)
        self.yaw_slider.blockSignals(True)

        self.pitch_slider.setValue(int(pitch * 10))
        self.roll_slider.setValue(int(roll * 10))
        self.yaw_slider.setValue(int(yaw * 10))

        self.pitch_slider.blockSignals(False)
        self.roll_slider.blockSignals(False)
        self.yaw_slider.blockSignals(False)

    def set_recovering(self, is_recovering):
        """Disable sliders during auto-recovery."""
        self.pitch_slider.setEnabled(not is_recovering)
        self.roll_slider.setEnabled(not is_recovering)
        self.yaw_slider.setEnabled(not is_recovering)
        self.recover_btn.setEnabled(not is_recovering)
        self.perturb_btn.setEnabled(not is_recovering)
