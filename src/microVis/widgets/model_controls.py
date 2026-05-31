"""Left sidebar controls for the Model tab."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox
from microVis.processing.model_arch import BACKBONE_NAMES, SSL_METHODS


class ModelControls(QScrollArea):
    """Left sidebar controls for deep learning model training."""

    # Action signals
    prepare_clicked = Signal()
    train_clicked = Signal()
    stop_clicked = Signal()
    save_model_clicked = Signal()
    apply_model_clicked = Signal()
    save_results_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)

        container = QWidget()
        container.setStyleSheet("""
            QComboBox, QDoubleSpinBox, QSpinBox {
                min-height: 20px;
                max-height: 24px;
                font-size: 8pt;
                padding: 2px 3px;
                min-width: 0;
            }
            QLabel {
                font-size: 8pt;
            }
            QPushButton {
                font-size: 9pt;
                padding: 3px 8px;
            }
            QGroupBox {
                font-size: 8pt;
                font-weight: bold;
                color: #5a8a9a;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)

        # ── Training Mode ──
        self._build_mode_section()

        # ── Model Architecture ──
        self._build_arch_section()

        # ── Training Parameters ──
        self._build_params_section()

        # ── Augmentation ──
        self._build_augment_section()

        # ── SSL Options ──
        self._build_ssl_section()

        # ── Device ──
        self._build_device_section()

        # ── Actions ──
        self._build_actions_section()

        # ── Status ──
        self._build_status_section()

        self._layout.addStretch()

        self.setWidget(container)

        # Initial state
        self._on_mode_changed()

    # ── Section builders ──

    def _build_mode_section(self) -> None:
        grp = QGroupBox("Training Mode")
        layout = QVBoxLayout(grp)
        layout.setSpacing(3)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self._mode_combo = NoScrollComboBox()
        self._mode_combo.addItems(["Supervised", "Self-Supervised"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        row.addWidget(self._mode_combo, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source:"))
        self._source_combo = NoScrollComboBox()
        self._source_combo.addItems(["Image tab annotations", "All objects (no labels)"])
        row.addWidget(self._source_combo, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mask:"))
        self._mask_combo = NoScrollComboBox()
        row.addWidget(self._mask_combo, stretch=1)
        layout.addLayout(row)

        self._layout.addWidget(grp)

    def _build_arch_section(self) -> None:
        grp = QGroupBox("Model Architecture")
        layout = QVBoxLayout(grp)
        layout.setSpacing(3)

        row = QHBoxLayout()
        row.addWidget(QLabel("Backbone:"))
        self._backbone_combo = NoScrollComboBox()
        self._backbone_combo.addItems(BACKBONE_NAMES)
        row.addWidget(self._backbone_combo, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input size:"))
        self._input_size_combo = NoScrollComboBox()
        self._input_size_combo.addItems(["32", "64", "128", "224"])
        self._input_size_combo.setCurrentText("64")
        row.addWidget(self._input_size_combo, stretch=1)
        layout.addLayout(row)

        self._pretrained_check = QCheckBox("Pretrained (ImageNet)")
        self._pretrained_check.setChecked(True)
        layout.addWidget(self._pretrained_check)

        self._layout.addWidget(grp)

    def _build_params_section(self) -> None:
        grp = QGroupBox("Training Parameters")
        layout = QVBoxLayout(grp)
        layout.setSpacing(3)

        def _row(label_text, widget):
            r = QHBoxLayout()
            r.setSpacing(4)
            r.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(60)
            r.addWidget(lbl)
            r.addWidget(widget, stretch=1)
            layout.addLayout(r)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 999)
        self._epochs_spin.setValue(50)
        self._epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
        _row("Epochs:", self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 1024)
        self._batch_spin.setValue(32)
        self._batch_spin.setButtonSymbols(QSpinBox.NoButtons)
        _row("Batch size:", self._batch_spin)

        self._lr_spin = NoScrollDoubleSpinBox()
        self._lr_spin.setRange(1e-6, 1.0)
        self._lr_spin.setValue(1e-3)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setSingleStep(1e-4)
        self._lr_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        _row("LR:", self._lr_spin)

        self._optimizer_combo = NoScrollComboBox()
        self._optimizer_combo.addItems(["AdamW", "Adam", "SGD"])
        _row("Optimizer:", self._optimizer_combo)

        self._scheduler_combo = NoScrollComboBox()
        self._scheduler_combo.addItems(["Cosine", "StepLR", "ReduceOnPlateau", "None"])
        _row("Scheduler:", self._scheduler_combo)

        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 100)
        self._patience_spin.setValue(10)
        self._patience_spin.setButtonSymbols(QSpinBox.NoButtons)
        _row("Patience:", self._patience_spin)

        self._layout.addWidget(grp)

    def _build_augment_section(self) -> None:
        grp = QGroupBox("Augmentation")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)

        self._flip_h_check = QCheckBox("Horizontal flip")
        self._flip_h_check.setChecked(True)
        layout.addWidget(self._flip_h_check)

        self._flip_v_check = QCheckBox("Vertical flip")
        self._flip_v_check.setChecked(True)
        layout.addWidget(self._flip_v_check)

        self._rotate_check = QCheckBox("90° rotation")
        self._rotate_check.setChecked(True)
        layout.addWidget(self._rotate_check)

        self._noise_check = QCheckBox("Gaussian noise")
        self._noise_check.setChecked(True)
        layout.addWidget(self._noise_check)

        self._layout.addWidget(grp)

    def _build_ssl_section(self) -> None:
        self._ssl_group = QGroupBox("SSL Options")
        layout = QVBoxLayout(self._ssl_group)
        layout.setSpacing(3)

        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self._ssl_method_combo = NoScrollComboBox()
        self._ssl_method_combo.addItems(SSL_METHODS)
        row.addWidget(self._ssl_method_combo, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Temp:"))
        self._ssl_temp_spin = NoScrollDoubleSpinBox()
        self._ssl_temp_spin.setRange(0.01, 10.0)
        self._ssl_temp_spin.setValue(0.5)
        self._ssl_temp_spin.setDecimals(2)
        self._ssl_temp_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        row.addWidget(self._ssl_temp_spin, stretch=1)
        layout.addLayout(row)

        self._layout.addWidget(self._ssl_group)

    def _build_device_section(self) -> None:
        grp = QGroupBox("Device")
        layout = QVBoxLayout(grp)
        layout.setSpacing(3)

        self._gpu_label = QLabel("Detecting GPU...")
        self._gpu_label.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(self._gpu_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Device:"))
        self._device_combo = NoScrollComboBox()
        self._device_combo.addItem("cpu")
        row.addWidget(self._device_combo, stretch=1)
        layout.addLayout(row)

        self._layout.addWidget(grp)

    def _build_actions_section(self) -> None:
        grp = QGroupBox("Actions")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        self._prepare_btn = QPushButton("1. Prepare Data")
        self._prepare_btn.clicked.connect(self.prepare_clicked.emit)
        layout.addWidget(self._prepare_btn)

        self._train_btn = QPushButton("2. Train Model")
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self.train_clicked.emit)
        layout.addWidget(self._train_btn)

        self._stop_btn = QPushButton("Stop Training")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("QPushButton { color: #e06060; }")
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._stop_btn)

        self._save_btn = QPushButton("3. Save Model")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_model_clicked.emit)
        layout.addWidget(self._save_btn)

        self._apply_btn = QPushButton("4. Apply Model")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self.apply_model_clicked.emit)
        layout.addWidget(self._apply_btn)

        # Separator
        sep = QLabel("")
        sep.setStyleSheet("max-height: 1px; background: #3a3a4a;")
        layout.addWidget(sep)

        self._save_results_btn = QPushButton("Save Results to DB")
        self._save_results_btn.setEnabled(False)
        self._save_results_btn.clicked.connect(self.save_results_clicked.emit)
        layout.addWidget(self._save_results_btn)

        self._layout.addWidget(grp)

    def _build_status_section(self) -> None:
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-size: 8pt; padding: 4px;")
        self._status_label.setWordWrap(True)
        self._layout.addWidget(self._status_label)

    # ── Slots ──

    def _on_mode_changed(self) -> None:
        is_ssl = self._mode_combo.currentText() == "Self-Supervised"
        self._ssl_group.setVisible(is_ssl)
        # Hide source combo for SSL (all objects, no labels)
        if is_ssl:
            self._source_combo.setCurrentText("All objects (no labels)")
            self._source_combo.setEnabled(False)
        else:
            self._source_combo.setEnabled(True)

    # ── Public API ──

    def get_config(self) -> dict:
        """Return all training parameters as a dict."""
        return {
            "mode": "self_supervised" if self._mode_combo.currentText() == "Self-Supervised" else "supervised",
            "source": self._source_combo.currentText(),
            "mask_name": self._mask_combo.currentText(),
            "backbone": self._backbone_combo.currentText(),
            "crop_size": int(self._input_size_combo.currentText()),
            "pretrained": self._pretrained_check.isChecked(),
            "epochs": self._epochs_spin.value(),
            "batch_size": self._batch_spin.value(),
            "learning_rate": self._lr_spin.value(),
            "optimizer": self._optimizer_combo.currentText(),
            "scheduler": self._scheduler_combo.currentText(),
            "early_stopping_patience": self._patience_spin.value(),
            "augment_flip_h": self._flip_h_check.isChecked(),
            "augment_flip_v": self._flip_v_check.isChecked(),
            "augment_rotate": self._rotate_check.isChecked(),
            "augment_noise": self._noise_check.isChecked(),
            "ssl_method": self._ssl_method_combo.currentText(),
            "ssl_temperature": self._ssl_temp_spin.value(),
            "device": self._device_combo.currentText(),
        }

    def set_gpu_info(self, info: str) -> None:
        """Update GPU info label and populate device combo."""
        self._gpu_label.setText(info)
        self._device_combo.clear()
        self._device_combo.addItem("cpu")
        # Parse CUDA devices from info string
        if "CUDA" in info:
            for line in info.split("\n"):
                line = line.strip()
                if line.startswith("CUDA:") or "GeForce" in line or "RTX" in line or "GTX" in line:
                    pass
            # Add cuda:0, cuda:1, etc.
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self._device_combo.addItem(f"cuda:{i}")
                self._device_combo.setCurrentText("cuda:0")

    def set_masks(self, mask_names: list[str]) -> None:
        """Populate mask dropdown."""
        self._mask_combo.clear()
        self._mask_combo.addItems(mask_names)

    def set_state(self, state: str) -> None:
        """Enable/disable buttons based on wizard state."""
        states = {
            "idle": (True, False, False, False, False, False),
            "prepared": (True, True, False, False, False, False),
            "training": (False, False, True, False, False, False),
            "trained": (True, False, False, True, True, False),
            "applied": (True, False, False, True, True, True),
        }
        vals = states.get(state, states["idle"])
        self._prepare_btn.setEnabled(vals[0])
        self._train_btn.setEnabled(vals[1])
        self._stop_btn.setEnabled(vals[2])
        self._save_btn.setEnabled(vals[3])
        self._apply_btn.setEnabled(vals[4])
        self._save_results_btn.setEnabled(vals[5])

    def set_status(self, text: str) -> None:
        """Update status label."""
        self._status_label.setText(text)
