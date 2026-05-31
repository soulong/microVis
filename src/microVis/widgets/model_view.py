"""Main wizard area for the Model tab."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from microVis.log_utils import get_logger

_log = get_logger("microVis.model_view")


class ModelView(QWidget):
    """Main area for the Model tab with a 4-step wizard."""

    step_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Wizard navigation ──
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        self._step_labels = []
        step_names = ["1. Data Preview", "2. Training", "3. Results", "4. Apply"]
        for i, name in enumerate(step_names):
            lbl = QLabel(name)
            lbl.setStyleSheet(
                "padding: 4px 8px; font-size: 9pt; color: #888; "
                "border-bottom: 2px solid transparent;"
            )
            lbl.setAlignment(Qt.AlignCenter)
            self._step_labels.append(lbl)
            nav_row.addWidget(lbl)

        nav_row.addStretch()
        layout.addLayout(nav_row)

        # ── Stacked pages ──
        self._stack = QStackedWidget()

        self._page_preview = self._build_preview_page()
        self._page_train = self._build_train_page()
        self._page_results = self._build_results_page()
        self._page_apply = self._build_apply_page()

        self._stack.addWidget(self._page_preview)
        self._stack.addWidget(self._page_train)
        self._stack.addWidget(self._page_results)
        self._stack.addWidget(self._page_apply)

        layout.addWidget(self._stack, stretch=1)

        # ── Progress bar ──
        self._progress = QProgressBar()
        self._progress.setMaximumHeight(16)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m")
        layout.addWidget(self._progress)

        # Initial state
        self._set_step_style(0)

    # ── Page builders ──

    def _build_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        # Summary labels
        self._preview_summary = QLabel("No data prepared yet.")
        self._preview_summary.setStyleSheet("font-size: 10pt; padding: 8px;")
        self._preview_summary.setWordWrap(True)
        layout.addWidget(self._preview_summary)

        # Class distribution chart
        self._class_dist_canvas = None
        self._class_dist_container = QWidget()
        self._class_dist_layout = QVBoxLayout(self._class_dist_container)
        self._class_dist_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._class_dist_container)

        # Sample crops gallery
        gallery_label = QLabel("Sample Crops:")
        gallery_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        layout.addWidget(gallery_label)

        self._gallery_container = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_container)
        self._gallery_layout.setSpacing(4)
        self._gallery_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gallery_container)

        layout.addStretch()
        return page

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        # Charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(8)

        # Loss chart
        loss_container = QWidget()
        loss_layout = QVBoxLayout(loss_container)
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_label = QLabel("Loss")
        loss_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        loss_label.setAlignment(Qt.AlignCenter)
        loss_layout.addWidget(loss_label)
        self._loss_canvas = None
        self._loss_container = QWidget()
        self._loss_inner_layout = QVBoxLayout(self._loss_container)
        self._loss_inner_layout.setContentsMargins(0, 0, 0, 0)
        loss_layout.addWidget(self._loss_container)
        charts_row.addWidget(loss_container, stretch=1)

        # Accuracy chart
        acc_container = QWidget()
        acc_layout = QVBoxLayout(acc_container)
        acc_layout.setContentsMargins(0, 0, 0, 0)
        acc_label = QLabel("Accuracy / F1")
        acc_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        acc_label.setAlignment(Qt.AlignCenter)
        acc_layout.addWidget(acc_label)
        self._acc_canvas = None
        self._acc_container = QWidget()
        self._acc_inner_layout = QVBoxLayout(self._acc_container)
        self._acc_inner_layout.setContentsMargins(0, 0, 0, 0)
        acc_layout.addWidget(self._acc_container)
        charts_row.addWidget(acc_container, stretch=1)

        layout.addLayout(charts_row, stretch=1)

        # Log
        log_label = QLabel("Training Log")
        log_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        layout.addWidget(log_label)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(180)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, monospace; font-size: 8pt; "
            "background: #1a1a2e; color: #e0e0e0; border: 1px solid #3a3a4a; }"
        )
        layout.addWidget(self._log_text)

        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        # Summary
        self._results_summary = QLabel("No results yet.")
        self._results_summary.setStyleSheet("font-size: 10pt; padding: 8px;")
        self._results_summary.setWordWrap(True)
        layout.addWidget(self._results_summary)

        # Charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(8)

        # Confusion matrix / UMAP
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        self._viz_title = QLabel("Confusion Matrix")
        self._viz_title.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        self._viz_title.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self._viz_title)
        self._viz_canvas = None
        self._viz_container = QWidget()
        self._viz_inner_layout = QVBoxLayout(self._viz_container)
        self._viz_inner_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.addWidget(self._viz_container)
        charts_row.addWidget(viz_container, stretch=1)

        # Metrics table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_label = QLabel("Per-Class Metrics")
        table_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        table_layout.addWidget(table_label)
        self._metrics_table = QTableView()
        self._metrics_table.setMaximumHeight(300)
        self._metrics_table.setStyleSheet(
            "QTableView { font-size: 8pt; } "
            "QHeaderView::section { font-size: 8pt; background: #2a2a3a; }"
        )
        table_layout.addWidget(self._metrics_table)
        charts_row.addWidget(table_container, stretch=1)

        layout.addLayout(charts_row, stretch=1)

        # Sample predictions
        pred_label = QLabel("Sample Predictions")
        pred_label.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 9pt;")
        layout.addWidget(pred_label)

        self._pred_gallery = QWidget()
        self._pred_gallery_layout = QHBoxLayout(self._pred_gallery)
        self._pred_gallery_layout.setSpacing(4)
        self._pred_gallery_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pred_gallery)

        layout.addStretch()
        return page

    def _build_apply_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        self._apply_summary = QLabel("Apply the trained model to new data.")
        self._apply_summary.setStyleSheet("font-size: 10pt; padding: 8px;")
        self._apply_summary.setWordWrap(True)
        layout.addWidget(self._apply_summary)

        # Results preview table
        self._apply_table = QTableView()
        self._apply_table.setStyleSheet(
            "QTableView { font-size: 8pt; } "
            "QHeaderView::section { font-size: 8pt; background: #2a2a3a; }"
        )
        layout.addWidget(self._apply_table, stretch=1)

        layout.addStretch()
        return page

    # ── Step management ──

    def set_step(self, step: int) -> None:
        """Switch to a wizard step (0-3)."""
        self._stack.setCurrentIndex(step)
        self._set_step_style(step)
        self.step_changed.emit(step)

    def _set_step_style(self, active: int) -> None:
        for i, lbl in enumerate(self._step_labels):
            if i == active:
                lbl.setStyleSheet(
                    "padding: 4px 8px; font-size: 9pt; font-weight: bold; "
                    "color: #5a8a9a; border-bottom: 2px solid #5a8a9a;"
                )
            else:
                lbl.setStyleSheet(
                    "padding: 4px 8px; font-size: 9pt; color: #888; "
                    "border-bottom: 2px solid transparent;"
                )

    # ── Preview page ──

    def set_preview_data(
        self,
        summary: dict,
        sample_crops: list[np.ndarray] | None = None,
    ) -> None:
        """Populate the data preview page."""
        total = summary.get("total_objects", 0)
        n_classes = summary.get("n_classes", 0)
        train_n = summary.get("train_count", 0)
        val_n = summary.get("val_count", 0)
        test_n = summary.get("test_count", 0)
        crop_size = summary.get("crop_size", 64)
        n_ch = summary.get("n_channels", 0)
        has_labels = summary.get("has_labels", False)

        text = (
            f"<b>Dataset Summary</b><br>"
            f"Total objects: <b>{total}</b><br>"
            f"Channels: {n_ch} | Crop size: {crop_size}×{crop_size}<br>"
            f"Split: Train={train_n} | Val={val_n} | Test={test_n}<br>"
        )
        if has_labels:
            text += f"Classes: {n_classes}<br>"
            for cls_name, count in summary.get("class_counts", {}).items():
                text += f"  • {cls_name}: {count}<br>"
        else:
            text += "Mode: Self-supervised (no labels)<br>"

        self._preview_summary.setText(text)

        # Class distribution bar chart
        self._update_class_distribution_chart(summary.get("class_counts", {}))

        # Sample crops
        self._update_gallery(sample_crops or [])

    def _update_class_distribution_chart(self, class_counts: dict[str, int]) -> None:
        """Draw a bar chart of class distribution."""
        # Clear old canvas
        if self._class_dist_canvas is not None:
            self._class_dist_canvas.setParent(None)
            self._class_dist_canvas.deleteLater()
            self._class_dist_canvas = None

        if not class_counts:
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 2), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")

        names = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ["#5a8a9a", "#8ab5a0", "#d4a76a", "#c47a7a", "#9a7ab5",
                   "#7ab5d4", "#b5d47a", "#d47ab5"][:len(names)]

        bars = ax.bar(names, counts, color=colors, edgecolor="#3a3a4a")
        ax.set_ylabel("Count", color="#e0e0e0", fontsize=8)
        ax.tick_params(colors="#e0e0e0", labelsize=7)
        ax.spines["bottom"].set_color("#3a3a4a")
        ax.spines["left"].set_color("#3a3a4a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        self._class_dist_canvas = FigureCanvasQTAgg(fig)
        self._class_dist_canvas.setMaximumHeight(200)
        self._class_dist_layout.addWidget(self._class_dist_canvas)

    def _update_gallery(self, crops: list[np.ndarray]) -> None:
        """Display sample crop thumbnails."""
        # Clear old
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for crop in crops[:16]:
            # crop is (C, H, W) or (H, W, C)
            if crop.ndim == 3 and crop.shape[0] <= 4:
                # CHW → HWC
                img = np.moveaxis(crop, 0, -1)
            else:
                img = crop

            # Ensure uint8
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            ch = img.shape[2] if img.ndim == 3 else 1

            if ch == 1:
                qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            elif ch == 3:
                qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
            else:
                # Use first 3 channels
                img3 = img[:, :, :3].copy()
                qimg = QImage(img3.data, w, h, 3 * w, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setFixedSize(68, 68)
            lbl.setStyleSheet("border: 1px solid #3a3a4a; border-radius: 3px;")
            self._gallery_layout.addWidget(lbl)

        self._gallery_layout.addStretch()

    # ── Training page ──

    def init_training_charts(self) -> None:
        """Initialize empty training charts."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # Loss chart
        if self._loss_canvas is not None:
            self._loss_canvas.setParent(None)
            self._loss_canvas.deleteLater()

        fig_loss = Figure(figsize=(4, 3), dpi=100)
        fig_loss.patch.set_facecolor("#1a1a2e")
        self._loss_ax = fig_loss.add_subplot(111)
        self._loss_ax.set_facecolor("#1a1a2e")
        self._loss_ax.set_xlabel("Epoch", color="#e0e0e0", fontsize=8)
        self._loss_ax.set_ylabel("Loss", color="#e0e0e0", fontsize=8)
        self._loss_ax.tick_params(colors="#e0e0e0", labelsize=7)
        self._loss_ax.spines["bottom"].set_color("#3a3a4a")
        self._loss_ax.spines["left"].set_color("#3a3a4a")
        self._loss_ax.spines["top"].set_visible(False)
        self._loss_ax.spines["right"].set_visible(False)
        fig_loss.tight_layout()

        self._loss_canvas = FigureCanvasQTAgg(fig_loss)
        self._loss_inner_layout.addWidget(self._loss_canvas)

        # Accuracy chart
        if self._acc_canvas is not None:
            self._acc_canvas.setParent(None)
            self._acc_canvas.deleteLater()

        fig_acc = Figure(figsize=(4, 3), dpi=100)
        fig_acc.patch.set_facecolor("#1a1a2e")
        self._acc_ax = fig_acc.add_subplot(111)
        self._acc_ax.set_facecolor("#1a1a2e")
        self._acc_ax.set_xlabel("Epoch", color="#e0e0e0", fontsize=8)
        self._acc_ax.set_ylabel("Score", color="#e0e0e0", fontsize=8)
        self._acc_ax.tick_params(colors="#e0e0e0", labelsize=7)
        self._acc_ax.spines["bottom"].set_color("#3a3a4a")
        self._acc_ax.spines["left"].set_color("#3a3a4a")
        self._acc_ax.spines["top"].set_visible(False)
        self._acc_ax.spines["right"].set_visible(False)
        fig_acc.tight_layout()

        self._acc_canvas = FigureCanvasQTAgg(fig_acc)
        self._acc_inner_layout.addWidget(self._acc_canvas)

        # Data storage for incremental updates
        self._train_losses = []
        self._val_losses = []
        self._val_accs = []
        self._val_f1s = []

        self._log_text.clear()

    def update_epoch(self, epoch: int, metrics: dict) -> None:
        """Update charts with new epoch data."""
        self._train_losses.append(metrics.get("train_loss", 0))
        self._val_losses.append(metrics.get("val_loss", 0))
        self._val_accs.append(metrics.get("val_accuracy", 0))
        self._val_f1s.append(metrics.get("val_macro_f1", 0))

        epochs = list(range(1, len(self._train_losses) + 1))

        # Update loss chart
        if self._loss_canvas is not None:
            self._loss_ax.clear()
            self._loss_ax.plot(epochs, self._train_losses, color="#5a8a9a",
                              label="Train", linewidth=1.5)
            self._loss_ax.plot(epochs, self._val_losses, color="#c47a7a",
                              label="Val", linewidth=1.5)
            self._loss_ax.legend(fontsize=7, facecolor="#1a1a2e",
                                edgecolor="#3a3a4a", labelcolor="#e0e0e0")
            self._loss_ax.set_xlabel("Epoch", color="#e0e0e0", fontsize=8)
            self._loss_ax.set_ylabel("Loss", color="#e0e0e0", fontsize=8)
            self._loss_ax.tick_params(colors="#e0e0e0", labelsize=7)
            self._loss_ax.spines["bottom"].set_color("#3a3a4a")
            self._loss_ax.spines["left"].set_color("#3a3a4a")
            self._loss_ax.spines["top"].set_visible(False)
            self._loss_ax.spines["right"].set_visible(False)
            self._loss_canvas.draw()

        # Update accuracy chart
        if self._acc_canvas is not None and any(v > 0 for v in self._val_accs):
            self._acc_ax.clear()
            self._acc_ax.plot(epochs, self._val_accs, color="#8ab5a0",
                              label="Accuracy", linewidth=1.5)
            self._acc_ax.plot(epochs, self._val_f1s, color="#d4a76a",
                              label="Macro F1", linewidth=1.5)
            self._acc_ax.legend(fontsize=7, facecolor="#1a1a2e",
                               edgecolor="#3a3a4a", labelcolor="#e0e0e0")
            self._acc_ax.set_xlabel("Epoch", color="#e0e0e0", fontsize=8)
            self._acc_ax.set_ylabel("Score", color="#e0e0e0", fontsize=8)
            self._acc_ax.tick_params(colors="#e0e0e0", labelsize=7)
            self._acc_ax.spines["bottom"].set_color("#3a3a4a")
            self._acc_ax.spines["left"].set_color("#3a3a4a")
            self._acc_ax.spines["top"].set_visible(False)
            self._acc_ax.spines["right"].set_visible(False)
            self._acc_canvas.draw()

    def append_log(self, message: str) -> None:
        """Append a message to the training log."""
        self._log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current: int, total: int) -> None:
        """Update the progress bar."""
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    # ── Results page ──

    def set_results_sl(
        self,
        metrics: dict,
        class_names: list[str],
        sample_crops: list[np.ndarray] | None = None,
        sample_preds: list[str] | None = None,
    ) -> None:
        """Display supervised learning results."""
        self._viz_title.setText("Confusion Matrix")

        # Summary
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("macro_f1", 0)
        self._results_summary.setText(
            f"<b>Training Results</b><br>"
            f"Accuracy: <b>{acc:.3f}</b> | Macro F1: <b>{f1:.3f}</b>"
        )

        # Confusion matrix
        self._update_confusion_matrix(
            metrics.get("confusion_matrix", np.array([])),
            class_names,
        )

        # Metrics table
        self._update_metrics_table(metrics, class_names)

        # Sample predictions
        self._update_pred_gallery(sample_crops or [], sample_preds or [])

    def set_results_ssl(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray | None,
        class_names: list[str] | None,
        quality_metrics: dict,
    ) -> None:
        """Display SSL results with UMAP plot."""
        self._viz_title.setText("Embedding UMAP")

        knn_acc = quality_metrics.get("knn_accuracy", "N/A")
        sil = quality_metrics.get("silhouette_score", "N/A")
        text = "<b>SSL Results</b><br>"
        if isinstance(knn_acc, float):
            text += f"kNN Accuracy: <b>{knn_acc:.3f}</b><br>"
        if isinstance(sil, float):
            text += f"Silhouette Score: <b>{sil:.3f}</b><br>"
        self._results_summary.setText(text)

        # UMAP plot
        self._update_umap_plot(embeddings_2d, labels, class_names)

        # Clear table
        from PySide6.QtGui import QStandardItemModel
        self._metrics_table.setModel(QStandardItemModel())

    def _update_confusion_matrix(self, cm: np.ndarray, class_names: list[str]) -> None:
        """Draw confusion matrix heatmap."""
        if self._viz_canvas is not None:
            self._viz_canvas.setParent(None)
            self._viz_canvas.deleteLater()
            self._viz_canvas = None

        if cm.size == 0:
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 3.5), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7, color="#e0e0e0")
        ax.set_yticklabels(class_names, fontsize=7, color="#e0e0e0")
        ax.set_xlabel("Predicted", color="#e0e0e0", fontsize=8)
        ax.set_ylabel("True", color="#e0e0e0", fontsize=8)

        # Annotate cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "#e0e0e0"
                ax.text(j, i, str(val), ha="center", va="center",
                       color=color, fontsize=8)

        fig.tight_layout()
        self._viz_canvas = FigureCanvasQTAgg(fig)
        self._viz_inner_layout.addWidget(self._viz_canvas)

    def _update_umap_plot(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray | None,
        class_names: list[str] | None,
    ) -> None:
        """Draw UMAP/t-SNE scatter plot."""
        if self._viz_canvas is not None:
            self._viz_canvas.setParent(None)
            self._viz_canvas.deleteLater()
            self._viz_canvas = None

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 3.5), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")

        colors = ["#5a8a9a", "#8ab5a0", "#d4a76a", "#c47a7a", "#9a7ab5",
                   "#7ab5d4", "#b5d47a", "#d47ab5"]

        if labels is not None and class_names:
            unique_labels = np.unique(labels)
            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                name = class_names[i] if i < len(class_names) else f"Class {lbl}"
                ax.scatter(
                    embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[i % len(colors)], label=name,
                    s=10, alpha=0.7,
                )
            ax.legend(fontsize=7, facecolor="#1a1a2e",
                     edgecolor="#3a3a4a", labelcolor="#e0e0e0")
        else:
            ax.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c="#5a8a9a", s=10, alpha=0.7,
            )

        ax.set_xlabel("Dim 1", color="#e0e0e0", fontsize=8)
        ax.set_ylabel("Dim 2", color="#e0e0e0", fontsize=8)
        ax.tick_params(colors="#e0e0e0", labelsize=7)
        ax.spines["bottom"].set_color("#3a3a4a")
        ax.spines["left"].set_color("#3a3a4a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        self._viz_canvas = FigureCanvasQTAgg(fig)
        self._viz_inner_layout.addWidget(self._viz_canvas)

    def _update_metrics_table(self, metrics: dict, class_names: list[str]) -> None:
        """Update the per-class metrics table."""
        from PySide6.QtGui import QStandardItem, QStandardItemModel

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1", "Support"])

        per_class = metrics.get("per_class", {})
        for i, name in enumerate(class_names):
            m = per_class.get(name, {})
            row = [
                QStandardItem(name),
                QStandardItem(f"{m.get('precision', 0):.3f}"),
                QStandardItem(f"{m.get('recall', 0):.3f}"),
                QStandardItem(f"{m.get('f1', 0):.3f}"),
                QStandardItem(str(m.get("support", 0))),
            ]
            model.appendRow(row)

        self._metrics_table.setModel(model)
        header = self._metrics_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

    def _update_pred_gallery(
        self,
        crops: list[np.ndarray],
        labels: list[str],
    ) -> None:
        """Display sample prediction thumbnails."""
        while self._pred_gallery_layout.count():
            item = self._pred_gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for crop, label_text in zip(crops[:12], labels[:12]):
            if crop.ndim == 3 and crop.shape[0] <= 4:
                img = np.moveaxis(crop, 0, -1)
            else:
                img = crop

            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            ch = img.shape[2] if img.ndim == 3 else 1

            if ch == 1:
                qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            elif ch == 3:
                qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
            else:
                img3 = img[:, :, :3].copy()
                qimg = QImage(img3.data, w, h, 3 * w, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg).scaled(
                48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )

            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(2, 2, 2, 2)
            v.setSpacing(2)

            img_lbl = QLabel()
            img_lbl.setPixmap(pixmap)
            img_lbl.setFixedSize(52, 52)
            img_lbl.setStyleSheet("border: 1px solid #3a3a4a; border-radius: 3px;")
            v.addWidget(img_lbl)

            txt_lbl = QLabel(label_text)
            txt_lbl.setStyleSheet("font-size: 7pt; color: #888;")
            txt_lbl.setAlignment(Qt.AlignCenter)
            v.addWidget(txt_lbl)

            self._pred_gallery_layout.addWidget(container)

        self._pred_gallery_layout.addStretch()

    # ── Apply page ──

    def set_apply_results(self, df) -> None:
        """Display apply results in the table."""
        from PySide6.QtGui import QStandardItem, QStandardItemModel

        model = QStandardItemModel()
        if df is None or df.empty:
            self._apply_table.setModel(model)
            return

        # Set headers
        headers = [str(c) for c in df.columns]
        model.setHorizontalHeaderLabels(headers)

        # Populate rows (limit to 1000 for performance)
        for _, row in df.head(1000).iterrows():
            items = [QStandardItem(str(v)) for v in row]
            model.appendRow(items)

        self._apply_table.setModel(model)
        header = self._apply_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        self._apply_summary.setText(
            f"Results: {len(df)} objects predicted. "
            f"{'Showing first 1000 rows.' if len(df) > 1000 else ''}"
        )

    # ── Clear ──

    def clear(self) -> None:
        """Reset all pages."""
        self._preview_summary.setText("No data prepared yet.")
        self._results_summary.setText("No results yet.")
        self._apply_summary.setText("Apply the trained model to new data.")
        self._log_text.clear()
        self._progress.reset()

        # Clear charts
        for canvas_attr in ["_class_dist_canvas", "_loss_canvas", "_acc_canvas", "_viz_canvas"]:
            canvas = getattr(self, canvas_attr, None)
            if canvas is not None:
                canvas.setParent(None)
                canvas.deleteLater()
                setattr(self, canvas_attr, None)

        # Clear gallery
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        while self._pred_gallery_layout.count():
            item = self._pred_gallery_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Clear tables
        from PySide6.QtGui import QStandardItemModel
        self._metrics_table.setModel(QStandardItemModel())
        self._apply_table.setModel(QStandardItemModel())

        self.set_step(0)
