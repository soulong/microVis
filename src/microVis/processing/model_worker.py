"""Background training and inference workers for the Model tab."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from microVis.log_utils import get_logger

_log = get_logger("microVis.model_worker")


# ── Configuration dataclasses ──


@dataclass
class TrainConfig:
    """Configuration for training."""

    mode: str = "supervised"  # "supervised" or "self_supervised"
    backbone: str = "ResNet-18"
    num_classes: int = 2
    in_channels: int = 1
    crop_size: int = 64
    pretrained: bool = True

    # Training params
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "AdamW"  # "AdamW", "Adam", "SGD"
    scheduler: str = "Cosine"  # "Cosine", "StepLR", "ReduceOnPlateau", "None"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # SSL params
    ssl_method: str = "SimCLR"  # "SimCLR", "Barlow Twins", "BYOL"
    ssl_temperature: float = 0.5

    # Device
    device: str = "cpu"


# ── Signal holders ──


class _TrainSignals(QObject):
    """Signals for the training worker."""

    epoch_done = Signal(int, dict)  # (epoch, metrics_dict)
    progress = Signal(int, int)  # (current_epoch, total_epochs)
    log_message = Signal(str)
    finished = Signal(dict)  # final metrics
    error = Signal(str)


class _InferenceSignals(QObject):
    """Signals for the inference worker."""

    progress = Signal(int, int)
    log_message = Signal(str)
    finished = Signal(object)  # DataFrame or ndarray
    error = Signal(str)


# ── PyTorch Dataset wrapper ──


class _CropTorchDataset:
    """Thin wrapper around CropRecord list for PyTorch DataLoader."""

    def __init__(
        self,
        records: list,
        dm: Any,
        mask_name: str,
        channel_names: list[str],
        crop_size: int,
        augment: bool = False,
        split: str | None = None,
    ):
        import torch
        from microVis.processing.model_dataset import extract_object_crops

        self._records = [r for r in records if split is None or r.split == split]
        self._dm = dm
        self._mask_name = mask_name
        self._channel_names = channel_names
        self._crop_size = crop_size
        self._augment = augment
        self._torch = torch
        self._extract = extract_object_crops

        # Cache loaded images to avoid re-reading
        self._img_cache: dict[int, tuple] = {}
        self._cache_max = 50

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple:
        rec = self._records[idx]
        torch = self._torch

        # Load image data (with simple cache)
        row_idx = rec.row_idx
        if row_idx not in self._img_cache:
            img_data, mask_dict = self._dm.get_imageset_with_masks(
                row_idx, channels=self._channel_names,
                masks=[f"mask_{self._mask_name}"],
            )
            self._img_cache[row_idx] = (img_data, mask_dict)
            # Evict old entries
            if len(self._img_cache) > self._cache_max:
                oldest = next(iter(self._img_cache))
                del self._img_cache[oldest]

        img_data, mask_dict = self._img_cache[row_idx]
        mask = mask_dict[f"mask_{self._mask_name}"]

        # Extract this specific object crop
        y_min, y_max, x_min, x_max = rec.bbox
        h, w = mask.shape
        pad = 4
        y_min_p = max(0, y_min - pad)
        y_max_p = min(h, y_max + pad + 1)
        x_min_p = max(0, x_min - pad)
        x_max_p = min(w, x_max + pad + 1)

        if img_data.ndim == 3:
            crop_img = img_data[y_min_p:y_max_p, x_min_p:x_max_p, :].astype(np.float64)
        else:
            crop_img = img_data[y_min_p:y_max_p, x_min_p:x_max_p].astype(np.float64)
            crop_img = crop_img[:, :, np.newaxis]

        crop_mask_local = mask[y_min_p:y_max_p, x_min_p:x_max_p]
        obj_mask = (crop_mask_local == rec.label_id).astype(np.float64)
        for ch in range(crop_img.shape[2]):
            crop_img[:, :, ch] *= obj_mask

        # Normalize per channel
        for ch in range(crop_img.shape[2]):
            ch_max = crop_img[:, :, ch].max()
            if ch_max > 0:
                crop_img[:, :, ch] /= ch_max

        # Resize
        from skimage.transform import resize as sk_resize
        ch_h, ch_w = crop_img.shape[:2]
        if ch_h != self._crop_size or ch_w != self._crop_size:
            crop_img = sk_resize(
                crop_img,
                (self._crop_size, self._crop_size, crop_img.shape[2]),
                preserve_range=True, anti_aliasing=True,
            ).astype(np.float64)

        # To tensor (C, H, W)
        tensor = torch.from_numpy(
            np.moveaxis(crop_img, -1, 0).astype(np.float32)
        )

        # Augmentation
        if self._augment:
            tensor = self._apply_augmentation(tensor)

        if rec.class_idx is not None:
            return tensor, rec.class_idx
        return tensor

    def _apply_augmentation(self, tensor):
        """Apply basic augmentations: flip, rotate, noise."""
        torch = self._torch
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, [-1])
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, [-2])
        # Random 90-degree rotation (0, 90, 180, 270)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            tensor = torch.rot90(tensor, k, [-2, -1])
        # Random Gaussian noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(tensor) * 0.02
            tensor = torch.clamp(tensor + noise, 0, 1)
        return tensor


# ── Training Worker ──


class TrainWorker(QRunnable):
    """Background worker for model training."""

    def __init__(
        self,
        config: TrainConfig,
        train_records: list,
        val_records: list,
        dm: Any,
        mask_name: str,
        channel_names: list[str],
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self.setAutoDelete(False)
        self._cfg = config
        self._train_records = train_records
        self._val_records = val_records
        self._dm = dm
        self._mask_name = mask_name
        self._channel_names = channel_names
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request early termination of training."""
        self._stop_requested = True

    def run(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader

            cfg = self._cfg
            device = torch.device(cfg.device)
            self.signals.log_message.emit(f"Using device: {device}")

            # Build datasets
            self.signals.log_message.emit("Building training dataset...")
            train_ds = _CropTorchDataset(
                self._train_records, self._dm, self._mask_name,
                self._channel_names, cfg.crop_size, augment=True,
            )
            val_ds = _CropTorchDataset(
                self._val_records, self._dm, self._mask_name,
                self._channel_names, cfg.crop_size, augment=False,
            )

            train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size,
                shuffle=True, num_workers=0, pin_memory=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size,
                shuffle=False, num_workers=0, pin_memory=False,
            )

            self.signals.log_message.emit(
                f"Train: {len(train_ds)} objects, Val: {len(val_ds)} objects"
            )

            if cfg.mode == "supervised":
                self._train_supervised(
                    train_loader, val_loader, device, cfg
                )
            else:
                self._train_ssl(train_loader, val_loader, device, cfg)

        except Exception as e:
            _log.exception("Training failed")
            self.signals.error.emit(str(e))

    def _train_supervised(self, train_loader, val_loader, device, cfg):
        """Supervised classification training loop."""
        import torch
        import torch.nn as nn
        from microVis.processing.model_arch import create_sl_model

        # Create model
        model = create_sl_model(
            cfg.backbone, cfg.num_classes, cfg.in_channels, cfg.pretrained,
        )
        model = model.to(device)

        # Optimizer
        if cfg.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.learning_rate,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.learning_rate,
                momentum=0.9, weight_decay=cfg.weight_decay,
            )

        # Scheduler
        scheduler = self._create_scheduler(optimizer, cfg, len(train_loader))
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accs = []
        val_f1s = []

        self.signals.log_message.emit(
            f"Starting training: {cfg.epochs} epochs, "
            f"backbone={cfg.backbone}, lr={cfg.learning_rate}"
        )

        for epoch in range(cfg.epochs):
            if self._stop_requested:
                self.signals.log_message.emit("Training stopped by user.")
                break

            # Train epoch
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if self._stop_requested:
                    break

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                running_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                break

            train_loss = running_loss / n_batches
            train_losses.append(train_loss)

            # Validate
            val_metrics = self._validate(model, val_loader, device, criterion)
            val_losses.append(val_metrics["loss"])
            val_accs.append(val_metrics["accuracy"])
            val_f1s.append(val_metrics["macro_f1"])

            # Check improvement
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
                # Save best model state
                self._best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1

            # Log
            current_lr = optimizer.param_groups[0]["lr"]
            self.signals.epoch_done.emit(epoch, {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": current_lr,
            })

            self.signals.log_message.emit(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"val_f1={val_metrics['macro_f1']:.3f} | "
                f"lr={current_lr:.2e}"
            )

            self.signals.progress.emit(epoch + 1, cfg.epochs)

            # Early stopping
            if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
                self.signals.log_message.emit(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {cfg.early_stopping_patience} epochs)"
                )
                break

        # Final results
        self.signals.log_message.emit(
            f"Training complete. Best val_acc={best_val_acc:.3f}, "
            f"best val_f1={best_val_f1:.3f}"
        )

        self.signals.finished.emit({
            "model_state": self._best_model_state,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "config": cfg,
        })

    def _train_ssl(self, train_loader, val_loader, device, cfg):
        """Self-supervised training loop."""
        import torch
        import torch.nn as nn
        from microVis.processing.model_arch import create_ssl_model

        backbone_model, projection_head = create_ssl_model(
            cfg.backbone, cfg.in_channels, cfg.pretrained,
            cfg.ssl_method, proj_dim=128,
        )
        backbone_model = backbone_model.to(device)
        projection_head = projection_head.to(device)

        # Combined parameters
        all_params = list(backbone_model.parameters()) + list(projection_head.parameters())

        if cfg.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                all_params, lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(all_params, lr=cfg.learning_rate)

        scheduler = self._create_scheduler(optimizer, cfg, len(train_loader))

        train_losses = []
        self.signals.log_message.emit(
            f"Starting SSL training: {cfg.epochs} epochs, "
            f"method={cfg.ssl_method}, backbone={cfg.backbone}"
        )

        for epoch in range(cfg.epochs):
            if self._stop_requested:
                self.signals.log_message.emit("Training stopped by user.")
                break

            backbone_model.train()
            projection_head.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if self._stop_requested:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                # Create two augmented views
                view1 = self._augment_batch(inputs)
                view2 = self._augment_batch(inputs)

                # Forward
                feat1 = backbone_model(view1)
                feat2 = backbone_model(view2)
                proj1 = projection_head(feat1)
                proj2 = projection_head(feat2)

                # Contrastive loss
                if cfg.ssl_method == "SimCLR":
                    loss = self._simclr_loss(proj1, proj2, cfg.ssl_temperature)
                elif cfg.ssl_method == "Barlow Twins":
                    loss = self._barlow_loss(proj1, proj2)
                elif cfg.ssl_method == "BYOL":
                    loss = self._byol_loss(proj1, proj2)
                else:
                    loss = self._simclr_loss(proj1, proj2, cfg.ssl_temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                running_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                break

            train_loss = running_loss / n_batches
            train_losses.append(train_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            self.signals.epoch_done.emit(epoch, {
                "train_loss": train_loss,
                "val_loss": train_loss,  # SSL doesn't have separate val loss
                "val_accuracy": 0.0,
                "val_macro_f1": 0.0,
                "lr": current_lr,
            })

            self.signals.log_message.emit(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"loss={train_loss:.4f} | lr={current_lr:.2e}"
            )

            self.signals.progress.emit(epoch + 1, cfg.epochs)

        # Save backbone state for embedding extraction
        self._best_model_state = {
            k: v.cpu().clone() for k, v in backbone_model.state_dict().items()
        }

        self.signals.log_message.emit("SSL training complete.")
        self.signals.finished.emit({
            "model_state": self._best_model_state,
            "train_losses": train_losses,
            "val_losses": train_losses,
            "val_accs": [],
            "val_f1s": [],
            "best_val_acc": 0.0,
            "best_val_f1": 0.0,
            "config": cfg,
        })

    def _validate(self, model, val_loader, device, criterion) -> dict:
        """Run validation and compute metrics."""
        import torch

        model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss += loss.item()
                n_batches += 1

        if n_batches == 0:
            return {"loss": 0, "accuracy": 0, "macro_f1": 0}

        from microVis.processing.model_eval import compute_classification_metrics
        metrics = compute_classification_metrics(
            np.array(all_labels), np.array(all_preds),
            [f"class_{i}" for i in range(max(max(all_labels) + 1, max(all_preds) + 1))],
        )

        return {
            "loss": running_loss / n_batches,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    def _create_scheduler(self, optimizer, cfg, steps_per_epoch):
        """Create learning rate scheduler."""
        import torch

        if cfg.scheduler == "Cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs * steps_per_epoch,
            )
        elif cfg.scheduler == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10 * steps_per_epoch, gamma=0.1,
            )
        elif cfg.scheduler == "ReduceOnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5,
            )
        return None

    def _augment_batch(self, batch):
        """Apply random augmentation to a batch of tensors."""
        import torch

        # Random flip
        if torch.rand(1).item() > 0.5:
            batch = torch.flip(batch, [-1])
        if torch.rand(1).item() > 0.5:
            batch = torch.flip(batch, [-2])
        # Random rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            batch = torch.rot90(batch, k, [-2, -1])
        return batch

    def _simclr_loss(self, z1, z2, temperature):
        """NT-Xent loss for SimCLR."""
        import torch
        import torch.nn.functional as F

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        return F.cross_entropy(sim, labels)

    def _barlow_loss(self, z1, z2, lambda_param=5e-3):
        """Barlow Twins loss."""
        import torch

        batch_size = z1.shape[0]
        dim = z1.shape[1]

        # Normalize
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)

        # Cross-correlation matrix
        c = torch.mm(z1.t(), z2) / batch_size

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        return on_diag + lambda_param * off_diag

    def _byol_loss(self, z1, z2):
        """Simple BYOL-like loss (negative cosine similarity)."""
        import torch.nn.functional as F

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return 2 - 2 * (z1 * z2).sum(dim=1).mean()

    @staticmethod
    def _off_diagonal(x):
        """Return off-diagonal elements of a square matrix."""
        import torch
        n = x.shape[0]
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ── Inference Worker ──


class InferenceWorker(QRunnable):
    """Background worker for applying a trained model to data."""

    def __init__(
        self,
        config: TrainConfig,
        model_state: dict,
        records: list,
        dm: Any,
        mask_name: str,
        channel_names: list[str],
        mode: str = "predict",  # "predict" or "embed"
    ):
        super().__init__()
        self.signals = _InferenceSignals()
        self.setAutoDelete(False)
        self._cfg = config
        self._model_state = model_state
        self._records = records
        self._dm = dm
        self._mask_name = mask_name
        self._channel_names = channel_names
        self._mode = mode

    def run(self) -> None:
        try:
            import torch
            from torch.utils.data import DataLoader

            cfg = self._cfg
            device = torch.device(cfg.device)

            # Build dataset
            ds = _CropTorchDataset(
                self._records, self._dm, self._mask_name,
                self._channel_names, cfg.crop_size, augment=False,
            )
            loader = DataLoader(
                ds, batch_size=cfg.batch_size,
                shuffle=False, num_workers=0,
            )

            if self._mode == "predict":
                self._run_predict(loader, device, cfg)
            else:
                self._run_embed(loader, device, cfg)

        except Exception as e:
            _log.exception("Inference failed")
            self.signals.error.emit(str(e))

    def _run_predict(self, loader, device, cfg):
        """Run classification prediction."""
        import torch
        import pandas as pd
        from microVis.processing.model_arch import create_sl_model

        model = create_sl_model(
            cfg.backbone, cfg.num_classes, cfg.in_channels, pretrained=False,
        )
        model.load_state_dict(self._model_state)
        model = model.to(device)
        model.eval()

        all_preds = []
        all_confs = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = probs.max(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())

                self.signals.progress.emit(i + 1, len(loader))

        # Build result DataFrame
        rows = []
        for rec, pred, conf in zip(self._records, all_preds, all_confs):
            rows.append({
                "well": rec.well,
                "field": rec.field,
                "stack": rec.stack,
                "tp": rec.tp,
                "label": rec.label_id,
                "predicted_class": int(pred),
                "confidence": float(conf),
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        self.signals.log_message.emit(f"Predictions complete: {len(df)} objects")
        self.signals.finished.emit(df)

    def _run_embed(self, loader, device, cfg):
        """Run embedding extraction."""
        import torch
        from microVis.processing.model_arch import create_embedding_model

        model = create_embedding_model(cfg.backbone, cfg.in_channels, pretrained=False)
        model.load_state_dict(self._model_state)
        model = model.to(device)
        model.eval()

        all_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)
                features = model(inputs)
                all_embeddings.append(features.cpu().numpy())

                self.signals.progress.emit(i + 1, len(loader))

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Build result DataFrame
        import pandas as pd
        feat_cols = [f"feat_{i}" for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, columns=feat_cols)
        df.insert(0, "well", [r.well for r in self._records])
        df.insert(1, "field", [r.field for r in self._records])
        df.insert(2, "stack", [r.stack for r in self._records])
        df.insert(3, "tp", [r.tp for r in self._records])
        df.insert(4, "label", [r.label_id for r in self._records])

        self.signals.log_message.emit(f"Embeddings complete: {len(df)} objects, {embeddings.shape[1]} features")
        self.signals.finished.emit(df)
