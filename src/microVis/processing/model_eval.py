"""Evaluation utilities for model training results."""

from __future__ import annotations

import numpy as np
from microVis.log_utils import get_logger

_log = get_logger("microVis.model_eval")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        class_names: List of class names.

    Returns:
        Dict with accuracy, macro_f1, per_class precision/recall/f1, confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    k: int = 5,
) -> dict:
    """Compute embedding quality metrics.

    Args:
        embeddings: (N, D) embedding vectors.
        labels: (N,) class labels (optional).
        k: Number of neighbors for kNN accuracy.

    Returns:
        Dict with knn_accuracy and silhouette_score (if labels provided).
    """
    result: dict = {}

    if labels is not None and len(np.unique(labels)) > 1:
        # kNN accuracy
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        knn = KNeighborsClassifier(n_neighbors=min(k, len(labels) - 1))
        scores = cross_val_score(knn, embeddings, labels, cv=min(5, len(labels)))
        result["knn_accuracy"] = float(scores.mean())

        # Silhouette score
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < len(embeddings):
            result["silhouette_score"] = float(
                silhouette_score(embeddings, labels)
            )

    return result


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensionality for visualization.

    Args:
        embeddings: (N, D) high-dimensional embeddings.
        method: 'umap' or 'tsne'.
        n_components: Target dimensionality (default 2).
        n_neighbors: UMAP n_neighbors parameter.
        random_state: Random seed.

    Returns:
        (N, n_components) reduced embeddings.
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            _log.warning("umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(embeddings) // 4))
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
        )
        return tsne.fit_transform(embeddings)

    raise ValueError(f"Unknown reduction method: {method}")


def format_metrics_table(metrics: dict) -> list[dict]:
    """Format per-class metrics as a list of dicts for table display.

    Returns:
        List of dicts with keys: class, precision, recall, f1, support.
    """
    rows = []
    for cls_name, cls_metrics in metrics.get("per_class", {}).items():
        rows.append({
            "class": cls_name,
            "precision": f"{cls_metrics['precision']:.3f}",
            "recall": f"{cls_metrics['recall']:.3f}",
            "f1": f"{cls_metrics['f1']:.3f}",
            "support": str(cls_metrics["support"]),
        })
    return rows
