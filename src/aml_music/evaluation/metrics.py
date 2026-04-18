"""Evaluation metrics for multi-label tagging and genre classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # PR-AUC via trapezoidal integration on recall axis.
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tag_names: list[str],
) -> dict[str, Any]:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have same shape.")

    per_tag = []
    aps = []
    rocs = []
    prs = []
    for idx, tag in enumerate(tag_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        ap = float(average_precision_score(yt, yp)) if len(np.unique(yt)) >= 2 else float("nan")
        roc = _safe_roc_auc(yt, yp)
        pr = _safe_pr_auc(yt, yp)
        support = int(yt.sum())
        per_tag.append(
            {
                "tag": tag,
                "support_pos": support,
                "roc_auc": roc,
                "pr_auc": pr,
                "ap": ap,
            }
        )
        aps.append(ap)
        rocs.append(roc)
        prs.append(pr)

    macro_roc = float(np.nanmean(np.array(rocs, dtype=float)))
    macro_pr = float(np.nanmean(np.array(prs, dtype=float)))
    map_score = float(np.nanmean(np.array(aps, dtype=float)))

    return {
        "macro_roc_auc": macro_roc,
        "macro_pr_auc": macro_pr,
        "map": map_score,
        "per_tag": per_tag,
    }


def rare_tag_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tag_names: list[str],
) -> dict[str, dict[str, float]]:
    supports = y_true.sum(axis=0)
    q1 = np.quantile(supports, 0.33)
    q2 = np.quantile(supports, 0.66)

    buckets = {
        "rare": np.where(supports <= q1)[0],
        "mid": np.where((supports > q1) & (supports <= q2))[0],
        "common": np.where(supports > q2)[0],
    }

    out: dict[str, dict[str, float]] = {}
    for name, idx in buckets.items():
        if len(idx) == 0:
            out[name] = {"macro_roc_auc": float("nan"), "macro_pr_auc": float("nan"), "map": float("nan")}
            continue
        metric = multilabel_metrics(y_true[:, idx], y_prob[:, idx], [tag_names[i] for i in idx])
        out[name] = {
            "macro_roc_auc": metric["macro_roc_auc"],
            "macro_pr_auc": metric["macro_pr_auc"],
            "map": metric["map"],
        }
    return out


def genre_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
