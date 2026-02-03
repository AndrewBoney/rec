from __future__ import annotations

from typing import Dict

import torch


def mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - labels)).item()


def rmse(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - labels) ** 2)).item()


def pearson_correlation(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if preds.numel() == 0:
        return 0.0
    preds_mean = torch.mean(preds)
    labels_mean = torch.mean(labels)
    preds_centered = preds - preds_mean
    labels_centered = labels - labels_mean
    numerator = torch.sum(preds_centered * labels_centered)
    denom = torch.sqrt(torch.sum(preds_centered ** 2) * torch.sum(labels_centered ** 2))
    if denom.item() == 0.0:
        return 0.0
    return (numerator / denom).item()


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return torch.mean((preds == labels).float()).item()


def aggregate_pointwise_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_func: str,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if logits.numel() == 0:
        return metrics

    metrics["rmse"] = rmse(logits, labels)
    metrics["mae"] = mae(logits, labels)
    metrics["corr"] = pearson_correlation(logits, labels)

    if loss_func == "binary_cross_entropy":
        unique_vals = torch.unique(labels).tolist()
        if all(v in (0.0, 1.0) for v in unique_vals):
            metrics["accuracy"] = binary_accuracy(logits, labels)
    return metrics
