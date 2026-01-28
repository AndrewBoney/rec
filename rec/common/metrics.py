from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def _as_list(ks: Iterable[int]) -> List[int]:
    return list(ks)


def topk_metrics_from_indices(
    scores: torch.Tensor,
    target_indices: torch.Tensor,
    ks: Iterable[int],
) -> Dict[str, torch.Tensor]:
    ks = _as_list(ks)
    max_candidates = scores.size(1)
    ks = [k for k in ks if k <= max_candidates]
    if not ks:
        return {}
    max_k = max(ks)
    topk = torch.topk(scores, max_k, dim=1).indices
    target_indices = target_indices.unsqueeze(1)
    matches = topk == target_indices
    hit = matches.any(dim=1)
    rank = torch.argmax(matches.int(), dim=1) + 1

    metrics: Dict[str, torch.Tensor] = {}
    reciprocal_rank = torch.where(
        hit, 1.0 / rank.float(), torch.zeros_like(rank, dtype=torch.float32)
    )
    metrics["mrr"] = reciprocal_rank.mean()

    for k in ks:
        hit_k = hit & (rank <= k)
        recall_k = hit_k.float().mean()
        precision_k = (hit_k.float() / float(k)).mean()
        ndcg_k = torch.where(
            hit_k,
            1.0 / torch.log2(rank.float() + 1.0),
            torch.zeros_like(rank, dtype=torch.float32),
        ).mean()
        metrics[f"recall@{k}"] = recall_k
        metrics[f"precision@{k}"] = precision_k
        metrics[f"ndcg@{k}"] = ndcg_k
    return metrics


def topk_metrics_from_labels(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ks: Iterable[int],
) -> Dict[str, torch.Tensor]:
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)
    target_indices = torch.argmax(labels, dim=1)
    return topk_metrics_from_indices(scores, target_indices, ks)
