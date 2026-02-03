from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def _as_list(ks: Iterable[int]) -> List[int]:
    return sorted({int(k) for k in ks if int(k) > 0})


def recall_at_k(hits: torch.Tensor, num_relevant: int, k: int) -> float:
    if num_relevant == 0:
        return 0.0
    return hits[:k].sum().item() / float(num_relevant)


def precision_at_k(hits: torch.Tensor, k: int) -> float:
    return hits[:k].sum().item() / float(k)


def dcg_at_k(hits: torch.Tensor, k: int) -> float:
    if hits[:k].any():
        ranks = torch.arange(1, k + 1, dtype=torch.float32, device=hits.device)
        return torch.sum(hits[:k].float() / torch.log2(ranks + 1.0)).item()
    return 0.0


def idcg_at_k(num_relevant: int, k: int) -> float:
    limit = min(num_relevant, k)
    if limit == 0:
        return 0.0
    ranks = torch.arange(1, limit + 1, dtype=torch.float32)
    return torch.sum(1.0 / torch.log2(ranks + 1.0)).item()


def ndcg_at_k(hits: torch.Tensor, num_relevant: int, k: int) -> float:
    ideal = idcg_at_k(num_relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(hits, k) / ideal


def mrr(hits: torch.Tensor) -> float:
    if not hits.any():
        return 0.0
    ranks = torch.arange(1, hits.numel() + 1, dtype=torch.float32, device=hits.device)
    first_rank = ranks[hits][0].item()
    return 1.0 / first_rank


def aggregate_retrieval_metrics(
    topk_indices: torch.Tensor,
    relevant_indices: Sequence[torch.Tensor],
    ks: Iterable[int],
) -> Dict[str, float]:
    ks_list = _as_list(ks)
    if not ks_list or topk_indices.numel() == 0:
        return {}

    max_k = max(ks_list)
    if topk_indices.size(1) < max_k:
        raise ValueError("topk_indices must have at least max(k) columns")

    totals = {f"recall@{k}": 0.0 for k in ks_list}
    totals.update({f"precision@{k}": 0.0 for k in ks_list})
    totals.update({f"ndcg@{k}": 0.0 for k in ks_list})
    totals["mrr"] = 0.0

    num_users = topk_indices.size(0)
    for idx in range(num_users):
        topk = topk_indices[idx]
        rel = relevant_indices[idx]
        if rel.numel() == 0:
            continue
        hits = torch.isin(topk, rel)

        totals["mrr"] += mrr(hits)
        num_rel = int(rel.numel())
        for k in ks_list:
            totals[f"recall@{k}"] += recall_at_k(hits, num_rel, k)
            totals[f"precision@{k}"] += precision_at_k(hits, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(hits, num_rel, k)

    if num_users == 0:
        return {}
    return {k: v / float(num_users) for k, v in totals.items()}
