from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def _as_list(ks: Iterable[int]) -> List[int]:
    return sorted({int(k) for k in ks})


def _ideal_dcg(num_relevant: int, k: int) -> float:
    limit = min(num_relevant, k)
    if limit == 0:
        return 0.0
    ranks = torch.arange(1, limit + 1, dtype=torch.float32)
    return torch.sum(1.0 / torch.log2(ranks + 1.0)).item()


def aggregate_ranking_metrics(
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
        ranks = torch.arange(1, topk.numel() + 1, dtype=torch.float32)

        if hits.any():
            first_rank = ranks[hits][0].item()
            totals["mrr"] += 1.0 / first_rank

        for k in ks_list:
            hits_k = hits[:k]
            num_hits = hits_k.sum().item()
            totals[f"recall@{k}"] += num_hits / float(rel.numel())
            totals[f"precision@{k}"] += num_hits / float(k)

            if num_hits > 0:
                rank_slice = ranks[:k]
                dcg = torch.sum((hits_k.float() / torch.log2(rank_slice + 1.0))).item()
            else:
                dcg = 0.0
            idcg = _ideal_dcg(int(rel.numel()), k)
            totals[f"ndcg@{k}"] += (dcg / idcg) if idcg > 0 else 0.0

    if num_users == 0:
        return {}
    return {k: v / float(num_users) for k, v in totals.items()}
