from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..common.model import TowerConfig, StackedEncoder as TwoTowerEncoder

class TwoTowerRetrieval(nn.Module):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        lr: float = 1e-3,
        init_temperature: float = 0.05,
        loss_func: str | None = None,
        user_dense_features: Optional[List[str]] = None,
        item_dense_features: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config, dense_feature_names=user_dense_features)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config, dense_feature_names=item_dense_features)
        self.lr = lr
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.loss_func = loss_func or "cross_entropy"
        if self.loss_func != "cross_entropy":
            raise ValueError(f"Unsupported retrieval loss_func: {self.loss_func}")
        self.loss_fn = nn.CrossEntropyLoss()

    def _prep_features(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        return user_features, item_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_features, item_features = self._prep_features(batch)
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        scores = user_emb @ item_emb.T
        return scores / self.temperature

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = self.forward(batch) 
        labels = torch.arange(scores.size(0), device=scores.device)
        return self.loss_fn(scores, labels)

    def get_topk_scores(
        self,
        feature_store,
        k: int,
        seen_user_item_map: Optional[Dict[int, List[int]]] = None,
        user_batch_size: int = 2048,
    ) -> torch.Tensor:
        all_user_features = feature_store.get_all_user_features()
        all_item_features = feature_store.get_all_item_features()
        user_ids = feature_store.get_all_user_ids()

        num_users = user_ids.numel()
        num_items = next(iter(all_item_features.values())).shape[0]
        if num_items == 0:
            return torch.empty((num_users, 0), dtype=torch.long)
        k = min(int(k), num_items)
        if k <= 0:
            return torch.empty((num_users, 0), dtype=torch.long)

        device = next(self.parameters()).device

        # Build seen-item index map once on CPU
        uid_list = user_ids.tolist()
        uid_to_row = {int(uid): idx for idx, uid in enumerate(uid_list)}
        seen_indices_map: Dict[int, torch.Tensor] = {}
        if seen_user_item_map:
            for uid, seen_item_ids in seen_user_item_map.items():
                row = uid_to_row.get(int(uid))
                if row is None or not seen_item_ids:
                    continue
                seen_indices_map[row] = feature_store.map_item_ids_to_indices(
                    torch.tensor(seen_item_ids, dtype=torch.long)
                )

        # Compute item embeddings once — shared across all user batches
        all_item_features_dev = {n: t.to(device) for n, t in all_item_features.items()}
        with torch.no_grad():
            item_emb = self.item_tower(all_item_features_dev)  # [num_items, D]

        topk_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, num_users, user_batch_size):
                end = min(start + user_batch_size, num_users)
                user_batch = {n: t[start:end].to(device) for n, t in all_user_features.items()}
                user_emb = self.user_tower(user_batch)
                scores = (user_emb @ item_emb.T) / self.temperature  # [batch, num_items]
                for local_i, global_row in enumerate(range(start, end)):
                    seen_idx = seen_indices_map.get(global_row)
                    if seen_idx is not None and seen_idx.numel() > 0:
                        scores[local_i, seen_idx.to(device)] = -torch.inf
                topk_chunks.append(torch.topk(scores, k, dim=1).indices.cpu())

        return torch.cat(topk_chunks, dim=0)


RETRIEVAL_MODEL_REGISTRY = {
    "two_tower": TwoTowerRetrieval,
}


def get_model_class(arch: str) -> type[nn.Module]:
    key = (arch or "").lower()
    if key in RETRIEVAL_MODEL_REGISTRY:
        return RETRIEVAL_MODEL_REGISTRY[key]
    available = ", ".join(sorted(RETRIEVAL_MODEL_REGISTRY))
    raise ValueError(f"Unsupported retrieval model_arch: {arch}. Available: {available}")
