from __future__ import annotations

from typing import Dict

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
    ) -> None:
        super().__init__()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config)
        self.lr = lr
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.loss_func = loss_func or "cross_entropy"
        if self.loss_func != "cross_entropy":
            raise ValueError(f"Unsupported retrieval loss_func: {self.loss_func}")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return self.score_all(user_emb, item_emb)

    def score_all(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        scores = user_emb @ item_emb.T
        return scores / self.temperature

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        scores = self.forward(user_features, item_features) 
        labels = torch.arange(scores.size(0), device=scores.device)
        return self.loss_fn(scores, labels)
