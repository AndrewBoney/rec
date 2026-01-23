from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
import lightning.pytorch as lit

from ..common.model import TowerConfig, TwoTowerEncoder


class TwoTowerRetrieval(lit.LightningModule):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        lr: float = 1e-3,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config)
        self.lr = lr
        self.temperature = temperature

    def forward(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        scores = user_emb @ item_emb.T
        return scores

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        user_features = {k.replace("user_", ""): v for k, v in batch.items() if k.startswith("user_")}
        item_features = {k.replace("item_", ""): v for k, v in batch.items() if k.startswith("item_")}
        scores = self.forward(user_features, item_features) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
