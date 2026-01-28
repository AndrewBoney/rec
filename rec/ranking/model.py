from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
import lightning.pytorch as lit

from ..common.model import MLP, TowerConfig, TwoTowerEncoder
from ..common.metrics import topk_metrics_from_labels


class TwoTowerRanking(lit.LightningModule):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config)
        joint_dim = self.user_tower.output_dim + self.item_tower.output_dim
        self.scorer = MLP(joint_dim * 2, [128, 64, 1], tower_config.dropout, activate_last=False)
        self.lr = lr

    def forward(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        joint = torch.cat([user_emb, item_emb, user_emb * item_emb, torch.abs(user_emb - item_emb)], dim=-1)
        return self.scorer(joint).squeeze(-1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        labels = batch["label"]
        logits = self.forward(user_features, item_features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        labels = batch["label"]
        logits = self.forward(user_features, item_features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        metrics = topk_metrics_from_labels(logits, labels, ks=[5, 10, 20])
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=name in {"recall@10", "ndcg@10"}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def load_retrieval_towers(ranking_model: TwoTowerRanking, retrieval_state: Dict[str, torch.Tensor]) -> None:
    user_prefix = "user_tower."
    item_prefix = "item_tower."
    user_state = {k.replace(user_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(user_prefix)}
    item_state = {k.replace(item_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(item_prefix)}
    ranking_model.user_tower.load_state_dict(user_state, strict=False)
    ranking_model.item_tower.load_state_dict(item_state, strict=False)
