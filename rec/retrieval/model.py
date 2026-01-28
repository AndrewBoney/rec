from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import lightning.pytorch as lit

from ..common.model import TowerConfig, TwoTowerEncoder
from ..common.metrics import topk_metrics_from_indices


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
        self._cached_item_emb: Optional[torch.Tensor] = None
        self._feature_store = None

    def forward(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        scores = user_emb @ item_emb.T
        return scores

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        scores = self.forward(user_features, item_features) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        if self.trainer is None or self.trainer.datamodule is None:
            return
        feature_store = getattr(self.trainer.datamodule, "feature_store", None)
        if feature_store is None:
            return
        self._feature_store = feature_store
        item_features = feature_store.get_all_item_features()
        item_features = {k: v.to(self.device) for k, v in item_features.items()}
        with torch.no_grad():
            self._cached_item_emb = self.item_tower(item_features)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        user_prefix = "user_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        if self._cached_item_emb is None:
            self.on_validation_epoch_start()
        if self._cached_item_emb is None or self._feature_store is None:
            raise RuntimeError("Validation requires cached item embeddings and feature store")

        user_emb = self.user_tower({k: v.to(self.device) for k, v in user_features.items()})
        scores = user_emb @ self._cached_item_emb.T
        labels = self._feature_store.map_item_ids_to_indices(batch["item_id"]).to(scores.device)
        loss = F.cross_entropy(scores / self.temperature, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        metrics = topk_metrics_from_indices(scores, labels, ks=[5, 10, 20])
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=name in {"recall@10", "ndcg@10"}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
