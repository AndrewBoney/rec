from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..common.model import MLP, TowerConfig, BaseEncoder, StackedEncoder as TwoTowerEncoder


class TwoTowerRanking(nn.Module):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        scorer_hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        loss_func: str | None = None,
        user_dense_features: Optional[List[str]] = None,
        item_dense_features: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config, dense_feature_names=user_dense_features)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config, dense_feature_names=item_dense_features)
        joint_dim = self.user_tower.output_dim + self.item_tower.output_dim
        scorer_hidden_dims = scorer_hidden_dims or [128, 64]
        self.scorer = MLP(joint_dim * 2, scorer_hidden_dims + [1], tower_config.dropout, activate_last=False)
        self.lr = lr
        self.loss_func = loss_func or "binary_cross_entropy"
        if self.loss_func == "binary_cross_entropy":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.loss_func == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported ranking loss_func: {self.loss_func}")

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
        joint = torch.cat([user_emb, item_emb, user_emb * item_emb, torch.abs(user_emb - item_emb)], dim=-1)
        return self.scorer(joint).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["label"]
        logits = self.forward(batch)
        return self.loss_fn(logits, labels)

class DLRM(nn.Module):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        scorer_hidden_dims: list[int] | None = None, # TODO: kept this for now so args are consistent with TwoTowerRanking, but consider removing
        lr: float = 1e-3,
        loss_func: str | None = None,
        user_dense_features: Optional[List[str]] = None,
        item_dense_features: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        cardinalities = {**user_cardinalities, **item_cardinalities}
        dense_features = (user_dense_features or []) + (item_dense_features or [])
        num_cardinalities = len(cardinalities)

        hidden_dims = tower_config.hidden_dims or []

        self.encoder = BaseEncoder(cardinalities, tower_config, dense_feature_names=dense_features)

        # Calculate input dimension for final MLP
        cat_dim = tower_config.embedding_dim * num_cardinalities
        interaction_dim = (num_cardinalities * (num_cardinalities - 1)) // 2
        dense_dim = self.encoder.dense_bottom_mlp.output_dim

        self.mlp = MLP(
            in_dim=cat_dim + interaction_dim + dense_dim,
            hidden_dims=hidden_dims + [1],
            dropout=tower_config.dropout,
            activate_last=False
        )

        self.lr = lr
        self.loss_func = loss_func or "binary_cross_entropy"
        if self.loss_func == "binary_cross_entropy":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.loss_func == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported ranking loss_func: {self.loss_func}") 
    
    def interact(self, features : List[torch.Tensor]) -> torch.Tensor:
        # features: List[Tensor[B, D]]
        interactions = []
        for x, y in itertools.combinations(features, 2):
            interactions.append(torch.sum(x * y, dim=1, keepdim=True))
        return torch.cat(interactions, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process categorical features
        features_list = [self.encoder.embeddings[name](batch[name]) for name in self.encoder.feature_names]
        interactions = self.interact(features_list)

        # Process dense features
        dense_features = {
            name: batch[name].unsqueeze(-1) if batch[name].dim() == 1 else batch[name]
            for name in self.encoder.dense_feature_names if name in batch
        }
        dense_emb = self.encoder.dense_bottom_mlp(dense_features)

        # Combine all features
        if dense_emb.shape[-1] > 0:
            joint = torch.cat(features_list + [interactions, dense_emb], dim=1)
        else:
            joint = torch.cat(features_list + [interactions], dim=1)

        return self.mlp(joint).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["label"]
        logits = self.forward(batch)
        return self.loss_fn(logits, labels)

def load_retrieval_towers(ranking_model: TwoTowerRanking, retrieval_state: Dict[str, torch.Tensor]) -> None:
    user_prefix = "user_tower."
    item_prefix = "item_tower."
    user_state = {k.replace(user_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(user_prefix)}
    item_state = {k.replace(item_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(item_prefix)}
    ranking_model.user_tower.load_state_dict(user_state, strict=False)
    ranking_model.item_tower.load_state_dict(item_state, strict=False)


RANKING_MODEL_REGISTRY = {
    "two_tower": TwoTowerRanking,
    "dlrm": DLRM,
}


def get_model_class(arch: str) -> type[nn.Module]:
    key = (arch or "").lower()
    if key in RANKING_MODEL_REGISTRY:
        return RANKING_MODEL_REGISTRY[key]
    available = ", ".join(sorted(RANKING_MODEL_REGISTRY))
    raise ValueError(f"Unsupported ranking model_arch: {arch}. Available: {available}")