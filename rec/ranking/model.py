from __future__ import annotations

import itertools
from typing import Dict

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
    ) -> None:
        super().__init__()
        self.user_tower = TwoTowerEncoder(user_cardinalities, tower_config)
        self.item_tower = TwoTowerEncoder(item_cardinalities, tower_config)
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

    def forward(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        joint = torch.cat([user_emb, item_emb, user_emb * item_emb, torch.abs(user_emb - item_emb)], dim=-1)
        return self.scorer(joint).squeeze(-1)

    def score_all(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        user_emb_exp = user_emb.unsqueeze(1).expand(-1, item_emb.size(0), -1)
        item_emb_exp = item_emb.unsqueeze(0).expand(user_emb.size(0), -1, -1)
        joint = torch.cat(
            [user_emb_exp, item_emb_exp, user_emb_exp * item_emb_exp, torch.abs(user_emb_exp - item_emb_exp)],
            dim=-1,
        )
        scores = self.scorer(joint).squeeze(-1)
        return scores

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_prefix = "user_"
        item_prefix = "item_"
        user_features = {k[len(user_prefix):]: v for k, v in batch.items() if k.startswith(user_prefix)}
        item_features = {k[len(item_prefix):]: v for k, v in batch.items() if k.startswith(item_prefix)}
        labels = batch["label"]
        logits = self.forward(user_features, item_features)
        return self.loss_fn(logits, labels)

class DLRM(nn.Module):
    def __init__(
        self,
        user_cardinalities: Dict[str, int],
        item_cardinalities: Dict[str, int],
        tower_config: TowerConfig,
        lr: float = 1e-3,
        loss_func: str | None = None,
    ) -> None:
        super().__init__() 
        
        cardinalities = {**user_cardinalities, **item_cardinalities}
        
        hidden_dims = tower_config.hidden_dims or []

        self.encoder = BaseEncoder(cardinalities, tower_config)
        self.mlp = MLP(
            in_dim = tower_config.embedding_dim * len(cardinalities) + (len(cardinalities) * (len(cardinalities) - 1)) // 2, 
            hidden_dims = hidden_dims + [1], 
            dropout = tower_config.dropout, 
            activate_last = False
        )

        self.lr = lr
        self.loss_func = loss_func or "binary_cross_entropy"
        if self.loss_func == "binary_cross_entropy":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.loss_func == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported ranking loss_func: {self.loss_func}") 
    
    def interact(self, features):
        # features: List[Tensor[B, D]]
        interactions = []
        for x, y in itertools.combinations(features, 2):
            interactions.append(torch.sum(x * y, dim=1, keepdim=True))
        return torch.cat(interactions, dim=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = {name: self.encoder.embeddings[name](features[name]) for name in self.encoder.feature_names}
        features_list = list(embs.values())
        interactions = self.interact(features_list)
        joint = torch.cat(features_list + [interactions], dim=1)
        return self.mlp(joint).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = {k: v for k, v in batch.items() if k != "label"}
        labels = batch["label"]
        logits = self.forward(features)
        return self.loss_fn(logits, labels)

def load_retrieval_towers(ranking_model: TwoTowerRanking, retrieval_state: Dict[str, torch.Tensor]) -> None:
    user_prefix = "user_tower."
    item_prefix = "item_tower."
    user_state = {k.replace(user_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(user_prefix)}
    item_state = {k.replace(item_prefix, ""): v for k, v in retrieval_state.items() if k.startswith(item_prefix)}
    ranking_model.user_tower.load_state_dict(user_state, strict=False)
    ranking_model.item_tower.load_state_dict(item_state, strict=False)
