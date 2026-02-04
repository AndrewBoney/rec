from __future__ import annotations
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class TowerConfig:
    embedding_dim: int = 64
    hidden_dims: Optional[List[int]] = None
    dropout: float = 0.1


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float, activate_last: bool = True) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == (len(dims) - 2)
            if activate_last or not is_last:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# TODO: implement EmbeddingBag version for high-cardinality features
class BaseEncoder(nn.Module):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_cardinalities.keys())
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, config.embedding_dim)
                for name, cardinality in feature_cardinalities.items()
            }
        )

class CatTwoTowerEncoder(BaseEncoder):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
    ) -> None:
        super().__init__(feature_cardinalities, config)
        input_dim = config.embedding_dim * len(self.feature_names)
        hidden_dims = config.hidden_dims or [128, 64]
        self.mlp = MLP(input_dim, hidden_dims, config.dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_list = [self.embeddings[name](features[name]) for name in self.feature_names]
        x = torch.cat(emb_list, dim=-1)
        x = self.mlp(x)
        return x

class StackedTwoTowerEncoder(BaseEncoder):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
    ) -> None:
        super().__init__(feature_cardinalities, config)
        input_dim = config.embedding_dim
        hidden_dims = config.hidden_dims or [128, 64]

        # define learnable weights for each feature embedding
        self.weights = nn.Parameter(torch.empty(config.embedding_dim, len(self.feature_names)))
        nn.init.xavier_uniform_(self.weights)

        # define MLP for combined embeddings
        self.mlp = MLP(input_dim, hidden_dims, config.dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # combine embeddings
        embs = torch.stack(
            [self.embeddings[name](features[name]) for name in self.feature_names],
            dim=-1,
        )

        # apply softmax to weights and combine embeddings
        weights = F.softmax(self.weights, dim=1).unsqueeze(0)
        weighted_embs = embs * weights
        x = weighted_embs.sum(dim=-1)

        # pass through MLP
        x = self.mlp(x)

        # normalize output embeddings
        x = F.normalize(x, dim=-1)

        return x