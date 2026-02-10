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
    dense_bottom_mlp_dims: Optional[List[int]] = None


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


class DenseBottomMLP(nn.Module):
    """Bottom MLP for processing dense features before combining with embeddings."""

    def __init__(
        self,
        num_dense_features: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_dense_features = num_dense_features
        if num_dense_features == 0 or not hidden_dims:
            self.mlp = nn.Identity()
            self.output_dim = 0
        else:
            self.mlp = MLP(num_dense_features, hidden_dims, dropout)
            self.output_dim = hidden_dims[-1]

    def forward(self, dense_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            dense_features: Dict mapping feature names to tensors of shape [B, 1] or [B]
        Returns:
            Tensor of shape [B, output_dim] (or [B, 0] if no dense features)
        """
        feature_list = [dense_features[name] for name in sorted(dense_features.keys())]
        x = torch.cat(feature_list, dim=-1)
        return self.mlp(x)


# TODO: implement torchrec version for larger than RAM embedding tables
class BaseEncoder(nn.Module):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
        dense_feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_cardinalities.keys())
        self.dense_feature_names = dense_feature_names or []
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, config.embedding_dim)
                for name, cardinality in feature_cardinalities.items()
            }
        )

        # Create dense bottom MLP
        if self.dense_feature_names:
            dense_mlp_dims = config.dense_bottom_mlp_dims or []
            self.dense_bottom_mlp = DenseBottomMLP(
                num_dense_features=len(self.dense_feature_names),
                hidden_dims=dense_mlp_dims,
                dropout=config.dropout,
            )

class CatEncoder(BaseEncoder):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
        dense_feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(feature_cardinalities, config, dense_feature_names)
        cat_dim = config.embedding_dim * len(self.feature_names)
        dense_dim = self.dense_bottom_mlp.output_dim
        input_dim = cat_dim + dense_dim
        hidden_dims = config.hidden_dims or [128, 64]
        self.mlp = MLP(input_dim, hidden_dims, config.dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process categorical features
        emb_list = [self.embeddings[name](features[name]) for name in self.feature_names]
        cat_emb = torch.cat(emb_list, dim=-1)

        # Process dense features
        if self.dense_feature_names:
            dense_features = {
                name: features[name].unsqueeze(-1) if features[name].dim() == 1 else features[name]
                for name in self.dense_feature_names
            }
            dense_emb = self.dense_bottom_mlp(dense_features)
            x = torch.cat([cat_emb, dense_emb], dim=-1)
        else:
            x = cat_emb

        x = self.mlp(x)
        return x

class StackedEncoder(BaseEncoder):
    def __init__(
        self,
        feature_cardinalities: Dict[str, int],
        config: TowerConfig,
        dense_feature_names: Optional[List[str]] = None,
    ) -> None:
        if config.dense_bottom_mlp_dims:
            assert config.dense_bottom_mlp_dims[-1] == config.embedding_dim, "Dense bottom MLP output dim must match embedding_dim for stacking"
        super().__init__(feature_cardinalities, config, dense_feature_names)
        cat_dim = config.embedding_dim
        input_dim = cat_dim
        hidden_dims = config.hidden_dims or [128, 64]

        # define learnable weights for each feature embedding
        self.weights = nn.Parameter(torch.empty(config.embedding_dim, len(self.feature_names) + int(len(self.dense_feature_names) > 0)))
        nn.init.xavier_uniform_(self.weights)

        # define MLP for combined embeddings
        self.mlp = MLP(input_dim, hidden_dims, config.dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # combine embeddings
        cat_features = [self.embeddings[name](features[name]) for name in self.feature_names]

        if len(self.dense_feature_names) > 0:
            dense_features_dict = {
                name: features[name].unsqueeze(-1) if features[name].dim() == 1 else features[name]
                for name in self.dense_feature_names
            }
            dense_emb = self.dense_bottom_mlp(dense_features_dict)
            x = cat_features + [dense_emb]
        else:
            x = cat_features

        embs = torch.stack(
            x,
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