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

        self.num_dense_features = len(dense_features)

        hidden_dims = tower_config.hidden_dims or []

        self.encoder = BaseEncoder(cardinalities, tower_config, dense_feature_names=dense_features)

        # Calculate input dimension for final MLP
        cat_dim = tower_config.embedding_dim * num_cardinalities
        interaction_dim = (num_cardinalities * (num_cardinalities - 1)) // 2
        dense_dim = self.encoder.dense_bottom_mlp.output_dim if dense_features else 0

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
    
    def _remove_prefixes(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k.removeprefix("user_").removeprefix("item_"): v for k, v in batch.items() if "_" in k}

    def interact(self, features : List[torch.Tensor]) -> torch.Tensor:
        # features: List[Tensor[B, D]]
        interactions = []
        for x, y in itertools.combinations(features, 2):
            interactions.append(torch.sum(x * y, dim=1, keepdim=True))
        return torch.cat(interactions, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Remove prefixes to get raw feature names for encoder
        batch = self._remove_prefixes(batch)        

        # Process categorical features
        features_list = [self.encoder.embeddings[name](batch[name]) for name in self.encoder.feature_names]

        # Process dense features
        if self.num_dense_features > 0:
            dense_features = {
                name: batch[name].unsqueeze(-1) if batch[name].dim() == 1 else batch[name]
                for name in self.encoder.dense_feature_names
            } 
            dense_emb = self.encoder.dense_bottom_mlp(dense_features)
            features_list = features_list + [dense_emb]
            interactions = self.interact(features_list)
            joint = torch.cat(features_list + [interactions, dense_emb], dim=1)
        else:
            interactions = self.interact(features_list)
            joint = torch.cat(features_list + [interactions], dim=1)

        return self.mlp(joint).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["label"]
        logits = self.forward(batch)
        return self.loss_fn(logits, labels)

def load_retrieval_towers(
    ranking_model: nn.Module,
    retrieval_state: Dict[str, torch.Tensor],
    max_mappings_to_log: int = 200,
) -> None:
    """Load retrieval checkpoint weights into a ranking model via safe partial matching.

    Matching strategy (in order):
    1) Exact key match (same parameter name in both state dicts).
    2) Prefix rewrite match (maps known retrieval prefixes to ranking prefixes).
    3) Unique suffix match (same trailing path after the first module name).

    A weight is copied only when both key mapping and tensor shape are compatible.
    This keeps initialization robust across architectures while avoiding unsafe loads.

    Emits a short log report showing which retrieval parameters were loaded and
    where they mapped in the ranking model.
    """
    ranking_state = ranking_model.state_dict()

    # Known cross-architecture rewrites.
    # Example: retrieval `user_tower.embeddings.user_id.weight`
    #      -> ranking   `encoder.embeddings.user_id.weight` (DLRM case)
    rewrite_rules = (
        ("user_tower.embeddings.", "encoder.embeddings."),
        ("item_tower.embeddings.", "encoder.embeddings."),
    )

    # Build an index by suffix (everything after the first module name).
    # This supports generic fallback matching when module roots differ.
    suffix_index: Dict[str, List[str]] = {}
    for target_key in ranking_state:
        parts = target_key.split(".", 1)
        if len(parts) == 2:
            suffix_index.setdefault(parts[1], []).append(target_key)

    mapped_state: Dict[str, torch.Tensor] = {}
    loaded_mappings: List[Tuple[str, str]] = []
    skipped_non_tensor = 0
    skipped_no_match = 0
    skipped_shape_mismatch = 0
    skipped_ambiguous = 0

    for source_key, source_value in retrieval_state.items():
        if not isinstance(source_value, torch.Tensor):
            skipped_non_tensor += 1
            continue

        # Try exact key first, then any rewrite-derived candidates.
        candidate_keys = [source_key]
        for source_prefix, target_prefix in rewrite_rules:
            if source_key.startswith(source_prefix):
                candidate_keys.append(target_prefix + source_key[len(source_prefix):])

        # Use the first candidate with an identical target tensor shape.
        matched_key: Optional[str] = None
        had_shape_mismatch = False
        for candidate_key in candidate_keys:
            target_value = ranking_state.get(candidate_key)
            if target_value is None:
                continue
            if target_value.shape != source_value.shape:
                had_shape_mismatch = True
                continue
            matched_key = candidate_key
            break

        if matched_key is None:
            # Fallback: match by unique suffix + shape.
            # If multiple targets share the same suffix, skip to avoid ambiguity.
            source_parts = source_key.split(".", 1)
            if len(source_parts) == 2:
                suffix = source_parts[1]
                suffix_matches = [
                    key
                    for key in suffix_index.get(suffix, [])
                    if ranking_state[key].shape == source_value.shape
                ]
                if len(suffix_matches) == 1:
                    matched_key = suffix_matches[0]
                elif len(suffix_matches) > 1:
                    skipped_ambiguous += 1

        # Keep the first successful assignment for each target parameter.
        # This avoids one target being overwritten by multiple source keys.
        if matched_key is not None and matched_key not in mapped_state:
            mapped_state[matched_key] = source_value
            loaded_mappings.append((source_key, matched_key))
        elif matched_key is None:
            if had_shape_mismatch:
                skipped_shape_mismatch += 1
            else:
                skipped_no_match += 1

    # `strict=False` allows partial initialization (matched subset only).
    ranking_model.load_state_dict(mapped_state, strict=False)

    print("[init_from_retrieval] Retrieval -> ranking parameter mapping")
    print(
        "[init_from_retrieval] "
        f"loaded={len(loaded_mappings)} "
        f"skipped_no_match={skipped_no_match} "
        f"skipped_shape_mismatch={skipped_shape_mismatch} "
        f"skipped_ambiguous={skipped_ambiguous} "
        f"skipped_non_tensor={skipped_non_tensor}"
    )

    if loaded_mappings:
        to_show = loaded_mappings[:max_mappings_to_log]
        for source_key, target_key in to_show:
            print(f"[init_from_retrieval]   {source_key} -> {target_key}")
        if len(loaded_mappings) > max_mappings_to_log:
            remaining = len(loaded_mappings) - max_mappings_to_log
            print(f"[init_from_retrieval]   ... {remaining} more mappings not shown")


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