from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .utils import CategoryEncoder, FeatureConfig, encode_dataframe, read_parquet_batches, read_table


@dataclass
class DataPaths:
    users_path: str
    items_path: str
    interactions_train_path: str
    interactions_val_path: str

# TODO: implement a faster version of this, not relying on pandas / python loops 
class FeatureStore:
    def __init__(
        self,
        user_df: pd.DataFrame,
        item_df: pd.DataFrame,
        user_encoders: Dict[str, CategoryEncoder],
        item_encoders: Dict[str, CategoryEncoder],
        feature_cfg: FeatureConfig,
    ) -> None:
        self.feature_cfg = feature_cfg
        self.user_features = encode_dataframe(
            user_df,
            user_encoders,
            [feature_cfg.user_id_col] + feature_cfg.user_cat_cols,
        )
        self.item_features = encode_dataframe(
            item_df,
            item_encoders,
            [feature_cfg.item_id_col] + feature_cfg.item_cat_cols,
        )

        self.user_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v]) for k, v in self.user_features.items()
        }
        self.item_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v]) for k, v in self.item_features.items()
        }

        user_id_tensor = self.user_features[feature_cfg.user_id_col][1:]
        item_id_tensor = self.item_features[feature_cfg.item_id_col][1:]
        self.item_id_tensor = item_id_tensor
        self.user_index: Dict[int, int] = {
            int(uid): idx + 1 for idx, uid in enumerate(user_id_tensor.tolist())
        }
        self.item_index: Dict[int, int] = {
            int(iid): idx + 1 for idx, iid in enumerate(item_id_tensor.tolist())
        }

    def get_user_features(self, user_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        indices = [self.user_index.get(int(uid), 0) for uid in user_ids.tolist()]
        indices = torch.tensor(indices, dtype=torch.long)
        return {k: v[indices] for k, v in self.user_features.items()}

    def get_item_features(self, item_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        indices = [self.item_index.get(int(iid), 0) for iid in item_ids.tolist()]
        indices = torch.tensor(indices, dtype=torch.long)
        return {k: v[indices] for k, v in self.item_features.items()}

    def get_all_item_features(self) -> Dict[str, torch.Tensor]:
        return {k: v[1:] for k, v in self.item_features.items()}

    def get_all_item_ids(self) -> torch.Tensor:
        return self.item_id_tensor

    def map_item_ids_to_indices(self, item_ids: torch.Tensor) -> torch.Tensor:
        indices = [self.item_index.get(int(iid), 0) for iid in item_ids.tolist()]
        indices = torch.tensor(indices, dtype=torch.long)
        indices = torch.clamp(indices - 1, min=0)
        return indices


class InteractionIterableDataset(IterableDataset):
    def __init__(
        self,
        interactions_path: str,
        feature_cfg: FeatureConfig,
        user_encoders: Dict[str, CategoryEncoder],
        item_encoders: Dict[str, CategoryEncoder],
        feature_store: FeatureStore,
        chunksize: int = 200_000,
        batch_size: int = 2048,
        negatives_per_pos: int = 0,
        item_id_pool: Optional[np.ndarray] = None,
        include_labels: bool = False,
    ) -> None:
        super().__init__()
        self.interactions_path = interactions_path
        self.feature_cfg = feature_cfg
        self.user_encoders = user_encoders
        self.item_encoders = item_encoders
        self.feature_store = feature_store
        self.chunksize = chunksize
        self.batch_size = batch_size
        self.negatives_per_pos = negatives_per_pos
        self.item_id_pool = item_id_pool
        self.include_labels = include_labels

    def _sample_negatives(self, size: int) -> np.ndarray:
        if self.item_id_pool is None:
            raise ValueError("item_id_pool is required for negative sampling")
        return np.random.choice(self.item_id_pool, size=size, replace=True)

    def __iter__(self):
        for chunk in read_parquet_batches(self.interactions_path, self.chunksize):
            user_ids = self.user_encoders[self.feature_cfg.user_id_col].transform(
                chunk[self.feature_cfg.interaction_user_col].astype(str).tolist()
            )
            item_ids = self.item_encoders[self.feature_cfg.item_id_col].transform(
                chunk[self.feature_cfg.interaction_item_col].astype(str).tolist()
            )

            for start in range(0, len(user_ids), self.batch_size):
                end = start + self.batch_size
                user_ids_t = torch.from_numpy(user_ids[start:end])
                item_ids_t = torch.from_numpy(item_ids[start:end])

                user_feats = self.feature_store.get_user_features(user_ids_t)
                item_feats = self.feature_store.get_item_features(item_ids_t)
                batch = {
                    "user_id": user_ids_t,
                    "item_id": item_ids_t,
                    **{f"user_{k}": v for k, v in user_feats.items()},
                    **{f"item_{k}": v for k, v in item_feats.items()},
                }
                if self.include_labels:
                    batch["label"] = torch.ones(len(user_ids_t), dtype=torch.float32)
                yield batch

                for _ in range(self.negatives_per_pos):
                    neg_item_ids = self._sample_negatives(len(user_ids_t))
                    neg_item_ids_t = torch.from_numpy(neg_item_ids)
                    neg_item_feats = self.feature_store.get_item_features(neg_item_ids_t)
                    neg_batch = {
                        "user_id": user_ids_t,
                        "item_id": neg_item_ids_t,
                        **{f"user_{k}": v for k, v in user_feats.items()},
                        **{f"item_{k}": v for k, v in neg_item_feats.items()},
                    }
                    if self.include_labels:
                        neg_batch["label"] = torch.zeros(len(user_ids_t), dtype=torch.float32)
                    yield neg_batch


def build_feature_store(
    paths: DataPaths,
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
) -> FeatureStore:
    users_df = read_table(paths.users_path)
    items_df = read_table(paths.items_path)
    return FeatureStore(users_df, items_df, user_encoders, item_encoders, feature_cfg)
