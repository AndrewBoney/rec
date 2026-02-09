from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


@dataclass
class FeatureConfig:
    user_id_col: str
    item_id_col: str
    user_cat_cols: List[str]
    item_cat_cols: List[str]
    interaction_user_col: str
    interaction_item_col: str
    user_dense_cols: List[str] = None
    item_dense_cols: List[str] = None
    interaction_time_col: Optional[str] = None
    interaction_label_col: Optional[str] = None

    def __post_init__(self):
        if self.user_dense_cols is None:
            self.user_dense_cols = []
        if self.item_dense_cols is None:
            self.item_dense_cols = []


class CategoryEncoder:
    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}
        self.unknown_index: int = 0

    def fit(self, values: Iterable) -> None:
        for v in values:
            str_v = str(v)
            if str_v not in self.mapping:
                self.mapping[str_v] = len(self.mapping) + 1

    def transform(self, values: Sequence) -> np.ndarray:
        str_values = [str(v) for v in values]
        return np.array([self.mapping.get(v, self.unknown_index) for v in str_values], dtype=np.int64)

    @property
    def num_embeddings(self) -> int:
        return len(self.mapping) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "category", "mapping": self.mapping}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryEncoder":
        enc = cls()
        enc.mapping = {str(k): int(v) for k, v in data["mapping"].items()}
        return enc


class DenseEncoder:
    def __init__(self) -> None:
        self.mean: float = 0.0
        self.std: float = 1.0
        self.fitted: bool = False

    def fit(self, values: Iterable) -> None:
        float_values = []
        for v in values:
            try:
                float_values.append(float(v))
            except (ValueError, TypeError):
                float_values.append(np.nan)

        vals = np.array(float_values, dtype=np.float32)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            self.mean = float(np.mean(vals))
            self.std = float(np.std(vals))
            if self.std == 0.0:
                self.std = 1.0
        self.fitted = True

    def transform(self, values: Sequence) -> np.ndarray:
        vals = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float32)
        vals = np.nan_to_num(vals, nan=self.mean)
        return (vals - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "dense",
            "mean": self.mean,
            "std": self.std,
            "fitted": self.fitted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DenseEncoder":
        enc = cls()
        enc.mean = data["mean"]
        enc.std = data["std"]
        enc.fitted = data["fitted"]
        return enc


def encode_dataframe(
    df: pd.DataFrame,
    encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    cols: Sequence[str],
) -> Dict[str, torch.Tensor]:
    encoded: Dict[str, torch.Tensor] = {}
    for col in cols:
        encoded[col] = torch.from_numpy(encoders[col].transform(df[col].tolist()))
    return encoded


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_encoders(
    users_path: str,
    items_path: str,
    interactions_path: str,
    feature_cfg: FeatureConfig,
    chunksize: int = 200_000,
) -> Tuple[Dict[str, Union[CategoryEncoder, DenseEncoder]],
           Dict[str, Union[CategoryEncoder, DenseEncoder]]]:
    from .io import read_parquet_batches

    user_encoders = {}
    item_encoders = {}

    # Initialize categorical encoders
    for col in [feature_cfg.user_id_col] + feature_cfg.user_cat_cols:
        user_encoders[col] = CategoryEncoder()
    for col in [feature_cfg.item_id_col] + feature_cfg.item_cat_cols:
        item_encoders[col] = CategoryEncoder()

    # Initialize dense encoders
    for col in feature_cfg.user_dense_cols:
        user_encoders[col] = DenseEncoder()
    for col in feature_cfg.item_dense_cols:
        item_encoders[col] = DenseEncoder()

    # Fit on users
    for chunk in read_parquet_batches(users_path, batch_size=chunksize):
        for col in [feature_cfg.user_id_col] + feature_cfg.user_cat_cols + feature_cfg.user_dense_cols:
            user_encoders[col].fit(chunk[col].tolist())

    # Fit on items
    for chunk in read_parquet_batches(items_path, batch_size=chunksize):
        for col in [feature_cfg.item_id_col] + feature_cfg.item_cat_cols + feature_cfg.item_dense_cols:
            item_encoders[col].fit(chunk[col].tolist())

    # Fit ID encoders on interactions
    for chunk in read_parquet_batches(interactions_path, batch_size=chunksize):
        user_encoders[feature_cfg.user_id_col].fit(chunk[feature_cfg.interaction_user_col].tolist())
        item_encoders[feature_cfg.item_id_col].fit(chunk[feature_cfg.interaction_item_col].tolist())

    return user_encoders, item_encoders


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
        user_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
        item_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
        feature_cfg: FeatureConfig,
    ) -> None:
        self.feature_cfg = feature_cfg
        self.user_encoders = user_encoders
        self.item_encoders = item_encoders

        # Encode ALL features (categorical + dense) in one call
        user_cols = [feature_cfg.user_id_col] + feature_cfg.user_cat_cols + feature_cfg.user_dense_cols
        item_cols = [feature_cfg.item_id_col] + feature_cfg.item_cat_cols + feature_cfg.item_dense_cols

        self.user_features = encode_dataframe(user_df, user_encoders, user_cols)
        self.item_features = encode_dataframe(item_df, item_encoders, item_cols)

        # Apply padding once to all features
        self.user_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v])
            for k, v in self.user_features.items()
        }
        self.item_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v])
            for k, v in self.item_features.items()
        }

        # Build index
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
        feature_store: FeatureStore,
        chunksize: int = 200_000,
        batch_size: int = 2048,
        negatives_per_pos: int = 0,
        item_id_pool: Optional[np.ndarray] = None,
        include_labels: bool = False,
    ) -> None:
        super().__init__()
        self.interactions_path = interactions_path
        self.feature_store = feature_store
        self.feature_cfg = feature_store.feature_cfg
        self.user_encoders = feature_store.user_encoders
        self.item_encoders = feature_store.item_encoders
        self.chunksize = chunksize
        self.batch_size = batch_size
        self.negatives_per_pos = negatives_per_pos
        self.item_id_pool = item_id_pool
        self.include_labels = include_labels

    def __len__(self) -> int:
        parquet = pq.ParquetFile(self.interactions_path)
        num_rows = parquet.metadata.num_rows if parquet.metadata else 0
        if num_rows == 0:
            return 0
        pos_batches = math.ceil(num_rows / self.batch_size)
        return pos_batches * (1 + self.negatives_per_pos)

    def _sample_negatives(self, size: int) -> np.ndarray:
        if self.item_id_pool is None:
            raise ValueError("item_id_pool is required for negative sampling")
        return np.random.choice(self.item_id_pool, size=size, replace=True)

    def __iter__(self):
        from .io import read_parquet_batches

        for chunk in read_parquet_batches(self.interactions_path, self.chunksize):
            user_ids = self.user_encoders[self.feature_cfg.user_id_col].transform(
                chunk[self.feature_cfg.interaction_user_col].astype(str).tolist()
            )
            item_ids = self.item_encoders[self.feature_cfg.item_id_col].transform(
                chunk[self.feature_cfg.interaction_item_col].astype(str).tolist()
            )
            label_col = self.feature_cfg.interaction_label_col

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
                    if label_col and label_col in chunk.columns:
                        labels = chunk[label_col].iloc[start:end].to_numpy(dtype=np.float32, copy=False)
                        batch["label"] = torch.from_numpy(labels)
                    else:
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
    user_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    item_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
) -> FeatureStore:
    from .io import read_table

    users_df = read_table(paths.users_path)
    items_df = read_table(paths.items_path)
    return FeatureStore(
        users_df,
        items_df,
        user_encoders,
        item_encoders,
        feature_cfg,
    )
