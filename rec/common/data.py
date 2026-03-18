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


class Tokenizer:
    """Maps categorical values to integer indices.

    min_freq <= 1: every value gets its own index (0=OOV, 1..N=values)
    min_freq > 1:  low-frequency values share a tail bucket
                   (0=OOV, 1..N=head, N+1=tail)

    Fitting is lazy: counts accumulate across multiple fit() calls and indices
    are only assigned on the first transform() or num_embeddings access.
    """

    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.unknown_index: int = 0
        self._counts: Dict[str, int] = {}
        self.mapping: Dict[str, int] = {}
        self._finalized: bool = False
        self._tail_index: Optional[int] = None

    def fit(self, values: Iterable) -> None:
        if self._finalized:
            raise RuntimeError("Tokenizer cannot be fit after transform() has been called.")
        for v in values:
            s = str(v)
            self._counts[s] = self._counts.get(s, 0) + 1

    def _finalize(self) -> None:
        threshold = max(self.min_freq, 1)
        self.mapping = {}
        for v in sorted(self._counts):  # sorted for stable indices across runs
            if self._counts[v] >= threshold:
                self.mapping[v] = len(self.mapping) + 1
        if self.min_freq > 1:
            self._tail_index = len(self.mapping) + 1
        self._finalized = True

    @property
    def tail_index(self) -> int:
        if not self._finalized:
            self._finalize()
        if self._tail_index is None:
            raise AttributeError("tail_index is only defined when min_freq > 1")
        return self._tail_index

    def transform(self, values: Sequence) -> np.ndarray:
        if not self._finalized:
            self._finalize()
        out: List[int] = []
        for v in [str(x) for x in values]:
            if v in self.mapping:
                out.append(self.mapping[v])
            elif self._tail_index is not None and v in self._counts:
                out.append(self._tail_index)  # seen but below threshold
            else:
                out.append(self.unknown_index)  # true OOV
        return np.array(out, dtype=np.int64)

    @property
    def num_embeddings(self) -> int:
        if not self._finalized:
            self._finalize()
        return len(self.mapping) + (2 if self._tail_index is not None else 1)

    def to_dict(self) -> Dict[str, Any]:
        if not self._finalized:
            self._finalize()
        d: Dict[str, Any] = {"type": "tokenizer", "min_freq": self.min_freq, "mapping": self.mapping}
        if self._tail_index is not None:
            d["tail_index"] = self._tail_index
            d["tail_values"] = [v for v in self._counts if v not in self.mapping]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tokenizer":
        enc = cls(min_freq=data.get("min_freq", 1))
        enc.mapping = {str(k): int(v) for k, v in data["mapping"].items()}
        if "tail_index" in data:
            enc._tail_index = int(data["tail_index"])
            # Restore _counts so transform can distinguish tail from true OOV
            enc._counts = {str(k): enc.min_freq for k in enc.mapping}
            for v in data.get("tail_values", []):
                enc._counts[str(v)] = 1
        enc._finalized = True
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
    encoders: Dict[str, Union[Tokenizer, DenseEncoder]],
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
    cat_col_min_counts: Dict[str, int] = {},
) -> Tuple[Dict[str, Union[Tokenizer, DenseEncoder]],
           Dict[str, Union[Tokenizer, DenseEncoder]]]:
    from .io import read_parquet_batches

    user_encoders = {}
    item_encoders = {}

    # Initialize categorical encoders with min_freq.
    # Defaults to 5 (low-frequency values share a tail bucket).
    # Set a column to 0 or 1 to give every value its own index.
    for col in [feature_cfg.user_id_col] + feature_cfg.user_cat_cols:
        user_encoders[col] = Tokenizer(min_freq=cat_col_min_counts.get(col, 5))
    for col in [feature_cfg.item_id_col] + feature_cfg.item_cat_cols:
        item_encoders[col] = Tokenizer(min_freq=cat_col_min_counts.get(col, 5))

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
        user_encoders: Dict[str, Union[Tokenizer, DenseEncoder]],
        item_encoders: Dict[str, Union[Tokenizer, DenseEncoder]],
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

        # Apply padding once to all features (index 0 = unknown/OOV)
        self.user_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v])
            for k, v in self.user_features.items()
        }
        self.item_features = {
            k: torch.cat([torch.zeros(1, dtype=v.dtype), v])
            for k, v in self.item_features.items()
        }

        # Positional user index: raw user_id string → row position (1-indexed).
        # This is always a 1-to-1 mapping regardless of how user_id_col is encoded,
        # so every user gets their own correct feature row even when user_id uses
        # a Tokenizer with min_freq > 1 (where multiple users share the same embedding index).
        raw_user_ids = user_df[feature_cfg.user_id_col].astype(str).tolist()
        self._user_pos_index: Dict[str, int] = {uid: idx + 1 for idx, uid in enumerate(raw_user_ids)}
        self._user_positions = torch.arange(1, len(raw_user_ids) + 1, dtype=torch.long)

        # Positional item index: raw item_id string → row position (1-indexed).
        # Same rationale as user: decouples feature-store row lookup from embedding index.
        raw_item_ids = item_df[feature_cfg.item_id_col].astype(str).tolist()
        self._item_pos_index: Dict[str, int] = {iid: idx + 1 for idx, iid in enumerate(raw_item_ids)}
        self._item_positions = torch.arange(1, len(raw_item_ids) + 1, dtype=torch.long)
        self._raw_item_ids: List[str] = raw_item_ids

    def get_user_features(self, raw_user_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Look up user features by raw (unencoded) user_id strings."""
        indices = [self._user_pos_index.get(str(uid), 0) for uid in raw_user_ids]
        indices = torch.tensor(indices, dtype=torch.long)
        return {k: v[indices] for k, v in self.user_features.items()}

    def get_user_position(self, raw_user_id: str) -> int:
        """Return the 1-indexed feature store row for a raw user_id. Returns 0 for unknown."""
        return self._user_pos_index.get(str(raw_user_id), 0)

    def get_item_features(self, raw_item_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Look up item features by raw (unencoded) item_id strings."""
        indices = [self._item_pos_index.get(str(iid), 0) for iid in raw_item_ids]
        indices = torch.tensor(indices, dtype=torch.long)
        return {k: v[indices] for k, v in self.item_features.items()}

    def get_item_position(self, raw_item_id: str) -> int:
        """Return the 1-indexed feature store row for a raw item_id. Returns 0 for unknown."""
        return self._item_pos_index.get(str(raw_item_id), 0)

    def get_all_item_features(self) -> Dict[str, torch.Tensor]:
        return {k: v[1:] for k, v in self.item_features.items()}

    def get_all_user_features(self) -> Dict[str, torch.Tensor]:
        return {k: v[1:] for k, v in self.user_features.items()}

    def get_all_user_ids(self) -> torch.Tensor:
        """Return unique user position indices (1..N), one per row in users_df."""
        return self._user_positions

    def get_all_item_ids(self) -> torch.Tensor:
        """Return unique item position indices (1..M), one per row in items_df."""
        return self._item_positions

    def get_all_raw_item_ids(self) -> List[str]:
        """Return raw (unencoded) item_id strings for all items, in feature store row order."""
        return self._raw_item_ids

    def map_item_ids_to_indices(self, item_positions: torch.Tensor) -> torch.Tensor:
        """Convert 1-indexed item positions to 0-indexed feature-matrix row indices."""
        return torch.clamp(item_positions - 1, min=0)

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
            raw_user_ids = chunk[self.feature_cfg.interaction_user_col].astype(str).tolist()
            raw_item_ids = chunk[self.feature_cfg.interaction_item_col].astype(str).tolist()
            user_ids = self.user_encoders[self.feature_cfg.user_id_col].transform(raw_user_ids)
            item_ids = self.item_encoders[self.feature_cfg.item_id_col].transform(raw_item_ids)
            label_col = self.feature_cfg.interaction_label_col

            for start in range(0, len(user_ids), self.batch_size):
                end = start + self.batch_size
                user_ids_t = torch.from_numpy(user_ids[start:end])
                item_ids_t = torch.from_numpy(item_ids[start:end])

                user_feats = self.feature_store.get_user_features(raw_user_ids[start:end])
                item_feats = self.feature_store.get_item_features(raw_item_ids[start:end])
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
                    # item_id_pool contains raw item ID strings; sample and encode
                    neg_raw_item_ids = self._sample_negatives(len(user_ids_t))
                    neg_item_ids = self.item_encoders[self.feature_cfg.item_id_col].transform(
                        neg_raw_item_ids
                    )
                    neg_item_ids_t = torch.from_numpy(neg_item_ids)
                    neg_item_feats = self.feature_store.get_item_features(neg_raw_item_ids)
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
    user_encoders: Dict[str, Union[Tokenizer, DenseEncoder]],
    item_encoders: Dict[str, Union[Tokenizer, DenseEncoder]],
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
