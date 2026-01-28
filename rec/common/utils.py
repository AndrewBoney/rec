import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

@dataclass
class FeatureConfig:
    user_id_col: str
    item_id_col: str
    user_cat_cols: List[str]
    item_cat_cols: List[str]
    interaction_user_col: str
    interaction_item_col: str
    interaction_time_col: Optional[str] = None
    interaction_label_col: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_parquet_batches(path: str, batch_size: int) -> Iterable[pd.DataFrame]:
    if not (path.endswith(".parquet") or path.endswith(".pq")):
        raise ValueError(f"Only parquet inputs are supported: {path}")
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def read_table(path: str) -> pd.DataFrame:
    if not (path.endswith(".parquet") or path.endswith(".pq")):
        raise ValueError(f"Only parquet inputs are supported: {path}")
    return pd.read_parquet(path)


class CategoryEncoder:
    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}
        self.unknown_index: int = 0

    def fit(self, values: Iterable[str]) -> None:
        for v in values:
            if v not in self.mapping:
                self.mapping[v] = len(self.mapping) + 1

    def transform(self, values: Sequence[str]) -> np.ndarray:
        return np.array([self.mapping.get(v, self.unknown_index) for v in values], dtype=np.int64)

    @property
    def num_embeddings(self) -> int:
        return len(self.mapping) + 1


def build_category_maps(
    users_path: str,
    items_path: str,
    interactions_path: str,
    feature_cfg: FeatureConfig,
    chunksize: int = 200_000,
) -> Tuple[Dict[str, CategoryEncoder], Dict[str, CategoryEncoder]]:
    user_encoders: Dict[str, CategoryEncoder] = {}
    item_encoders: Dict[str, CategoryEncoder] = {}

    for col in [feature_cfg.user_id_col] + feature_cfg.user_cat_cols:
        user_encoders[col] = CategoryEncoder()
    for col in [feature_cfg.item_id_col] + feature_cfg.item_cat_cols:
        item_encoders[col] = CategoryEncoder()

    users_iter = read_parquet_batches(users_path, batch_size=chunksize)
    for chunk in users_iter:
        for col in [feature_cfg.user_id_col] + feature_cfg.user_cat_cols:
            user_encoders[col].fit(chunk[col].astype(str).tolist())

    items_iter = read_parquet_batches(items_path, batch_size=chunksize)
    for chunk in items_iter:
        for col in [feature_cfg.item_id_col] + feature_cfg.item_cat_cols:
            item_encoders[col].fit(chunk[col].astype(str).tolist())

    interactions_iter = read_parquet_batches(interactions_path, batch_size=chunksize)
    for chunk in interactions_iter:
        user_encoders[feature_cfg.user_id_col].fit(
            chunk[feature_cfg.interaction_user_col].astype(str).tolist()
        )
        item_encoders[feature_cfg.item_id_col].fit(
            chunk[feature_cfg.interaction_item_col].astype(str).tolist()
        )

    return user_encoders, item_encoders


def encode_dataframe(
    df: pd.DataFrame,
    encoders: Dict[str, CategoryEncoder],
    cols: Sequence[str],
) -> Dict[str, torch.Tensor]:
    encoded: Dict[str, torch.Tensor] = {}
    for col in cols:
        encoded[col] = torch.from_numpy(encoders[col].transform(df[col].astype(str).tolist()))
    return encoded


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def save_encoders(path: str, encoders: Dict[str, CategoryEncoder]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    payload = {k: v.mapping for k, v in encoders.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_encoders(path: str) -> Dict[str, CategoryEncoder]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    encoders: Dict[str, CategoryEncoder] = {}
    for k, mapping in payload.items():
        enc = CategoryEncoder()
        enc.mapping = {str(key): int(value) for key, value in mapping.items()}
        encoders[k] = enc
    return encoders


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
