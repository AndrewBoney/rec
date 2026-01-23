from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
import lightning.pytorch as lit
from torch.utils.data import DataLoader, IterableDataset

from ..common.data import DataPaths, FeatureStore
from ..common.utils import CategoryEncoder, FeatureConfig, read_csv_chunks


class RetrievalIterableDataset(IterableDataset):
    def __init__(
        self,
        interactions_path: str,
        feature_cfg: FeatureConfig,
        user_encoders: Dict[str, CategoryEncoder],
        item_encoders: Dict[str, CategoryEncoder],
        feature_store: FeatureStore,
        chunksize: int = 200_000,
        batch_size: int = 2048,
    ) -> None:
        super().__init__()
        self.interactions_path = interactions_path
        self.feature_cfg = feature_cfg
        self.user_encoders = user_encoders
        self.item_encoders = item_encoders
        self.feature_store = feature_store
        self.chunksize = chunksize
        self.batch_size = batch_size

    def __iter__(self):
        for chunk in read_csv_chunks(self.interactions_path, self.chunksize):
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
                yield batch


class RetrievalDataModule(lit.LightningDataModule):
    def __init__(
        self,
        paths: DataPaths,
        feature_cfg: FeatureConfig,
        user_encoders: Dict[str, CategoryEncoder],
        item_encoders: Dict[str, CategoryEncoder],
        batch_size: int = 1024,
        num_workers: int = 0,
        chunksize: int = 200_000,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.user_encoders = user_encoders
        self.item_encoders = item_encoders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunksize = chunksize

        self.feature_store: Optional[FeatureStore] = None

    def setup(self, stage: Optional[str] = None) -> None:
        users_df = pd.read_csv(self.paths.users_path)
        items_df = pd.read_csv(self.paths.items_path)
        self.feature_store = FeatureStore(
            users_df, items_df, self.user_encoders, self.item_encoders, self.feature_cfg
        )

    def train_dataloader(self) -> DataLoader:
        if self.feature_store is None:
            raise RuntimeError("setup() must be called before train_dataloader")
        dataset = RetrievalIterableDataset(
            interactions_path=self.paths.interactions_path,
            feature_cfg=self.feature_cfg,
            user_encoders=self.user_encoders,
            item_encoders=self.item_encoders,
            feature_store=self.feature_store,
            chunksize=self.chunksize,
            batch_size=self.batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
        )
