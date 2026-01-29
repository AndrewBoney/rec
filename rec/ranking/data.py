from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader

from ..common.data import DataPaths, FeatureStore, InteractionIterableDataset, build_feature_store
from ..common.utils import CategoryEncoder, FeatureConfig, read_table


def build_ranking_dataloader(
    paths: DataPaths,
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
    batch_size: int = 1024,
    num_workers: int = 0,
    chunksize: int = 200_000,
    negatives_per_pos: int = 4,
) -> Tuple[DataLoader, FeatureStore, np.ndarray]:
    feature_store = build_feature_store(paths, feature_cfg, user_encoders, item_encoders)
    items_df = read_table(paths.items_path)
    item_ids = item_encoders[feature_cfg.item_id_col].transform(
        items_df[feature_cfg.item_id_col].astype(str).tolist()
    )
    item_id_pool = np.array(item_ids, dtype=np.int64)

    dataset = InteractionIterableDataset(
        interactions_path=paths.interactions_train_path,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        feature_store=feature_store,
        chunksize=chunksize,
        batch_size=batch_size,
        negatives_per_pos=negatives_per_pos,
        item_id_pool=item_id_pool,
        include_labels=True,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return loader, feature_store, item_id_pool
