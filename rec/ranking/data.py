from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader

from ..common.data import DataPaths, FeatureStore, InteractionIterableDataset, build_feature_store
from ..common.utils import CategoryEncoder, FeatureConfig


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
    item_id_pool = feature_store.get_all_item_ids().cpu().numpy().astype(np.int64, copy=False)

    dataset = InteractionIterableDataset(
        interactions_path=paths.interactions_train_path,
        feature_store=feature_store,
        chunksize=chunksize,
        batch_size=batch_size,
        negatives_per_pos=negatives_per_pos,
        item_id_pool=item_id_pool,
        include_labels=True,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return loader, feature_store, item_id_pool
