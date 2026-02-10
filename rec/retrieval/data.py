from __future__ import annotations

from typing import Dict, Tuple, Union
from torch.utils.data import DataLoader

from ..common.data import CategoryEncoder, DataPaths, DenseEncoder, FeatureStore, InteractionIterableDataset, build_feature_store, FeatureConfig


def build_retrieval_dataloader(
    paths: DataPaths,
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    item_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    batch_size: int = 1024,
    num_workers: int = 0,
    chunksize: int = 200_000,
) -> Tuple[DataLoader, FeatureStore]:
    feature_store = build_feature_store(
        paths,
        feature_cfg,
        user_encoders,
        item_encoders,
    )
    dataset = InteractionIterableDataset(
        interactions_path=paths.interactions_train_path,
        feature_store=feature_store,
        chunksize=chunksize,
        batch_size=batch_size,
        negatives_per_pos=0,
        include_labels=False,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return loader, feature_store
