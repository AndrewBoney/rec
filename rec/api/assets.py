from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from ..common.data import FeatureStore, build_feature_store
from ..common.io import load_model_from_bundle, load_model_from_wandb
from ..common.utils import FeatureConfig, load_config, read_parquet_batches
from .config import build_data_paths, build_feature_config


@dataclass
class LatestInteraction:
    item_id: str
    timestamp: Optional[str]


@dataclass
class ModelAssets:
    feature_store: FeatureStore
    feature_cfg: FeatureConfig
    user_encoders: Dict[str, Any]
    item_encoders: Dict[str, Any]
    retrieval_model: Any
    ranking_model: Any
    latest_interactions: Dict[str, LatestInteraction]


def _load_latest_interactions(
    interactions_path: str,
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    chunksize: int,
) -> Dict[str, LatestInteraction]:
    latest: Dict[str, LatestInteraction] = {}
    latest_time: Dict[str, pd.Timestamp] = {}

    for chunk in read_parquet_batches(interactions_path, batch_size=chunksize):
        if time_col and time_col in chunk.columns:
            chunk = chunk.copy()
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
            sorted_chunk = chunk.sort_values(time_col)
        else:
            sorted_chunk = chunk

        latest_rows = sorted_chunk.groupby(user_col, as_index=False).tail(1)
        for row in latest_rows.itertuples(index=False):
            user_val = str(getattr(row, user_col))
            item_val = str(getattr(row, item_col))
            ts_val = getattr(row, time_col) if time_col and time_col in chunk.columns else None
            if time_col and ts_val is not None and not pd.isna(ts_val):
                prev = latest_time.get(user_val)
                if prev is not None and ts_val <= prev:
                    continue
                latest_time[user_val] = ts_val
                latest[user_val] = LatestInteraction(item_id=item_val, timestamp=str(ts_val.date()))
            else:
                latest[user_val] = LatestInteraction(item_id=item_val, timestamp=None)

    return latest


def _encoders_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key, left_enc in left.items():
        right_enc = right[key]
        left_mapping = getattr(left_enc, "mapping", None)
        right_mapping = getattr(right_enc, "mapping", None)
        if left_mapping != right_mapping:
            return False
    return True


def load_model_assets(args, load_interactions: bool = True) -> ModelAssets:
    cfg = load_config(args.config)
    data_paths = build_data_paths(cfg)

    if args.retrieval_wandb:
        retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_wandb(
            args.retrieval_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
    else:
        retrieval_model, retrieval_meta, user_encoders, item_encoders = load_model_from_bundle(
            args.retrieval_bundle
        )

    if args.ranking_wandb:
        ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_wandb(
            args.ranking_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
    else:
        ranking_model, ranking_meta, rank_user_enc, rank_item_enc = load_model_from_bundle(
            args.ranking_bundle
        )

    if retrieval_meta.get("feature_config"):
        feature_cfg = FeatureConfig(**retrieval_meta["feature_config"])
    else:
        feature_cfg = build_feature_config(cfg, stage="retrieval")

    ranking_feature_cfg = ranking_meta.get("feature_config")
    if ranking_feature_cfg and ranking_feature_cfg != retrieval_meta.get("feature_config"):
        raise ValueError("Ranking and retrieval feature configs differ; ensure consistent training configs.")

    if not _encoders_match(rank_user_enc, user_encoders) or not _encoders_match(
        rank_item_enc, item_encoders
    ):
        raise ValueError("Ranking and retrieval encoders differ; ensure consistent artifacts.")

    feature_store = build_feature_store(data_paths, feature_cfg, user_encoders, item_encoders)

    latest_interactions: Dict[str, LatestInteraction] = {}
    if load_interactions:
        latest_interactions = _load_latest_interactions(
            data_paths.interactions_train_path,
            feature_cfg.interaction_user_col,
            feature_cfg.interaction_item_col,
            feature_cfg.interaction_time_col,
            chunksize=args.interactions_chunksize,
        )

    return ModelAssets(
        feature_store=feature_store,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        retrieval_model=retrieval_model,
        ranking_model=ranking_model,
        latest_interactions=latest_interactions,
    )
