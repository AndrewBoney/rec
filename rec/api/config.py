import argparse
from typing import Any, Dict

from ..common.data import DataPaths, FeatureConfig


def build_feature_config(cfg: Dict[str, Any], stage: str) -> FeatureConfig:
    dataset = cfg.get("dataset", {})
    base_columns = dataset.get("columns", {})
    stage_columns = cfg.get(stage, {}).get("columns", {})
    columns = {**base_columns, **stage_columns}

    return FeatureConfig(
        user_id_col=columns.get("user_id", "user_id"),
        item_id_col=columns.get("item_id", "item_id"),
        user_cat_cols=columns.get("user_cat_cols", []),
        item_cat_cols=columns.get("item_cat_cols", []),
        interaction_user_col=columns.get("interaction_user_col", columns.get("user_id", "user_id")),
        interaction_item_col=columns.get("interaction_item_col", columns.get("item_id", "item_id")),
        interaction_time_col=columns.get("interaction_time_col"),
        interaction_label_col=columns.get("label_col"),
    )


def build_data_paths(cfg: Dict[str, Any]) -> DataPaths:
    dataset = cfg.get("dataset", {})
    paths = dataset.get("paths", {})
    return DataPaths(
        users_path=paths["users"],
        items_path=paths["items"],
        interactions_train_path=paths["interactions_train"],
        interactions_val_path=paths["interactions_val"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve retrieval + ranking models via FastAPI")
    parser.add_argument("--config", default="config/movielens/movielens_1m_large.yaml")
    parser.add_argument("--retrieval-bundle", default="artifacts/retrieval")
    parser.add_argument("--ranking-bundle", default="artifacts/ranking")
    parser.add_argument("--retrieval-wandb", default=None)
    parser.add_argument("--ranking-wandb", default=None)
    parser.add_argument("--wandb-project", default="rec")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chroma-path", default="artifacts/chroma")
    parser.add_argument("--chroma-collection", default="items")
    parser.add_argument("--chroma-ranking-collection", default="items_ranking")
    parser.add_argument("--chroma-batch-size", type=int, default=2048)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--index-device", default="cpu")
    parser.add_argument("--interactions-chunksize", type=int, default=200_000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--require-auth", action="store_true")
    return parser.parse_args()
