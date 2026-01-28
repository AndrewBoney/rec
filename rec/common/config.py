from __future__ import annotations

import argparse
from typing import Any, Dict

from .utils import load_config


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--users", required=False)
    parser.add_argument("--items", required=False)
    parser.add_argument("--interactions-train", required=False)
    parser.add_argument("--interactions-val", required=False)
    parser.add_argument("--user-id-col", default="customer_id")
    parser.add_argument("--item-id-col", default="article_id")
    parser.add_argument("--user-cat-cols", nargs="*", default=["FN", "Active", "club_member_status", "fashion_news_frequency", "age"])
    parser.add_argument("--item-cat-cols", nargs="*", default=["product_type_no", "graphical_appearance_no", "colour_group_code", "section_no", "garment_group_no"])
    parser.add_argument("--interaction-user-col", default="customer_id")
    parser.add_argument("--interaction-item-col", default="article_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-cache", default="encoders.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def add_retrieval_args(parser: argparse.ArgumentParser, include_checkpoint: bool = True) -> argparse.ArgumentParser:
    parser.add_argument("--temperature", type=float, default=0.05)
    if include_checkpoint:
        parser.add_argument("--save-checkpoint", default="retrieval.ckpt")
    return parser


def add_ranking_args(parser: argparse.ArgumentParser, include_checkpoint: bool = True) -> argparse.ArgumentParser:
    parser.add_argument("--negatives-per-pos", type=int, default=4)
    parser.add_argument("--init-from-retrieval", default=None)
    if include_checkpoint:
        parser.add_argument("--save-checkpoint", default="ranking.ckpt")
    return parser


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    return load_config(path)


def apply_dataset_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    dataset = cfg.get("dataset", {})
    paths = dataset.get("paths", {})
    columns = dataset.get("columns", {})

    args.users = paths.get("users", args.users)
    args.items = paths.get("items", args.items)
    args.interactions_train = paths.get("interactions_train", args.interactions_train)
    args.interactions_val = paths.get("interactions_val", args.interactions_val)

    args.user_id_col = columns.get("user_id", args.user_id_col)
    args.item_id_col = columns.get("item_id", args.item_id_col)
    args.user_cat_cols = columns.get("user_cat_cols", args.user_cat_cols)
    args.item_cat_cols = columns.get("item_cat_cols", args.item_cat_cols)
    args.interaction_user_col = columns.get("interaction_user_col", args.interaction_user_col)
    args.interaction_item_col = columns.get("interaction_item_col", args.interaction_item_col)

    return args


def apply_shared_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    shared = cfg.get("shared", {})
    args.seed = shared.get("seed", args.seed)
    args.encoder_cache = shared.get("encoder_cache", args.encoder_cache)
    return args


def apply_stage_config(args: argparse.Namespace, cfg: Dict[str, Any], stage: str) -> argparse.Namespace:
    stage_cfg = cfg.get(stage, {})
    model_cfg = stage_cfg.get("model", {})
    training_cfg = stage_cfg.get("training", {})

    args.embedding_dim = model_cfg.get("embedding_dim", args.embedding_dim)
    args.hidden_dims = model_cfg.get("hidden_dims", args.hidden_dims)
    args.dropout = model_cfg.get("dropout", args.dropout)

    args.batch_size = training_cfg.get("batch_size", args.batch_size)
    args.num_workers = training_cfg.get("num_workers", args.num_workers)
    args.chunksize = training_cfg.get("chunksize", args.chunksize)
    args.max_epochs = training_cfg.get("max_epochs", args.max_epochs)
    args.lr = training_cfg.get("lr", args.lr)

    if "seed" in training_cfg:
        args.seed = training_cfg["seed"]
    if "encoder_cache" in training_cfg:
        args.encoder_cache = training_cfg["encoder_cache"]
    if "temperature" in training_cfg:
        args.temperature = training_cfg["temperature"]
    if "negatives_per_pos" in training_cfg:
        args.negatives_per_pos = training_cfg["negatives_per_pos"]
    if "checkpoint" in training_cfg:
        args.save_checkpoint = training_cfg["checkpoint"]

    return args


def ensure_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    missing = [name for name in ("users", "items") if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required dataset args: {', '.join(missing)}")
    if not args.interactions_train:
        raise ValueError("Missing required dataset args: interactions_train")
    if not args.interactions_val:
        raise ValueError("Missing required dataset args: interactions_val")
    return args
