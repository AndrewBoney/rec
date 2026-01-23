from __future__ import annotations

import argparse
import os
from typing import Dict

import lightning.pytorch as lit
import torch

from ..common.data import DataPaths
from ..common.model import TowerConfig
from ..common.utils import CategoryEncoder, FeatureConfig, build_category_maps, load_encoders, save_encoders, set_seed
from .data import RetrievalDataModule
from .model import TwoTowerRetrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-tower retrieval training")
    parser.add_argument("--users", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--user-id-col", default="customer_id")
    parser.add_argument("--item-id-col", default="article_id")
    parser.add_argument("--user-cat-cols", nargs="*", default=["FN", "Active", "club_member_status", "fashion_news_frequency", "age"])
    parser.add_argument("--item-cat-cols", nargs="*", default=["product_type_no", "graphical_appearance_no", "colour_group_code", "section_no", "garment_group_no"])
    parser.add_argument("--interaction-user-col", default="customer_id")
    parser.add_argument("--interaction-item-col", default="article_id")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--encoder-cache", default="encoders.json")
    parser.add_argument("--save-checkpoint", default="retrieval.ckpt")
    return parser.parse_args()


def build_feature_config(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        user_cat_cols=args.user_cat_cols,
        item_cat_cols=args.item_cat_cols,
        interaction_user_col=args.interaction_user_col,
        interaction_item_col=args.interaction_item_col,
    )


def build_cardinalities(encoders: Dict[str, CategoryEncoder], cols) -> Dict[str, int]:
    return {col: encoders[col].num_embeddings for col in cols}


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)

    feature_cfg = build_feature_config(args)
    encoder_cache_path = args.encoder_cache

    if os.path.exists(encoder_cache_path + ".users") and os.path.exists(encoder_cache_path + ".items"):
        user_encoders = load_encoders(encoder_cache_path + ".users")
        item_encoders = load_encoders(encoder_cache_path + ".items")
    else:
        user_encoders, item_encoders = build_category_maps(
            args.users,
            args.items,
            args.interactions,
            feature_cfg,
            chunksize=args.chunksize,
        )
        save_encoders(encoder_cache_path + ".users", user_encoders)
        save_encoders(encoder_cache_path + ".items", item_encoders)

    paths = DataPaths(args.users, args.items, args.interactions)

    user_cols = [feature_cfg.user_id_col] + feature_cfg.user_cat_cols
    item_cols = [feature_cfg.item_id_col] + feature_cfg.item_cat_cols
    user_cardinalities = build_cardinalities(user_encoders, user_cols)
    item_cardinalities = build_cardinalities(item_encoders, item_cols)

    tower_cfg = TowerConfig(
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )

    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_cfg,
        lr=args.lr,
        temperature=args.temperature,
    )

    datamodule = RetrievalDataModule(
        paths=paths,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )

    trainer = lit.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=datamodule)

    if args.save_checkpoint:
        trainer.save_checkpoint(args.save_checkpoint)
    return args.save_checkpoint


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
