from __future__ import annotations

import argparse
import os
from typing import Dict

import lightning.pytorch as lit
import torch

from ..common.config import (
    add_ranking_args,
    apply_dataset_config,
    apply_shared_config,
    apply_stage_config,
    build_base_parser,
    ensure_dataset_args,
    load_yaml_config,
)
from ..common.data import DataPaths
from ..common.model import TowerConfig
from ..common.utils import CategoryEncoder, FeatureConfig, build_category_maps, load_encoders, save_encoders, set_seed
from .data import RankingDataModule
from .model import TwoTowerRanking, load_retrieval_towers


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Two-tower ranking training")
    add_ranking_args(parser)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)
    if not cfg:
        return args
    args = apply_dataset_config(args, cfg)
    args = apply_shared_config(args, cfg)
    args = apply_stage_config(args, cfg, "ranking")
    return args


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


def _load_retrieval_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


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

    model = TwoTowerRanking(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_cfg,
        lr=args.lr,
    )

    if args.init_from_retrieval:
        retrieval_state = _load_retrieval_state(args.init_from_retrieval)
        load_retrieval_towers(model, retrieval_state)

    datamodule = RankingDataModule(
        paths=paths,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        negatives_per_pos=args.negatives_per_pos,
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
    args = apply_config(args)
    args = ensure_dataset_args(args)
    train(args)


if __name__ == "__main__":
    main()
