from __future__ import annotations

import argparse
from typing import Dict

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
from ..common.model import TowerConfig
from ..common.utils import set_seed, to_device
from ..common.train_utils import (
    build_cardinalities,
    build_feature_config,
    build_paths,
    build_user_item_map,
    evaluate_ranking,
    get_device,
    load_or_build_encoders,
)
from .data import build_ranking_dataloader
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
    user_cols = [feature_cfg.user_id_col] + feature_cfg.user_cat_cols
    item_cols = [feature_cfg.item_id_col] + feature_cfg.item_cat_cols
    user_encoders, item_encoders = load_or_build_encoders(args, feature_cfg, user_cols, item_cols)
    paths = build_paths(args)

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

    device = get_device()
    model.to(device)

    train_loader, feature_store, _ = build_ranking_dataloader(
        paths=paths,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        negatives_per_pos=args.negatives_per_pos,
    )

    user_item_map = build_user_item_map(
        paths.interactions_val_path,
        feature_cfg,
        user_encoders,
        item_encoders,
        chunksize=args.chunksize,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        metrics = evaluate_ranking(
            model,
            feature_store,
            user_item_map,
            device=device,
            ks=[5, 10, 20],
        )
        metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} {metrics_str}")

    if args.save_checkpoint:
        torch.save({"state_dict": model.state_dict()}, args.save_checkpoint)
    return args.save_checkpoint


def main() -> None:
    args = parse_args()
    args = apply_config(args)
    args = ensure_dataset_args(args)
    train(args)


if __name__ == "__main__":
    main()
