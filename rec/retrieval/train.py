from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from ..common.config import (
    add_retrieval_args,
    apply_config,
    build_base_parser,
    ensure_dataset_args,
    load_yaml_config,
)
from ..common.model import TowerConfig
from ..common.utils import set_seed, to_device
from ..common.train import (
    build_cardinalities,
    build_feature_config,
    build_paths,
    build_user_item_map,
    evaluate,
    get_device,
    init_wandb,
    load_or_build_encoders,
    save_inference_bundle,
)
from .data import build_retrieval_dataloader
from .model import TwoTowerRetrieval


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Two-tower retrieval training")
    add_retrieval_args(parser)
    return parser.parse_args()


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

    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_cfg,
        lr=args.lr,
        temperature=args.temperature,
    )
    device = get_device()
    model.to(device)

    run = init_wandb(args, "retrieval")

    train_loader, feature_store = build_retrieval_dataloader(
        paths=paths,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )

    user_item_map = build_user_item_map(
        paths.interactions_val_path,
        feature_cfg,
        user_encoders,
        item_encoders,
        chunksize=args.chunksize,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", total=total_batches)
        for batch in progress:
            batch = to_device(batch, device)
            loss = model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.log_steps > 0 and global_step % args.log_steps == 0:
                if run:
                    run.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_step)

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(
                    model,
                    feature_store,
                    user_item_map,
                    device=device,
                    ks=[5, 10, 20],
                )
                if run and metrics:
                    run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                model.train()

        avg_loss = total_loss / max(1, steps)
        metrics = evaluate(
            model,
            feature_store,
            user_item_map,  
            device=device,
            ks=[5, 10, 20],
        )
        metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} {metrics_str}")
        if run:
            run.log({"train/epoch_loss": avg_loss}, step=global_step)
            if metrics:
                run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

    if args.save_checkpoint:
        torch.save({"state_dict": model.state_dict()}, args.save_checkpoint)

    artifact_dir = args.artifact_dir or (f"{args.save_checkpoint}.artifact" if args.save_checkpoint else None)
    if artifact_dir:
        bundle = save_inference_bundle(
            artifact_dir=artifact_dir,
            stage="retrieval",
            model_state=model.state_dict(),
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            tower_cfg=tower_cfg,
            extra_metadata={
                "user_cardinalities": user_cardinalities,
                "item_cardinalities": item_cardinalities,
                "temperature": args.temperature,
            },
        )
        if run:
            try:
                import importlib
                wandb = importlib.import_module("wandb")
                artifact = wandb.Artifact("retrieval_bundle", type="model")
                artifact.add_dir(bundle["artifact_dir"])
                run.log_artifact(artifact)
            except Exception as exc:
                print(f"Failed to log wandb artifact: {exc}")
    if run:
        run.finish()
    return args.save_checkpoint


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if cfg:
        args = apply_config(args, cfg, stage="retrieval")
    args = ensure_dataset_args(args)
    train(args)


if __name__ == "__main__":
    main()
