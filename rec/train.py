from __future__ import annotations

import argparse
from typing import Dict

import torch
from tqdm import tqdm

from .common.model import TowerConfig
from .common.utils import set_seed, to_device
from .common.train import (
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


def _load_retrieval_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def train(args: argparse.Namespace, stage: str) -> str:
    if stage not in {"retrieval", "ranking"}:
        raise ValueError(f"Unsupported stage: {stage}")

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

    if stage == "retrieval":
        from .retrieval.data import build_retrieval_dataloader
        from .retrieval.model import TwoTowerRetrieval

        model = TwoTowerRetrieval(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            lr=args.lr,
            temperature=args.temperature,
            loss_func=getattr(args, "loss_func", None),
        )
        train_loader, feature_store = build_retrieval_dataloader(
            paths=paths,
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
        )
    else:
        from .ranking.data import build_ranking_dataloader
        from .ranking.model import TwoTowerRanking, load_retrieval_towers

        model = TwoTowerRanking(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            lr=args.lr,
            loss_func=getattr(args, "loss_func", None),
        )
        if args.init_from_retrieval:
            retrieval_state = _load_retrieval_state(args.init_from_retrieval)
            load_retrieval_towers(model, retrieval_state)

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

    device = get_device()
    model.to(device)

    run = init_wandb(args, stage)

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
        extra_metadata = {
            "user_cardinalities": user_cardinalities,
            "item_cardinalities": item_cardinalities,
        }
        if stage == "retrieval":
            extra_metadata["temperature"] = args.temperature
        else:
            extra_metadata.update(
                {
                    "negatives_per_pos": args.negatives_per_pos,
                    "init_from_retrieval": args.init_from_retrieval,
                }
            )
        bundle = save_inference_bundle(
            artifact_dir=artifact_dir,
            stage=stage,
            model_state=model.state_dict(),
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            tower_cfg=tower_cfg,
            extra_metadata=extra_metadata,
        )
        if run:
            try:
                import importlib
                wandb = importlib.import_module("wandb")
                artifact = wandb.Artifact(f"{stage}_bundle", type="model")
                artifact.add_dir(bundle["artifact_dir"])
                run.log_artifact(artifact)
            except Exception as exc:
                print(f"Failed to log wandb artifact: {exc}")
    if run:
        run.finish()
    return args.save_checkpoint
