import argparse
import importlib
import json
import os
import torch

from dataclasses import asdict
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

from .data import DataPaths, FeatureStore
from .metrics import aggregate_ranking_metrics
from .model import TowerConfig
from .utils import CategoryEncoder, FeatureConfig, build_category_maps, load_encoders, read_parquet_batches, save_encoders, set_seed, to_device


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_feature_config(args) -> FeatureConfig:
    return FeatureConfig(
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        user_cat_cols=args.user_cat_cols,
        item_cat_cols=args.item_cat_cols,
        interaction_user_col=args.interaction_user_col,
        interaction_item_col=args.interaction_item_col,
        interaction_label_col=getattr(args, "interaction_label_col", None),
    )


def build_cardinalities(encoders: Dict[str, CategoryEncoder], cols: Iterable[str]) -> Dict[str, int]:
    return {col: encoders[col].num_embeddings for col in cols}


def load_or_build_encoders(
    args,
    feature_cfg: FeatureConfig,
    user_cols: List[str],
    item_cols: List[str],
) -> Tuple[Dict[str, CategoryEncoder], Dict[str, CategoryEncoder]]:
    encoder_cache_path = args.encoder_cache
    if os.path.exists(encoder_cache_path + ".users") and os.path.exists(encoder_cache_path + ".items"):
        user_encoders = load_encoders(encoder_cache_path + ".users")
        item_encoders = load_encoders(encoder_cache_path + ".items")
        missing_user = [col for col in user_cols if col not in user_encoders]
        missing_item = [col for col in item_cols if col not in item_encoders]
        if missing_user or missing_item:
            user_encoders, item_encoders = build_category_maps(
                args.users,
                args.items,
                args.interactions_train,
                feature_cfg,
                chunksize=args.chunksize,
            )
            save_encoders(encoder_cache_path + ".users", user_encoders)
            save_encoders(encoder_cache_path + ".items", item_encoders)
    else:
        user_encoders, item_encoders = build_category_maps(
            args.users,
            args.items,
            args.interactions_train,
            feature_cfg,
            chunksize=args.chunksize,
        )
        save_encoders(encoder_cache_path + ".users", user_encoders)
        save_encoders(encoder_cache_path + ".items", item_encoders)
    return user_encoders, item_encoders


def build_paths(args) -> DataPaths:
    return DataPaths(
        users_path=args.users,
        items_path=args.items,
        interactions_train_path=args.interactions_train,
        interactions_val_path=args.interactions_val,
    )


def build_user_item_map(
    interactions_path: str,
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
    chunksize: int = 200_000,
) -> Dict[int, List[int]]:
    user_to_items: Dict[int, List[int]] = {}
    for chunk in read_parquet_batches(interactions_path, chunksize):
        user_ids = user_encoders[feature_cfg.user_id_col].transform(
            chunk[feature_cfg.interaction_user_col].astype(str).tolist()
        )
        item_ids = item_encoders[feature_cfg.item_id_col].transform(
            chunk[feature_cfg.interaction_item_col].astype(str).tolist()
        )
        for uid, iid in zip(user_ids, item_ids):
            bucket = user_to_items.setdefault(int(uid), [])
            bucket.append(int(iid))
    return user_to_items


def init_wandb(args, stage: str) -> Optional[Any]:
    load_dotenv(override=False)
    use_wandb = bool(getattr(args, "use_wandb", False)) or bool(os.getenv("WANDB_API_KEY"))
    if not use_wandb:
        return None
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found in environment; skipping wandb logging.")
        return None
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Please install it to enable logging.") from exc
    wandb.login()
    run = wandb.init(
        project=getattr(args, "wandb_project", "rec"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        config={k: v for k, v in vars(args).items() if k != "config"},
        tags=[stage],
    )
    return run


def save_inference_bundle(
    artifact_dir: str,
    stage: str,
    model_state: Dict[str, torch.Tensor],
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
    tower_cfg,
    extra_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "model.pt",
) -> Dict[str, str]:
    os.makedirs(artifact_dir, exist_ok=True)
    user_enc_path = os.path.join(artifact_dir, "encoders.users.json")
    item_enc_path = os.path.join(artifact_dir, "encoders.items.json")
    save_encoders(user_enc_path, user_encoders)
    save_encoders(item_enc_path, item_encoders)

    ckpt_path = os.path.join(artifact_dir, checkpoint_name)
    torch.save({"state_dict": model_state}, ckpt_path)

    metadata = {
        "stage": stage,
        "feature_config": asdict(feature_cfg),
        "tower_config": asdict(tower_cfg),
        "checkpoint": checkpoint_name,
        "encoders": {
            "users": os.path.basename(user_enc_path),
            "items": os.path.basename(item_enc_path),
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {
        "artifact_dir": artifact_dir,
        "metadata": metadata_path,
        "checkpoint": ckpt_path,
        "encoders_users": user_enc_path,
        "encoders_items": item_enc_path,
    }


def load_inference_bundle(artifact_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    user_encoders = load_encoders(os.path.join(artifact_dir, metadata["encoders"]["users"]))
    item_encoders = load_encoders(os.path.join(artifact_dir, metadata["encoders"]["items"]))
    checkpoint = os.path.join(artifact_dir, metadata["checkpoint"])
    return {
        "metadata": metadata,
        "checkpoint": checkpoint,
        "user_encoders": user_encoders,
        "item_encoders": item_encoders,
    }


def _load_retrieval_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")

# TODO: add batching to prevent risk of OOM
def evaluate(
    model,
    feature_store: FeatureStore,
    user_item_map: Dict[int, List[int]],
    device: torch.device,
    ks: Iterable[int],
) -> Dict[str, float]:
    model.eval()
    item_features = feature_store.get_all_item_features()
    item_features = to_device(item_features, device)
    with torch.no_grad():
        item_emb = model.item_tower(item_features)
    user_ids = list(user_item_map.keys())
    ks_list = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_list:
        return {}
    num_items = item_emb.size(0)
    ks_list = [k for k in ks_list if k <= num_items]
    if not ks_list:
        return {}
    max_k = max(ks_list)
    topk_indices = []
    relevant_indices = []
    with torch.no_grad():
        for uid in user_ids:
            user_feats = feature_store.get_user_features(torch.tensor([uid], dtype=torch.long))
            user_feats = to_device(user_feats, device)
            user_emb = model.user_tower(user_feats)
            scores = model.score_all(user_emb, item_emb)
            scores = scores.squeeze(0)
            topk = torch.topk(scores, max_k).indices
            topk_indices.append(topk.cpu())
            rel_ids = torch.tensor(user_item_map[uid], dtype=torch.long)
            rel_indices = feature_store.map_item_ids_to_indices(rel_ids)
            relevant_indices.append(rel_indices)
    if not topk_indices:
        return {}
    topk_tensor = torch.stack(topk_indices, dim=0)
    metrics = aggregate_ranking_metrics(topk_tensor, relevant_indices, ks_list)
    return metrics


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
        from ..retrieval.data import build_retrieval_dataloader
        from ..retrieval.model import TwoTowerRetrieval

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
        from ..ranking.data import build_ranking_dataloader
        from ..ranking.model import TwoTowerRanking, load_retrieval_towers

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
                wandb = importlib.import_module("wandb")
                artifact = wandb.Artifact(f"{stage}_bundle", type="model")
                artifact.add_dir(bundle["artifact_dir"])
                run.log_artifact(artifact)
            except Exception as exc:
                print(f"Failed to log wandb artifact: {exc}")
    if run:
        run.finish()
    return args.save_checkpoint