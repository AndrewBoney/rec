import argparse
import importlib
import os
import torch
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

from .data import DataPaths, FeatureStore, InteractionIterableDataset
from .model import TowerConfig
from .utils import CategoryEncoder, FeatureConfig, build_category_maps, load_encoders, read_parquet_batches, save_encoders, set_seed, to_device
from .io import load_model_bundle, save_model_bundle
from ..ranking.metrics import aggregate_pointwise_metrics
from ..retrieval.metrics import aggregate_retrieval_metrics


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
    users_cache_path = encoder_cache_path + ".users"
    items_cache_path = encoder_cache_path + ".items"

    def _build_and_save() -> Tuple[Dict[str, CategoryEncoder], Dict[str, CategoryEncoder]]:
        user_encs, item_encs = build_category_maps(
            args.users,
            args.items,
            args.interactions_train,
            feature_cfg,
            chunksize=args.chunksize,
        )
        save_encoders(users_cache_path, user_encs)
        save_encoders(items_cache_path, item_encs)
        return user_encs, item_encs

    if os.path.exists(users_cache_path) and os.path.exists(items_cache_path):
        user_encoders = load_encoders(users_cache_path)
        item_encoders = load_encoders(items_cache_path)
        missing_user = [col for col in user_cols if col not in user_encoders]
        missing_item = [col for col in item_cols if col not in item_encoders]
        if not missing_user and not missing_item:
            return user_encoders, item_encoders

    return _build_and_save()


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
    user_to_items: Dict[int, set[int]] = {}
    for chunk in read_parquet_batches(interactions_path, chunksize):
        user_ids = user_encoders[feature_cfg.user_id_col].transform(
            chunk[feature_cfg.interaction_user_col].astype(str).tolist()
        )
        item_ids = item_encoders[feature_cfg.item_id_col].transform(
            chunk[feature_cfg.interaction_item_col].astype(str).tolist()
        )
        for uid, iid in zip(user_ids, item_ids):
            bucket = user_to_items.setdefault(int(uid), set())
            bucket.add(int(iid))
    return {uid: sorted(items) for uid, items in user_to_items.items()}


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
        job_type=f"{stage}_train",
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
) -> Dict[str, Any]:
    return save_model_bundle(
        output_dir=artifact_dir,
        stage=stage,
        model_state=model_state,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        tower_cfg=tower_cfg,
        extra_metadata=extra_metadata,
        checkpoint_name=checkpoint_name,
    )


def load_inference_bundle(artifact_dir: str) -> Dict[str, Any]:
    return load_model_bundle(artifact_dir)


def _load_retrieval_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")

def _map_user_items_to_indices(
    user_item_map: Dict[int, List[int]],
    feature_store: FeatureStore,
) -> Dict[int, torch.Tensor]:
    mapped: Dict[int, torch.Tensor] = {}
    for uid, items in user_item_map.items():
        if not items:
            continue
        item_ids = torch.tensor(items, dtype=torch.long)
        mapped[uid] = feature_store.map_item_ids_to_indices(item_ids)
    return mapped


def evaluate_retrieval(
    model,
    feature_store: FeatureStore,
    user_item_map: Dict[int, List[int]],
    seen_user_item_map: Dict[int, List[int]],
    device: torch.device,
    ks: Iterable[int],
) -> Dict[str, float]:
    model.eval()
    item_features = feature_store.get_all_item_features()
    item_features = to_device(item_features, device)
    with torch.no_grad():
        item_emb = model.item_tower(item_features)

    ks_list = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_list:
        return {}
    num_items = item_emb.size(0)
    ks_list = [k for k in ks_list if k <= num_items]
    if not ks_list:
        return {}
    max_k = max(ks_list)

    relevant_indices_map = _map_user_items_to_indices(user_item_map, feature_store)
    seen_indices_map = _map_user_items_to_indices(seen_user_item_map, feature_store)

    topk_indices = []
    relevant_indices = []
    with torch.no_grad():
        for uid, rel_indices in relevant_indices_map.items():
            user_feats = feature_store.get_user_features(torch.tensor([uid], dtype=torch.long))
            user_feats = to_device(user_feats, device)
            user_emb = model.user_tower(user_feats)
            scores = model.score_all(user_emb, item_emb).squeeze(0)

            seen_indices = seen_indices_map.get(uid)
            if seen_indices is not None and seen_indices.numel() > 0:
                scores[seen_indices.to(scores.device)] = -torch.inf

            topk = torch.topk(scores, max_k).indices
            topk_indices.append(topk.cpu())
            relevant_indices.append(rel_indices)

    if not topk_indices:
        return {}
    topk_tensor = torch.stack(topk_indices, dim=0)
    return aggregate_retrieval_metrics(topk_tensor, relevant_indices, ks_list)


def evaluate_ranking(
    model,
    interactions_path: str,
    feature_store: FeatureStore,
    device: torch.device,
    chunksize: int,
    batch_size: int,
) -> Dict[str, float]:
    dataset = InteractionIterableDataset(
        interactions_path=interactions_path,
        feature_store=feature_store,
        chunksize=chunksize,
        batch_size=batch_size,
        negatives_per_pos=0,
        include_labels=True,
    )

    model.eval()
    logits_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataset:
            batch = to_device(batch, device)
            labels = batch.get("label")
            if labels is None:
                continue

            user_features = {k[len("user_"):]: v for k, v in batch.items() if k.startswith("user_")}
            item_features = {k[len("item_"):]: v for k, v in batch.items() if k.startswith("item_")}
            logits = model.forward(user_features, item_features)

            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(labels.detach().cpu())

    if not logits_chunks:
        return {}

    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(label_chunks, dim=0)
    return aggregate_pointwise_metrics(logits_all, labels_all, model.loss_func)


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
            scorer_hidden_dims=args.scorer_hidden_dims,
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
    if run and getattr(args, "wandb_log_datasets", False):
        try:
            wandb = importlib.import_module("wandb")
            dataset_artifact = wandb.Artifact(f"{stage}_dataset", type="dataset")
            for path in (args.users, args.items, args.interactions_train, args.interactions_val):
                if path:
                    dataset_artifact.add_file(path)
            run.log_artifact(dataset_artifact, aliases=["latest"])
        except Exception as exc:
            print(f"Failed to log wandb dataset artifact: {exc}")

    val_user_item_map = build_user_item_map(
        paths.interactions_val_path,
        feature_cfg,
        user_encoders,
        item_encoders,
        chunksize=args.chunksize,
    )
    train_user_item_map = build_user_item_map(
        paths.interactions_train_path,
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
                if stage == "retrieval":
                    metrics = evaluate_retrieval(
                        model,
                        feature_store,
                        val_user_item_map,
                        train_user_item_map,
                        device=device,
                        ks=[5, 10, 20],
                    )
                else:
                    metrics = {}
                    metrics.update(
                        evaluate_ranking(
                            model,
                            paths.interactions_val_path,
                            feature_store,
                            device=device,
                            chunksize=args.chunksize,
                            batch_size=args.batch_size,
                        )
                    )
                    metrics.update(
                        evaluate_retrieval(
                            model,
                            feature_store,
                            val_user_item_map,
                            train_user_item_map,
                            device=device,
                            ks=[5, 10, 20],
                        )
                    )
                if run and metrics:
                    run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                model.train()

        avg_loss = total_loss / max(1, steps)
        if stage == "retrieval":
            metrics = evaluate_retrieval(
                model,
                feature_store,
                val_user_item_map,
                train_user_item_map,
                device=device,
                ks=[5, 10, 20],
            )
        else:
            metrics = {}
            metrics.update(
                evaluate_ranking(
                    model,
                    paths.interactions_val_path,
                    feature_store,
                    device=device,
                    chunksize=args.chunksize,
                    batch_size=args.batch_size,
                )
            )
            metrics.update(
                evaluate_retrieval(
                    model,
                    feature_store,
                    val_user_item_map,
                    train_user_item_map,
                    device=device,
                    ks=[5, 10, 20],
                )
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
            "loss_func": args.loss_func,
        }
        if stage == "retrieval":
            extra_metadata["temperature"] = args.temperature
        else:
            extra_metadata.update(
                {
                    "negatives_per_pos": args.negatives_per_pos,
                    "init_from_retrieval": args.init_from_retrieval,
                    "scorer_hidden_dims": args.scorer_hidden_dims,
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
                artifact_name = getattr(args, "wandb_artifact_name", None) or f"{stage}_bundle"
                artifact_aliases = getattr(args, "wandb_artifact_aliases", None) or []
                artifact = wandb.Artifact(artifact_name, type="model", metadata=bundle.get("metadata_dict"))
                artifact.add_dir(bundle["artifact_dir"])
                if artifact_aliases:
                    run.log_artifact(artifact, aliases=artifact_aliases)
                else:
                    run.log_artifact(artifact)
            except Exception as exc:
                print(f"Failed to log wandb artifact: {exc}")
    if run:
        run.finish()
    return args.save_checkpoint