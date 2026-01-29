import importlib
import json
import os
import torch

from dataclasses import asdict
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .data import DataPaths, FeatureStore
from .metrics import aggregate_ranking_metrics
from .utils import CategoryEncoder, FeatureConfig, build_category_maps, load_encoders, save_encoders, to_device, read_parquet_batches


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

# TODO: simplify below by implementing a method in both ranking and retrieval methods that convert user and item embs to scores 
def _score_all_items_retrieval(
    model,
    user_features: Dict[str, torch.Tensor],
    item_emb: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    user_emb = model.user_tower(user_features)
    scores = []
    for start in range(0, item_emb.size(0), batch_size):
        end = start + batch_size
        chunk = item_emb[start:end]
        chunk_scores = (user_emb @ chunk.T).squeeze(0)
        scores.append(chunk_scores)
    return torch.cat(scores, dim=0)


def _score_all_items_ranking(
    model,
    user_features: Dict[str, torch.Tensor],
    item_emb: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    user_emb = model.user_tower(user_features)
    scores = []
    for start in range(0, item_emb.size(0), batch_size):
        end = start + batch_size
        chunk = item_emb[start:end]
        expanded_user = user_emb.expand(chunk.size(0), -1)
        joint = torch.cat(
            [expanded_user, chunk, expanded_user * chunk, torch.abs(expanded_user - chunk)], dim=-1
        )
        chunk_scores = model.scorer(joint).squeeze(-1)
        scores.append(chunk_scores)
    return torch.cat(scores, dim=0)


def evaluate_retrieval(
    model,
    feature_store: FeatureStore,
    user_item_map: Dict[int, List[int]],
    device: torch.device,
    ks: Iterable[int],
    batch_size: int = 4096,
) -> Dict[str, float]:
    model.eval()
    item_features = feature_store.get_all_item_features()
    item_features = to_device(item_features, device)
    with torch.no_grad():
        item_emb = model.item_tower(item_features)
    user_ids = list(user_item_map.keys())
    max_k = max(ks) if ks else 0
    topk_indices = []
    relevant_indices = []
    with torch.no_grad():
        for uid in user_ids:
            user_feats = feature_store.get_user_features(torch.tensor([uid], dtype=torch.long))
            user_feats = to_device(user_feats, device)
            scores = _score_all_items_retrieval(model, user_feats, item_emb, batch_size=batch_size)
            topk = torch.topk(scores, max_k).indices
            topk_indices.append(topk.cpu())
            rel_ids = torch.tensor(user_item_map[uid], dtype=torch.long)
            rel_indices = feature_store.map_item_ids_to_indices(rel_ids)
            relevant_indices.append(rel_indices)
    if not topk_indices:
        return {}
    topk_tensor = torch.stack(topk_indices, dim=0)
    metrics = aggregate_ranking_metrics(topk_tensor, relevant_indices, ks)
    return metrics


def evaluate_ranking(
    model,
    feature_store: FeatureStore,
    user_item_map: Dict[int, List[int]],
    device: torch.device,
    ks: Iterable[int],
    batch_size: int = 4096,
) -> Dict[str, float]:
    model.eval()
    item_features = feature_store.get_all_item_features()
    item_features = to_device(item_features, device)
    with torch.no_grad():
        item_emb = model.item_tower(item_features)
    user_ids = list(user_item_map.keys())
    max_k = max(ks) if ks else 0
    topk_indices = []
    relevant_indices = []
    with torch.no_grad():
        for uid in user_ids:
            user_feats = feature_store.get_user_features(torch.tensor([uid], dtype=torch.long))
            user_feats = to_device(user_feats, device)
            scores = _score_all_items_ranking(model, user_feats, item_emb, batch_size=batch_size)
            topk = torch.topk(scores, max_k).indices
            topk_indices.append(topk.cpu())
            rel_ids = torch.tensor(user_item_map[uid], dtype=torch.long)
            rel_indices = feature_store.map_item_ids_to_indices(rel_ids)
            relevant_indices.append(rel_indices)
    if not topk_indices:
        return {}
    topk_tensor = torch.stack(topk_indices, dim=0)
    metrics = aggregate_ranking_metrics(topk_tensor, relevant_indices, ks)
    return metrics
