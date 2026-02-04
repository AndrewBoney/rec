from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .model import TowerConfig
from .utils import CategoryEncoder, FeatureConfig, load_encoders, save_encoders
from ..ranking.model import TwoTowerRanking
from ..retrieval.model import TwoTowerRetrieval


def save_model_bundle(
    *,
    output_dir: str | Path,
    stage: str,
    model_state: Dict[str, torch.Tensor],
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
    tower_cfg,
    extra_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "model.pt",
) -> Dict[str, Any]:
    """Persist a complete model bundle for later inference or artifact logging."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    user_enc_path = output_path / "encoders.users.json"
    item_enc_path = output_path / "encoders.items.json"
    save_encoders(str(user_enc_path), user_encoders)
    save_encoders(str(item_enc_path), item_encoders)

    ckpt_path = output_path / checkpoint_name
    torch.save({"state_dict": model_state}, ckpt_path)

    metadata = {
        "stage": stage,
        "feature_config": asdict(feature_cfg),
        "tower_config": asdict(tower_cfg),
        "checkpoint": checkpoint_name,
        "encoders": {
            "users": user_enc_path.name,
            "items": item_enc_path.name,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = output_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "artifact_dir": str(output_path),
        "metadata": str(metadata_path),
        "metadata_dict": metadata,
        "checkpoint": str(ckpt_path),
        "encoders_users": str(user_enc_path),
        "encoders_items": str(item_enc_path),
    }


def load_model_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """Load a previously saved model bundle from disk."""
    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    user_encoders = load_encoders(str(bundle_path / metadata["encoders"]["users"]))
    item_encoders = load_encoders(str(bundle_path / metadata["encoders"]["items"]))
    checkpoint = bundle_path / metadata["checkpoint"]

    return {
        "metadata": metadata,
        "checkpoint": str(checkpoint),
        "user_encoders": user_encoders,
        "item_encoders": item_encoders,
    }


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_model_from_bundle(
    bundle_dir: str | Path,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load a model + encoders from a local bundle directory."""
    bundle = load_model_bundle(bundle_dir)
    metadata = bundle["metadata"]
    tower_cfg = TowerConfig(**metadata["tower_config"])

    user_cardinalities = metadata.get("user_cardinalities")
    item_cardinalities = metadata.get("item_cardinalities")
    if user_cardinalities is None or item_cardinalities is None:
        raise ValueError("Bundle metadata is missing user/item cardinalities.")

    stage = metadata.get("stage")
    if stage == "retrieval":
        model = TwoTowerRetrieval(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            temperature=metadata.get("temperature", 0.05),
            loss_func=metadata.get("loss_func"),
        )
    elif stage == "ranking":
        model = TwoTowerRanking(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            scorer_hidden_dims=metadata.get("scorer_hidden_dims"),
            loss_func=metadata.get("loss_func"),
        )
    else:
        raise ValueError(f"Unsupported stage in metadata: {stage}")

    state_dict = _load_state_dict(bundle["checkpoint"])
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata, bundle["user_encoders"], bundle["item_encoders"]


def load_model_from_wandb(
    artifact_ref: str,
    *,
    project: str = "rec",
    entity: Optional[str] = None,
    job_type: str = "inference",
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Download a W&B model artifact and load it for inference."""
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Please install it to load artifacts.") from exc

    run = wandb.init(project=project, entity=entity, job_type=job_type)
    artifact = run.use_artifact(artifact_ref, type="model")
    bundle_dir = Path(artifact.download())
    model, metadata, user_encoders, item_encoders = load_model_from_bundle(bundle_dir)
    run.finish()

    return model, metadata, user_encoders, item_encoders