from __future__ import annotations

import importlib
import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from .model import TowerConfig


def read_parquet_batches(path: str, batch_size: int) -> Iterable[pd.DataFrame]:
    if not (path.endswith(".parquet") or path.endswith(".pq")):
        raise ValueError(f"Only parquet inputs are supported: {path}")
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def read_table(path: str) -> pd.DataFrame:
    if not (path.endswith(".parquet") or path.endswith(".pq")):
        raise ValueError(f"Only parquet inputs are supported: {path}")
    return pd.read_parquet(path)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def save_encoders(path: str, encoders: Dict[str, Union["CategoryEncoder", "DenseEncoder"]]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    payload = {k: v.to_dict() for k, v in encoders.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_encoders(path: str) -> Dict[str, Union["CategoryEncoder", "DenseEncoder"]]:
    from .data import CategoryEncoder, DenseEncoder

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    encoders = {}
    for k, data in payload.items():
        if isinstance(data, dict) and data.get("type") == "category":
            encoders[k] = CategoryEncoder.from_dict(data)
        elif isinstance(data, dict) and data.get("type") == "dense":
            encoders[k] = DenseEncoder.from_dict(data)
        else:
            # Backward compatibility: no type field means CategoryEncoder (old format)
            enc = CategoryEncoder()
            enc.mapping = {str(key): int(value) for key, value in data.items()}
            encoders[k] = enc
    return encoders


def save_model_bundle(
    *,
    output_dir: str | Path,
    stage: str,
    model_state: Dict[str, torch.Tensor],
    feature_cfg: "FeatureConfig",
    user_encoders: Dict[str, Union["CategoryEncoder", "DenseEncoder"]],
    item_encoders: Dict[str, Union["CategoryEncoder", "DenseEncoder"]],
    tower_cfg,
    extra_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "model.pt",
) -> Dict[str, Any]:
    """Persist a complete model bundle for later inference or artifact logging."""
    from .data import CategoryEncoder, DenseEncoder, FeatureConfig

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


def _resolve_model_class(stage: str, model_arch: str | None) -> type[torch.nn.Module]:
    if stage == "retrieval":
        from ..retrieval import model as model_module
        from ..retrieval.model import get_model_class
    elif stage == "ranking":
        from ..ranking import model as model_module
        from ..ranking.model import get_model_class
    else:
        raise ValueError(f"Unsupported stage in metadata: {stage}")

    arch = model_arch or "two_tower"
    try:
        return get_model_class(arch)
    except ValueError:
        pass

    direct_match = getattr(model_module, arch, None)
    if isinstance(direct_match, type) and issubclass(direct_match, torch.nn.Module):
        return direct_match

    arch_lower = arch.lower()
    for name in dir(model_module):
        candidate = getattr(model_module, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, torch.nn.Module)
            and name.lower() == arch_lower
        ):
            return candidate

    raise ValueError(f"Unsupported {stage} model_arch: {arch}")


def _build_model_instance(
    *,
    stage: str,
    model_arch: str | None,
    metadata: Dict[str, Any],
    tower_cfg: TowerConfig,
) -> torch.nn.Module:
    model_cls = _resolve_model_class(stage, model_arch)
    feature_cfg = metadata.get("feature_config") or {}
    init_kwargs = {
        "user_cardinalities": metadata["user_cardinalities"],
        "item_cardinalities": metadata["item_cardinalities"],
        "tower_config": tower_cfg,
        "loss_func": metadata.get("loss_func"),
        "user_dense_features": feature_cfg.get("user_dense_cols") or [],
        "item_dense_features": feature_cfg.get("item_dense_cols") or [],
        "init_temperature": metadata.get("temperature", 0.05),
        "scorer_hidden_dims": metadata.get("scorer_hidden_dims"),
    }

    init_sig = inspect.signature(model_cls.__init__)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values())
    if accepts_var_kwargs:
        return model_cls(**init_kwargs)

    supported = {
        name
        for name, param in init_sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in supported}
    return model_cls(**filtered_kwargs)


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
    model = _build_model_instance(
        stage=stage,
        model_arch=metadata.get("model_arch"),
        metadata=metadata,
        tower_cfg=tower_cfg,
    )

    state_dict = _load_state_dict(bundle["checkpoint"])
    model.load_state_dict(state_dict)
    model.eval()

    return (
        model,
        metadata,
        bundle["user_encoders"],
        bundle["item_encoders"],
    )


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