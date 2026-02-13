from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from .io import load_config


def parse_optimizer_args(optimizer_args: List[str]) -> Dict[str, Any]:
    """Parse optimizer arguments from key=value pairs.

    Handles type conversion for common optimizer parameters.
    """
    parsed = {}
    for arg in optimizer_args:
        if "=" not in arg:
            raise ValueError(f"Invalid optimizer argument format: '{arg}'. Expected key=value")

        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Try int first
            if "." not in value and "e" not in value.lower():
                parsed[key] = int(value)
            else:
                # Try float
                parsed[key] = float(value)
        except ValueError:
            # Keep as string for bool or other types
            if value.lower() == "true":
                parsed[key] = True
            elif value.lower() == "false":
                parsed[key] = False
            else:
                parsed[key] = value

    return parsed


_COLUMN_ALIAS_MAP = {
    "user_id": "user_id_col",
    "item_id": "item_id_col",
    "label_col": "interaction_label_col",
}


def _apply_column_config(
    args: argparse.Namespace,
    columns: Dict[str, Any],
) -> argparse.Namespace:
    column_aliases = {k: v for k, v in columns.items() if k in _COLUMN_ALIAS_MAP}
    if column_aliases:
        _apply_config_values(args, column_aliases, key_map=_COLUMN_ALIAS_MAP)
    _apply_config_values(
        args,
        {k: v for k, v in columns.items() if k not in _COLUMN_ALIAS_MAP},
    )
    return args


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--users", required=False)
    parser.add_argument("--items", required=False)
    parser.add_argument("--interactions-train", required=False)
    parser.add_argument("--interactions-val", required=False)
    parser.add_argument("--user-id-col", default="customer_id")
    parser.add_argument("--item-id-col", default="article_id")
    parser.add_argument("--user-cat-cols", nargs="*", default=["FN", "Active", "club_member_status", "fashion_news_frequency", "age"])
    parser.add_argument("--item-cat-cols", nargs="*", default=["product_type_no", "graphical_appearance_no", "colour_group_code", "section_no", "garment_group_no"])
    parser.add_argument("--user-dense-cols", nargs="*", default=[])
    parser.add_argument("--item-dense-cols", nargs="*", default=[])
    parser.add_argument("--interaction-user-col", default="customer_id")
    parser.add_argument("--interaction-item-col", default="article_id")
    parser.add_argument("--interaction-label-col", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-cache", default="encoders.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--dense-bottom-mlp-dims", nargs="*", type=int, default=None, help="Hidden dims for dense feature bottom MLP (e.g., 64 32)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        help="Optimizer class name from torch.optim (e.g., Adam, AdamW, SGD, RMSprop)",
    )
    parser.add_argument(
        "--optimizer-args",
        nargs="*",
        default=[],
        help="Additional optimizer arguments as key=value pairs (e.g., momentum=0.9 weight_decay=1e-4)",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        help="LR scheduler class name from torch.optim.lr_scheduler (e.g., StepLR, CosineAnnealingLR, OneCycleLR)",
    )
    parser.add_argument(
        "--scheduler-args",
        nargs="*",
        default=[],
        help="Additional scheduler arguments as key=value pairs (e.g., step_size=10 gamma=0.1)",
    )
    parser.add_argument(
        "--scheduler-interval",
        default="epoch",
        choices=["epoch", "step"],
        help="When to step the scheduler: per epoch or per training step (default: epoch)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (FP16) using torch.cuda.amp",
    )
    parser.add_argument("--loss-func", default=None, help="Loss function override (stage-dependent defaults apply)")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="rec")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-artifact-name", default=None, help="W&B artifact name for logged model bundles")
    parser.add_argument(
        "--wandb-artifact-aliases",
        nargs="*",
        default=["latest"],
        help="Aliases to attach to logged model artifacts (default: latest)",
    )
    parser.add_argument(
        "--wandb-log-datasets",
        action="store_true",
        default=True,
        help="Log dataset files (users/items/interactions) as W&B artifacts",
    )
    return parser


def add_retrieval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-arch",
        default="two_tower",
        choices=["two_tower"],
        help="Retrieval model architecture",
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--save-checkpoint", default="retrieval.ckpt")
    parser.add_argument("--artifact-dir", default=None)
    return parser


def add_ranking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-arch",
        default="two_tower",
        choices=["two_tower", "dlrm"],
        help="Ranking model architecture",
    )
    parser.add_argument("--negatives-per-pos", type=int, default=4)
    parser.add_argument("--init-from-retrieval", default=None)
    parser.add_argument(
        "--scorer-hidden-dims",
        nargs="*",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the ranking scorer MLP (output layer is fixed to 1)",
    )
    parser.add_argument("--save-checkpoint", default="ranking.ckpt")
    parser.add_argument("--artifact-dir", default=None)
    return parser


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    return load_config(path)


def _apply_config_values(
    args: argparse.Namespace,
    values: Dict[str, Any],
    key_map: Optional[Dict[str, str]] = None,
) -> argparse.Namespace:
    mapping = key_map or {}
    for key, value in values.items():
        target = mapping.get(key, key)
        setattr(args, target, value)
    return args


def _flatten_config(cfg: Dict[str, Any], stage: Optional[str] = None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(cfg.get("args", {}))

    dataset = cfg.get("dataset", {})
    merged.update(dataset.get("paths", {}))

    base_columns = dataset.get("columns", {})
    stage_columns = {}
    if stage:
        stage_columns = cfg.get(stage, {}).get("columns", {})
    columns = {**base_columns, **stage_columns}
    merged.update({_COLUMN_ALIAS_MAP.get(k, k): v for k, v in columns.items()})

    merged.update(cfg.get("shared", {}))

    if stage:
        stage_cfg = cfg.get(stage, {})
        merged.update(stage_cfg.get("model", {}))
        training_cfg = stage_cfg.get("training", {})
        merged.update(training_cfg)
        if "checkpoint" in stage_cfg:
            merged["save_checkpoint"] = stage_cfg["checkpoint"]
        elif "checkpoint" in training_cfg and "save_checkpoint" not in merged:
            merged["save_checkpoint"] = training_cfg["checkpoint"]

    # Handle optimizer config - can be string or dict
    _handle_optimizer_config(merged)

    # Handle scheduler config - can be string or dict
    _handle_scheduler_config(merged)

    return merged


def _handle_optimizer_config(merged: Dict[str, Any]) -> None:
    """Process optimizer config from dict format to args format.

    Converts:
        optimizer: {name: "SGD", momentum: 0.9, weight_decay: 1e-4}
    To:
        optimizer: "SGD"
        optimizer_args: {"momentum": 0.9, "weight_decay": 1e-4}
    """
    if "optimizer" in merged and isinstance(merged["optimizer"], dict):
        optimizer_dict = merged["optimizer"]
        # Extract name
        optimizer_name = optimizer_dict.get("name")
        if not optimizer_name:
            raise ValueError("optimizer config must have 'name' field")

        # Extract other args (excluding 'name' and 'lr')
        optimizer_kwargs = {
            k: v for k, v in optimizer_dict.items()
            if k not in ("name", "lr")
        }

        # Update merged config
        merged["optimizer"] = optimizer_name
        if optimizer_kwargs:
            merged["optimizer_kwargs"] = optimizer_kwargs

        # If lr is in optimizer dict, use it
        if "lr" in optimizer_dict:
            merged["lr"] = optimizer_dict["lr"]


def _handle_scheduler_config(merged: Dict[str, Any]) -> None:
    """Process scheduler config from dict format to args format.

    Converts:
        scheduler: {name: "StepLR", step_size: 10, gamma: 0.1, interval: "epoch"}
    To:
        scheduler: "StepLR"
        scheduler_kwargs: {"step_size": 10, "gamma": 0.1}
        scheduler_interval: "epoch"
    """
    if "scheduler" in merged and isinstance(merged["scheduler"], dict):
        scheduler_dict = merged["scheduler"]
        # Extract name
        scheduler_name = scheduler_dict.get("name")
        if not scheduler_name:
            raise ValueError("scheduler config must have 'name' field")

        # Extract interval if present
        scheduler_interval = scheduler_dict.get("interval", "epoch")

        # Extract other args (excluding 'name' and 'interval')
        scheduler_kwargs = {
            k: v for k, v in scheduler_dict.items()
            if k not in ("name", "interval")
        }

        # Update merged config
        merged["scheduler"] = scheduler_name
        if scheduler_kwargs:
            merged["scheduler_kwargs"] = scheduler_kwargs
        merged["scheduler_interval"] = scheduler_interval


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any], stage: Optional[str] = None) -> argparse.Namespace:
    values = _flatten_config(cfg, stage=stage)
    return _apply_config_values(args, values)


def apply_dataset_config(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    stage: Optional[str] = None,
) -> argparse.Namespace:
    dataset = cfg.get("dataset", {})
    paths = dataset.get("paths", {})
    base_columns = dataset.get("columns", {})
    stage_columns = {}
    if stage:
        stage_columns = cfg.get(stage, {}).get("columns", {})
    columns = {**base_columns, **stage_columns}

    _apply_config_values(args, paths)
    _apply_column_config(args, columns)

    return args


def apply_shared_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    shared = cfg.get("shared", {})
    _apply_config_values(args, shared)
    return args


def apply_stage_config(args: argparse.Namespace, cfg: Dict[str, Any], stage: str) -> argparse.Namespace:
    stage_cfg = cfg.get(stage, {})
    columns = stage_cfg.get("columns", {})
    model_cfg = stage_cfg.get("model", {})
    training_cfg = stage_cfg.get("training", {})

    _apply_column_config(args, columns)
    _apply_config_values(args, model_cfg)
    _apply_config_values(
        args,
        training_cfg,
        key_map={
            "checkpoint": "save_checkpoint",
        },
    )

    return args


def ensure_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    missing = [name for name in ("users", "items") if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required dataset args: {', '.join(missing)}")
    if not args.interactions_train:
        raise ValueError("Missing required dataset args: interactions_train")
    if not args.interactions_val:
        raise ValueError("Missing required dataset args: interactions_val")
    return args
