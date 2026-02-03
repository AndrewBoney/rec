from __future__ import annotations

import argparse
from typing import Callable

from .config import apply_config, build_base_parser, ensure_dataset_args, load_yaml_config
from .train import train as stage_train


def build_stage_parser(
    description: str,
    add_stage_args: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = build_base_parser(description)
    add_stage_args(parser)
    return parser


def run_stage_cli(
    stage: str,
    description: str,
    add_stage_args: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
    train_fn: Callable[[argparse.Namespace], str] = stage_train,
) -> str:
    parser = build_stage_parser(description, add_stage_args)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    if cfg:
        args = apply_config(args, cfg, stage=stage)
    args = ensure_dataset_args(args)
    return train_fn(args)
