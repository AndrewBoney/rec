from __future__ import annotations

import argparse

from ..common.config import (
    add_retrieval_args,
    apply_config,
    build_base_parser,
    ensure_dataset_args,
    load_yaml_config,
)
from ..common.train import train as stage_train


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Two-tower retrieval training")
    add_retrieval_args(parser)
    return parser.parse_args()


def train(args: argparse.Namespace) -> str:
    return stage_train(args, stage="retrieval")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if cfg:
        args = apply_config(args, cfg, stage="retrieval")
    args = ensure_dataset_args(args)
    train(args)


if __name__ == "__main__":
    main()
