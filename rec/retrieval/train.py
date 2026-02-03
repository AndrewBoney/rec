from __future__ import annotations

import argparse

from ..common.cli import build_stage_parser, run_stage_cli
from ..common.config import add_retrieval_args
from ..common.train import train as stage_train


def parse_args() -> argparse.Namespace:
    parser = build_stage_parser("Two-tower retrieval training", add_retrieval_args)
    return parser.parse_args()


def train(args: argparse.Namespace) -> str:
    return stage_train(args, stage="retrieval")


def main() -> None:
    run_stage_cli(
        stage="retrieval",
        description="Two-tower retrieval training",
        add_stage_args=add_retrieval_args,
        train_fn=train,
    )


if __name__ == "__main__":
    main()
