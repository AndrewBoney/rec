from __future__ import annotations

import argparse

from .common.config import (
    add_ranking_args,
    add_retrieval_args,
    apply_dataset_config,
    apply_shared_config,
    apply_stage_config,
    build_base_parser,
    ensure_dataset_args,
    load_yaml_config,
)
from .retrieval import train as retrieval_train
from .ranking import train as ranking_train

# Build argument parser
def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Train retrieval then ranking")
    add_retrieval_args(parser, include_checkpoint=False)
    add_ranking_args(parser, include_checkpoint=False)
    parser.add_argument("--retrieval-checkpoint", default="retrieval.ckpt")
    parser.add_argument("--ranking-checkpoint", default="ranking.ckpt")
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)
    if not cfg:
        return args
    args = apply_dataset_config(args, cfg)
    args = apply_shared_config(args, cfg)
    return args


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if cfg:
        args = apply_dataset_config(args, cfg)
        args = apply_shared_config(args, cfg)
    args = ensure_dataset_args(args)

    retrieval_args = argparse.Namespace(**vars(args))
    retrieval_args.save_checkpoint = args.retrieval_checkpoint
    if cfg:
        retrieval_args = apply_stage_config(retrieval_args, cfg, "retrieval")

    retrieval_ckpt = retrieval_train.train(retrieval_args)

    ranking_args = argparse.Namespace(**vars(args))
    ranking_args.save_checkpoint = args.ranking_checkpoint
    if cfg:
        ranking_args = apply_stage_config(ranking_args, cfg, "ranking")
    ranking_args.init_from_retrieval = retrieval_ckpt

    ranking_train.train(ranking_args)


if __name__ == "__main__":
    main()
