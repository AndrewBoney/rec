from __future__ import annotations

import argparse

from .common.config import (
    apply_config,
    build_base_parser,
    ensure_dataset_args,
    load_yaml_config,
)
from .retrieval.train import train as retrieval_train
from .ranking.train import train as ranking_train

# Build argument parser
def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Train retrieval then ranking")
    parser.add_argument("--retrieval-artifact-dir", default=None)
    parser.add_argument("--ranking-artifact-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if cfg:
        args = apply_config(args, cfg)
    args = ensure_dataset_args(args)

    retrieval_args = argparse.Namespace(**vars(args))
    if cfg:
        retrieval_args = apply_config(retrieval_args, cfg, stage="retrieval")
    if args.retrieval_artifact_dir:
        retrieval_args.artifact_dir = args.retrieval_artifact_dir

    retrieval_ckpt = retrieval_train(retrieval_args)

    ranking_args = argparse.Namespace(**vars(args))
    if cfg:
        ranking_args = apply_config(ranking_args, cfg, stage="ranking")
    if args.ranking_artifact_dir:
        ranking_args.artifact_dir = args.ranking_artifact_dir
    ranking_args.init_from_retrieval = retrieval_ckpt

    ranking_train(ranking_args)

if __name__ == "__main__":
    main()
