from __future__ import annotations

import argparse

from .retrieval import train as retrieval_train
from .ranking import train as ranking_train

# Build argument parser
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval then ranking")
    parser.add_argument("--users", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--user-id-col", default="customer_id")
    parser.add_argument("--item-id-col", default="article_id")
    parser.add_argument("--user-cat-cols", nargs="*", default=["FN", "Active", "club_member_status", "fashion_news_frequency", "age"])
    parser.add_argument("--item-cat-cols", nargs="*", default=["product_type_no", "graphical_appearance_no", "colour_group_code", "section_no", "garment_group_no"])
    parser.add_argument("--interaction-user-col", default="customer_id")
    parser.add_argument("--interaction-item-col", default="article_id")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--negatives-per-pos", type=int, default=4)
    parser.add_argument("--encoder-cache", default="encoders.json")
    parser.add_argument("--retrieval-checkpoint", default="retrieval.ckpt")
    parser.add_argument("--ranking-checkpoint", default="ranking.ckpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retrieval_args = argparse.Namespace(**vars(args))
    retrieval_args.save_checkpoint = args.retrieval_checkpoint

    retrieval_ckpt = retrieval_train.train(retrieval_args)

    ranking_args = argparse.Namespace(**vars(args))
    ranking_args.init_from_retrieval = retrieval_ckpt
    ranking_args.save_checkpoint = args.ranking_checkpoint

    ranking_train.train(ranking_args)


if __name__ == "__main__":
    main()
