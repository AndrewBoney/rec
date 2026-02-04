from __future__ import annotations

import argparse
import logging
import os
import re
import zipfile
from datetime import datetime
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .utils import split_interactions_by_time

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _download_and_extract(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, "ml-1m.zip")
    extract_dir = os.path.join(output_dir, "ml-1m")
    if os.path.exists(extract_dir):
        return extract_dir
    if not os.path.exists(archive_path):
        logging.info("Downloading MovieLens data from %s", url)
        urlretrieve(url, archive_path)
    logging.info("Extracting %s", archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)
    return extract_dir


def _load_users(path: str) -> pd.DataFrame:
    cols = ["user_id", "gender", "age", "occupation", "zip"]
    df = pd.read_csv(path, sep="::", names=cols, engine="python", encoding="latin-1")
    df["user_id"] = df["user_id"].astype(str)
    df["gender"] = df["gender"].astype(str)
    df["age_group"] = df["age"].astype(str).apply(lambda x: f"age_{x}")
    df["occupation"] = df["occupation"].astype(str).apply(lambda x: f"occ_{x}")
    df["zip_prefix"] = df["zip"].astype(str).str.slice(0, 3).fillna("000")
    return df[["user_id", "gender", "age_group", "occupation", "zip_prefix"]]


def _extract_year(title: str) -> Tuple[str, str]:
    match = re.search(r"\((\d{4})\)", str(title))
    if not match:
        return "year_unknown", "year_unknown"
    year = int(match.group(1))
    decade = (year // 10) * 10
    return str(year), f"year_{decade}s"


def _load_movies(path: str) -> pd.DataFrame:
    cols = ["item_id", "title", "genres"]
    df = pd.read_csv(path, sep="::", names=cols, engine="python", encoding="latin-1")
    df["item_id"] = df["item_id"].astype(str)
    df["genre_primary"] = df["genres"].astype(str).apply(lambda x: x.split("|")[0])
    years = df["title"].apply(_extract_year)
    df["year_bucket"] = years.apply(lambda x: x[1])
    return df[["item_id", "genre_primary", "year_bucket"]]


def _load_ratings(path: str) -> pd.DataFrame:
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="::", names=cols, engine="python", encoding="latin-1")
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    return df[["user_id", "item_id", "rating", "timestamp"]]


def generate_movielens_1m(
    output_dir: str,
    seed: int = 42,
    val_t: float = 0.2,
    url: str = MOVIELENS_1M_URL,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    prepared_dir = os.path.join(output_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    logging.info("Preparing MovieLens 1M dataset")
    extract_dir = _download_and_extract(url, output_dir)

    users_df = _load_users(os.path.join(extract_dir, "users.dat"))
    items_df = _load_movies(os.path.join(extract_dir, "movies.dat"))
    interactions_df = _load_ratings(os.path.join(extract_dir, "ratings.dat"))

    train_df, val_df = split_interactions_by_time(interactions_df, val_t=val_t)

    users_path = os.path.join(prepared_dir, "users.parquet")
    items_path = os.path.join(prepared_dir, "items.parquet")
    interactions_train_path = os.path.join(prepared_dir, "interactions_train.parquet")
    interactions_val_path = os.path.join(prepared_dir, "interactions_val.parquet")

    users_df.to_parquet(users_path, index=False)
    items_df.to_parquet(items_path, index=False)
    train_df.to_parquet(interactions_train_path, index=False)
    val_df.to_parquet(interactions_val_path, index=False)

    logging.info("Wrote MovieLens data to %s", prepared_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare MovieLens 1M")
    parser.add_argument("--output-dir", default="data/movielens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-t", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2, dest="val_t")
    parser.add_argument("--url", default=MOVIELENS_1M_URL)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generate_movielens_1m(
        output_dir=args.output_dir,
        seed=args.seed,
        val_t=args.val_t,
        url=args.url,
    )


if __name__ == "__main__":
    main()
