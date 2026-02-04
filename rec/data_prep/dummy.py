from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import split_interactions_by_time


def _make_ids(prefix: str, n: int) -> List[str]:
    width = max(6, len(str(n)))
    return [f"{prefix}_{i:0{width}d}" for i in range(n)]


def _sample_categories(rng: np.random.Generator, n: int, k: int, prefix: str) -> List[str]:
    return [f"{prefix}_{i}" for i in rng.integers(0, k, size=n)]


def _bucket_age(age: np.ndarray) -> np.ndarray:
    bins = np.array([0, 18, 25, 35, 45, 55, 65, 200], dtype=np.int64)
    labels = np.array(
        ["age_0_17", "age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_64", "age_65_plus"],
        dtype=object,
    )
    idx = np.digitize(age, bins, right=False) - 1
    return labels[idx]


def _build_user_table(rng: np.random.Generator, n_users: int, n_user_groups: int) -> pd.DataFrame:
    users = _make_ids("u", n_users)
    age = rng.integers(16, 70, size=n_users)
    region = _sample_categories(rng, n_users, 8, "region")
    user_group = _sample_categories(rng, n_users, n_user_groups, "ug")
    return pd.DataFrame(
        {
            "user_id": users,
            "age_group": _bucket_age(age),
            "region": region,
            "user_group": user_group,
        }
    )


def _build_item_table(
    rng: np.random.Generator,
    n_items: int,
    n_item_groups: int,
    n_price_bands: int,
    n_style_groups: int,
) -> pd.DataFrame:
    items = _make_ids("i", n_items)
    price_band = _sample_categories(rng, n_items, n_price_bands, "price")
    category = _sample_categories(rng, n_items, n_item_groups, "cat")
    style = _sample_categories(rng, n_items, n_style_groups, "style")
    return pd.DataFrame(
        {
            "item_id": items,
            "price_band": price_band,
            "category": category,
            "style": style,
        }
    )


def _make_affinity_mapping(
    rng: np.random.Generator, source_keys: List[str], target_keys: List[str]
) -> Dict[str, str]:
    if len(target_keys) < len(source_keys):
        raise ValueError("Target keys must be >= source keys to keep 1-1 mapping")
    shuffled = target_keys.copy()
    rng.shuffle(shuffled)
    mapping: Dict[str, str] = {}
    for i, key in enumerate(source_keys):
        mapping[key] = shuffled[i]
    return mapping


def _build_item_feature_arrays(items: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        items["category"].to_numpy(),
        items["price_band"].to_numpy(),
        items["style"].to_numpy(),
    )


def _compute_user_item_probs(
    users: pd.DataFrame,
    items: pd.DataFrame,
    rng: np.random.Generator,
    noise_scale: float = 0.15,
) -> np.ndarray:
    categories, price_bands, styles = _build_item_feature_arrays(items)

    user_groups = sorted(users["user_group"].unique().tolist())
    age_groups = sorted(users["age_group"].unique().tolist())
    regions = sorted(users["region"].unique().tolist())

    cat_keys = sorted(items["category"].unique().tolist())
    price_keys = sorted(items["price_band"].unique().tolist())
    style_keys = sorted(items["style"].unique().tolist())

    group_to_category = _make_affinity_mapping(rng, user_groups, cat_keys)
    group_to_price = _make_affinity_mapping(rng, user_groups, price_keys)
    group_to_style = _make_affinity_mapping(rng, user_groups, style_keys)

    age_to_category = _make_affinity_mapping(rng, age_groups, cat_keys)
    age_to_price = _make_affinity_mapping(rng, age_groups, price_keys)
    age_to_style = _make_affinity_mapping(rng, age_groups, style_keys)

    region_to_category = _make_affinity_mapping(rng, regions, cat_keys)
    region_to_price = _make_affinity_mapping(rng, regions, price_keys)
    region_to_style = _make_affinity_mapping(rng, regions, style_keys)

    n_users = len(users)
    n_items = len(items)
    probs = np.empty((n_users, n_items), dtype=np.float32)

    user_group_arr = users["user_group"].to_numpy()
    age_group_arr = users["age_group"].to_numpy()
    region_arr = users["region"].to_numpy()

    for u in range(n_users):
        ug = user_group_arr[u]
        ag = age_group_arr[u]
        rg = region_arr[u]

        score = (
            (categories == group_to_category[ug]).astype(np.float32)
            + (price_bands == group_to_price[ug]).astype(np.float32)
            + (styles == group_to_style[ug]).astype(np.float32)
            + (categories == age_to_category[ag]).astype(np.float32)
            + (price_bands == age_to_price[ag]).astype(np.float32)
            + (styles == age_to_style[ag]).astype(np.float32)
            + (categories == region_to_category[rg]).astype(np.float32)
            + (price_bands == region_to_price[rg]).astype(np.float32)
            + (styles == region_to_style[rg]).astype(np.float32)
        )
        score += rng.normal(0.0, noise_scale, size=n_items).astype(np.float32)
        score = np.clip(score, -2.0, None)
        exp_score = np.exp(score - np.max(score))
        probs[u] = exp_score / exp_score.sum()

    return probs


def generate_dummy_data(
    output_dir: str,
    n_users: int,
    n_items: int,
    n_interactions: int,
    seed: int,
    chunk_size: int = 200_000,
    val_t: float = 0.2,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    prepared_dir = os.path.join(output_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(seed)

    n_user_groups = max(4, int(n_users ** (1 / 3)))
    n_age_groups = 7
    n_regions = 8
    min_item_groups = max(n_user_groups, n_age_groups, n_regions)

    n_item_groups = max(min_item_groups, int(n_items ** (1 / 3)))
    n_price_bands = max(min_item_groups, 7)
    n_style_groups = max(min_item_groups, 10)

    logger.info("Building user and item tables")
    users = _build_user_table(rng, n_users, n_user_groups)
    items = _build_item_table(rng, n_items, n_item_groups, n_price_bands, n_style_groups)

    users_path = os.path.join(prepared_dir, "users.parquet")
    items_path = os.path.join(prepared_dir, "items.parquet")
    interactions_train_path = os.path.join(prepared_dir, "interactions_train.parquet")
    interactions_val_path = os.path.join(prepared_dir, "interactions_val.parquet")

    users.to_parquet(users_path, index=False)
    items.to_parquet(items_path, index=False)

    logger.info("Computing user-item probability matrix")
    t0 = time.perf_counter()
    user_item_probs = _compute_user_item_probs(users, items, rng)
    logger.info("Computed probabilities in %.2fs", time.perf_counter() - t0)

    start_date = datetime(2020, 1, 1)
    total_written = 0

    interaction_chunks: List[pd.DataFrame] = []
    user_id_arr = users["user_id"].to_numpy()
    item_id_arr = items["item_id"].to_numpy()

    t_start = time.perf_counter()
    chunk_count = 0
    while total_written < n_interactions:
        batch = min(chunk_size, n_interactions - total_written)
        user_idx = rng.integers(0, n_users, size=batch)
        user_ids = user_id_arr[user_idx]
        item_idx = np.empty(batch, dtype=np.int64)
        unique_users, counts = np.unique(user_idx, return_counts=True)
        for u, cnt in zip(unique_users, counts):
            item_idx[user_idx == u] = rng.choice(n_items, size=cnt, p=user_item_probs[u])
        item_ids = item_id_arr[item_idx]

        days = rng.integers(0, 365, size=batch)
        timestamps = [(start_date + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in days]
        base_price = rng.normal(20, 5, size=batch).clip(1, 100)
        price = np.round(base_price + (item_idx % 5) * 2, 2)

        chunk_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "item_id": item_ids,
                "timestamp": timestamps,
                "price": price,
            }
        )
        interaction_chunks.append(chunk_df)
        total_written += batch
        chunk_count += 1
        if chunk_count % 5 == 0 or total_written == n_interactions:
            elapsed = time.perf_counter() - t_start
            rate = total_written / max(elapsed, 1e-6)
            logger.info("Generated %d/%d interactions (%.1f%%) at %.0f rows/s", total_written, n_interactions, 100 * total_written / n_interactions, rate)

    logger.info("Concatenating %d chunks", len(interaction_chunks))
    interactions = pd.concat(interaction_chunks, ignore_index=True)

    train_df, val_df = split_interactions_by_time(interactions, val_t=val_t)

    train_df.to_parquet(interactions_train_path, index=False)
    val_df.to_parquet(interactions_val_path, index=False)
    logger.info("Wrote parquet outputs to %s", prepared_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy recommender datasets")
    parser.add_argument("--output-dir", default="data/dummy")
    parser.add_argument("--n-users", type=int, default=10000)
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--n-interactions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--val-t", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2, dest="val_t")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generate_dummy_data(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        n_interactions=args.n_interactions,
        seed=args.seed,
        chunk_size=args.chunk_size,
        val_t=args.val_t,
    )


if __name__ == "__main__":
    main()
