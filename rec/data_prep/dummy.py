from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


def _make_ids(prefix: str, n: int) -> List[str]:
    width = max(6, len(str(n)))
    return [f"{prefix}_{i:0{width}d}" for i in range(n)]


def _sample_categories(rng: np.random.Generator, n: int, k: int, prefix: str) -> List[str]:
    return [f"{prefix}_{i}" for i in rng.integers(0, k, size=n)]


def _build_user_table(rng: np.random.Generator, n_users: int, n_user_groups: int) -> pd.DataFrame:
    users = _make_ids("u", n_users)
    age = rng.integers(16, 70, size=n_users)
    region = _sample_categories(rng, n_users, 8, "region")
    user_group = _sample_categories(rng, n_users, n_user_groups, "ug")
    return pd.DataFrame(
        {
            "user_id": users,
            "age": age.astype(str),
            "region": region,
            "user_group": user_group,
        }
    )


def _build_item_table(rng: np.random.Generator, n_items: int, n_item_groups: int) -> pd.DataFrame:
    items = _make_ids("i", n_items)
    price_band = _sample_categories(rng, n_items, 6, "price")
    category = _sample_categories(rng, n_items, n_item_groups, "cat")
    style = _sample_categories(rng, n_items, 10, "style")
    return pd.DataFrame(
        {
            "item_id": items,
            "price_band": price_band,
            "category": category,
            "style": style,
        }
    )


def _build_item_index(items: pd.DataFrame) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for idx, cat in enumerate(items["category"].tolist()):
        groups.setdefault(cat, []).append(idx)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def _sample_items_for_users(
    rng: np.random.Generator,
    user_groups: np.ndarray,
    item_groups: Dict[str, np.ndarray],
    n_items: int,
    p_prefer: float = 0.85,
) -> np.ndarray:
    chosen_items = np.empty(len(user_groups), dtype=np.int64)
    for i, ug in enumerate(user_groups):
        if rng.random() < p_prefer and ug in item_groups:
            chosen_items[i] = rng.choice(item_groups[ug])
        else:
            chosen_items[i] = rng.integers(0, n_items)
    return chosen_items


def generate_dummy_data(
    output_dir: str,
    n_users: int,
    n_items: int,
    n_interactions: int,
    seed: int,
    chunk_size: int = 200_000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_user_groups = max(4, int(np.sqrt(n_users)))
    n_item_groups = max(4, int(np.sqrt(n_items)))

    users = _build_user_table(rng, n_users, n_user_groups)
    items = _build_item_table(rng, n_items, n_item_groups)

    users_path = os.path.join(output_dir, "users.csv")
    items_path = os.path.join(output_dir, "items.csv")
    interactions_path = os.path.join(output_dir, "interactions.csv")

    users.to_csv(users_path, index=False)
    items.to_csv(items_path, index=False)

    user_groups = users["user_group"].to_numpy()
    item_groups = _build_item_index(items)

    start_date = datetime(2020, 1, 1)
    total_written = 0

    with open(interactions_path, "w", encoding="utf-8") as f:
        f.write("user_id,item_id,timestamp,price\n")

    while total_written < n_interactions:
        batch = min(chunk_size, n_interactions - total_written)
        user_idx = rng.integers(0, n_users, size=batch)
        user_ids = np.array(users["user_id"].tolist())[user_idx]
        user_group_batch = user_groups[user_idx]

        item_idx = _sample_items_for_users(
            rng,
            user_group_batch,
            item_groups,
            n_items,
            p_prefer=0.85,
        )
        item_ids = np.array(items["item_id"].tolist())[item_idx]

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
        chunk_df.to_csv(interactions_path, mode="a", header=False, index=False)
        total_written += batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy recommender datasets")
    parser.add_argument("--output-dir", default="data/dummy")
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--n-items", type=int, default=1000)
    parser.add_argument("--n-interactions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dummy_data(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        n_interactions=args.n_interactions,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
