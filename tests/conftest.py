"""Shared pytest fixtures for testing."""
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(params=["small", "medium"])
def dataset_size(request):
    """Parameterized fixture for different dataset sizes."""
    sizes = {
        "small": {"n_users": 100, "n_items": 50, "n_interactions": 500},
        "medium": {"n_users": 1000, "n_items": 500, "n_interactions": 10000},
    }
    return sizes[request.param]


@pytest.fixture
def dummy_data(dataset_size) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate dummy recommendation data."""
    n_users = dataset_size["n_users"]
    n_items = dataset_size["n_items"]
    n_interactions = dataset_size["n_interactions"]

    # Generate user features
    users = pd.DataFrame({
        "user_id": range(n_users),
        "age": torch.randint(18, 80, (n_users,)).tolist(),
        "gender": torch.randint(0, 3, (n_users,)).tolist(),
        "country": torch.randint(0, 10, (n_users,)).tolist(),
    })

    # Generate item features
    items = pd.DataFrame({
        "item_id": range(n_items),
        "category": torch.randint(0, 20, (n_items,)).tolist(),
        "price": (torch.rand(n_items) * 100).tolist(),
        "brand": torch.randint(0, 50, (n_items,)).tolist(),
    })

    # Generate interactions
    interactions = pd.DataFrame({
        "user_id": torch.randint(0, n_users, (n_interactions,)).tolist(),
        "item_id": torch.randint(0, n_items, (n_interactions,)).tolist(),
        "rating": (torch.rand(n_interactions) * 5).tolist(),
        "timestamp": torch.randint(1000000, 2000000, (n_interactions,)).tolist(),
    })

    # Remove duplicates to ensure unique user-item pairs
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"])

    return users, items, interactions


@pytest.fixture
def feature_config():
    """Standard feature configuration for testing."""
    from rec.common.data import FeatureConfig

    return FeatureConfig(
        user_id_col="user_id",
        item_id_col="item_id",
        user_cat_cols=["gender", "country"],
        user_dense_cols=["age"],
        item_cat_cols=["category", "brand"],
        item_dense_cols=["price"],
        interaction_user_col="user_id",
        interaction_item_col="item_id",
        interaction_time_col="timestamp",
        interaction_label_col="rating",
    )


@pytest.fixture
def device():
    """Get compute device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def encoders(dummy_data, feature_config):
    """Build encoders from dummy data."""
    from rec.common.data import CategoryEncoder, DenseEncoder

    users, items, interactions = dummy_data

    user_encoders = {}
    item_encoders = {}

    # User categorical encoders
    for col in [feature_config.user_id_col] + feature_config.user_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    # User dense encoders
    for col in feature_config.user_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    # Item categorical encoders
    for col in [feature_config.item_id_col] + feature_config.item_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    # Item dense encoders
    for col in feature_config.item_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    # Also fit ID encoders on interactions
    user_encoders[feature_config.user_id_col].fit(interactions[feature_config.interaction_user_col])
    item_encoders[feature_config.item_id_col].fit(interactions[feature_config.interaction_item_col])

    return user_encoders, item_encoders


@pytest.fixture
def feature_store(dummy_data, feature_config, encoders):
    """Build feature store from dummy data."""
    from rec.common.data import FeatureStore

    users, items, interactions = dummy_data
    user_encoders, item_encoders = encoders

    return FeatureStore(users, items, user_encoders, item_encoders, feature_config)


class BenchmarkLogger:
    """Simple logger for benchmark results."""

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.results = []

    def log(self, test_name: str, metrics: Dict):
        """Log a benchmark result."""
        self.results.append({
            "test": test_name,
            **metrics
        })

    def save(self):
        """Save results to JSON file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results if file exists
        existing = []
        if self.output_file.exists():
            with open(self.output_file) as f:
                existing = json.load(f)

        # Append new results
        existing.extend(self.results)

        # Save combined results
        with open(self.output_file, "w") as f:
            json.dump(existing, f, indent=2)


@pytest.fixture
def benchmark_logger():
    """Fixture for logging benchmark results."""
    # Save in project root for easy access and tracking
    project_root = Path(__file__).parent.parent
    logger = BenchmarkLogger(project_root / "benchmark_results.json")
    yield logger
    logger.save()
