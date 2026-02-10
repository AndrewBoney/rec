"""Performance benchmark tests with timing."""
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from rec.common.data import CategoryEncoder, DenseEncoder, FeatureStore
from rec.common.model import TowerConfig
from rec.retrieval.model import TwoTowerRetrieval


@pytest.mark.benchmark
def test_encoder_build_performance(dummy_data, feature_config, dataset_size, benchmark_logger):
    """Benchmark encoder building time."""
    users, items, interactions = dummy_data

    start_time = time.time()

    # Build encoders manually
    user_encoders = {}
    item_encoders = {}

    for col in [feature_config.user_id_col] + feature_config.user_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    for col in feature_config.user_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    for col in [feature_config.item_id_col] + feature_config.item_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    for col in feature_config.item_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    elapsed = time.time() - start_time

    metrics = {
        "n_users": dataset_size["n_users"],
        "n_items": dataset_size["n_items"],
        "time_seconds": elapsed,
    }

    benchmark_logger.log("encoder_build", metrics)
    print(f"\nEncoder build ({dataset_size['n_users']} users, {dataset_size['n_items']} items): {elapsed:.4f}s")


@pytest.mark.benchmark
def test_feature_store_build_performance(
    dummy_data, feature_config, encoders, dataset_size, benchmark_logger
):
    """Benchmark feature store building time."""
    users, items, interactions = dummy_data
    user_encoders, item_encoders = encoders

    start_time = time.time()
    feature_store = FeatureStore(users, items, user_encoders, item_encoders, feature_config)
    elapsed = time.time() - start_time

    metrics = {
        "n_users": dataset_size["n_users"],
        "n_items": dataset_size["n_items"],
        "time_seconds": elapsed,
    }

    benchmark_logger.log("feature_store_build", metrics)
    print(f"\nFeature store build ({dataset_size['n_users']} users): {elapsed:.4f}s")


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [32, 128, 1024])
def test_batch_loading_performance(
    dummy_data, feature_store, dataset_size, batch_size, benchmark_logger, temp_dir
):
    """Benchmark batch loading time with different batch sizes."""
    users, items, interactions = dummy_data

    # Save interactions to temp file for InteractionIterableDataset
    interactions_path = temp_dir / "interactions.parquet"
    interactions.to_parquet(interactions_path)

    from rec.common.data import InteractionIterableDataset
    import numpy as np

    # Create item pool for negative sampling
    item_id_pool = np.array(items["item_id"].tolist())

    dataset = InteractionIterableDataset(
        interactions_path=str(interactions_path),
        feature_store=feature_store,
        chunksize=10000,
        batch_size=batch_size,
        negatives_per_pos=2,
        item_id_pool=item_id_pool,
        include_labels=False,
    )

    # Measure time to load N batches
    n_batches = 10
    start_time = time.time()

    for i, batch in enumerate(dataset):
        if i >= n_batches:
            break

    elapsed = time.time() - start_time
    time_per_batch = elapsed / min(n_batches, i + 1) if i >= 0 else 0

    metrics = {
        "n_users": dataset_size["n_users"],
        "n_items": dataset_size["n_items"],
        "n_interactions": dataset_size["n_interactions"],
        "batch_size": batch_size,
        "batches_loaded": min(n_batches, i + 1),
        "total_time_seconds": elapsed,
        "time_per_batch_ms": time_per_batch * 1000,
    }

    benchmark_logger.log("batch_loading", metrics)
    print(
        f"\nBatch loading (size={batch_size}, n={min(n_batches, i + 1)}): "
        f"{time_per_batch*1000:.2f}ms/batch"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [32, 128, 1024])
def test_model_forward_performance(
    cardinalities, feature_store, dataset_size, batch_size, device, benchmark_logger
):
    """Benchmark model forward pass time."""
    user_cardinalities, item_cardinalities = cardinalities

    # Build model
    tower_config = TowerConfig(
        embedding_dim=32,
        hidden_dims=[64, 32],
        dropout=0.1,
    )

    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
    ).to(device)

    model.eval()

    # Create batch
    user_ids = torch.randint(0, dataset_size["n_users"], (batch_size,))
    item_ids = torch.randint(0, dataset_size["n_items"], (batch_size,))

    user_features = feature_store.get_user_features(user_ids)
    item_features = feature_store.get_item_features(item_ids)

    batch = {
        **{f"user_{k}": v for k, v in user_features.items()},
        **{f"item_{k}": v for k, v in item_features.items()},
    }


    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)

    # Benchmark
    n_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(n_iterations):
            scores = model(batch)

    elapsed = time.time() - start_time
    time_per_forward = elapsed / n_iterations

    metrics = {
        "batch_size": batch_size,
        "device": str(device),
        "n_iterations": n_iterations,
        "total_time_seconds": elapsed,
        "time_per_forward_ms": time_per_forward * 1000,
        "throughput_samples_per_sec": batch_size / time_per_forward,
    }

    benchmark_logger.log("model_forward", metrics)
    print(
        f"\nModel forward (batch_size={batch_size}, device={device}): "
        f"{time_per_forward*1000:.2f}ms/batch, "
        f"{batch_size/time_per_forward:.0f} samples/sec"
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_end_to_end_workflow(
    dummy_data, feature_config, dataset_size, device, benchmark_logger, temp_dir
):
    """Benchmark complete workflow: data prep -> model creation -> training step."""
    users, items, interactions = dummy_data

    timings = {}

    # 1. Build encoders
    start = time.time()
    user_encoders = {}
    item_encoders = {}

    for col in [feature_config.user_id_col] + feature_config.user_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    for col in feature_config.user_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(users[col])
        user_encoders[col] = encoder

    for col in [feature_config.item_id_col] + feature_config.item_cat_cols:
        encoder = CategoryEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    for col in feature_config.item_dense_cols:
        encoder = DenseEncoder()
        encoder.fit(items[col])
        item_encoders[col] = encoder

    timings["encoder_build"] = time.time() - start

    # 2. Build feature store
    start = time.time()
    feature_store = FeatureStore(users, items, user_encoders, item_encoders, feature_config)
    timings["feature_store_build"] = time.time() - start

    # 3. Create dataset
    start = time.time()

    # Save interactions to temp file
    interactions_path = temp_dir / "interactions.parquet"
    interactions.to_parquet(interactions_path)

    from rec.common.data import InteractionIterableDataset
    import numpy as np

    # Create item pool for negative sampling
    item_id_pool = np.array(items["item_id"].tolist())

    dataset = InteractionIterableDataset(
        interactions_path=str(interactions_path),
        feature_store=feature_store,
        chunksize=10000,
        batch_size=32,
        negatives_per_pos=2,
        item_id_pool=item_id_pool,
        include_labels=True,
    )
    timings["dataset_creation"] = time.time() - start

    # 4. Build model
    start = time.time()

    # Build cardinalities from encoders (only categorical)
    user_cardinalities = {
        name: enc.num_embeddings
        for name, enc in user_encoders.items()
        if isinstance(enc, CategoryEncoder)
    }
    item_cardinalities = {
        name: enc.num_embeddings
        for name, enc in item_encoders.items()
        if isinstance(enc, CategoryEncoder)
    }

    tower_config = TowerConfig(
        embedding_dim=32,
        hidden_dims=[64],
        dropout=0.1,
    )
    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    timings["model_creation"] = time.time() - start

    # 5. Training step (limited batches)
    start = time.time()
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataset:
        optimizer.zero_grad()

        # Batch already has user_<feature> and item_<feature> keys
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model.compute_loss(batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches >= 20:  # Limit for faster testing
            break

    timings["training_20_batches"] = time.time() - start
    timings["avg_loss"] = total_loss / n_batches if n_batches > 0 else 0

    metrics = {
        "n_users": dataset_size["n_users"],
        "n_items": dataset_size["n_items"],
        "n_interactions": dataset_size["n_interactions"],
        "device": str(device),
        **{f"{k}_seconds": v for k, v in timings.items() if k != "avg_loss"},
        "avg_loss": timings["avg_loss"],
        "total_workflow_seconds": sum(
            v for k, v in timings.items() if k != "avg_loss"
        ),
    }

    benchmark_logger.log("end_to_end_workflow", metrics)
    print(f"\n=== End-to-End Workflow Benchmark ===")
    print(f"Dataset: {dataset_size['n_users']} users, {dataset_size['n_items']} items")
    for key, value in timings.items():
        if key == "avg_loss":
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}s")
    print(f"  Total: {metrics['total_workflow_seconds']:.4f}s")
