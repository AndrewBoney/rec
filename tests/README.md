# Testing Guide

This directory contains the test suite for the recommendation system.

## Test Structure

- `conftest.py` - Shared fixtures and utilities
- `test_data.py` - Unit tests for data processing (encoders, feature stores)
- `test_models.py` - Unit tests for model architectures
- `test_benchmarks.py` - Performance benchmarks with timing

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only benchmark tests
```bash
pytest -m benchmark
```

### Run with specific dataset size
```bash
pytest -k "small"  # Only small dataset tests
pytest -k "medium"  # Only medium dataset tests
```

### Run with verbose output
```bash
pytest -v
```

## Benchmark Results

Benchmark tests automatically log timing results to `benchmark_results.json` in the project root.

### View benchmark results
```bash
python scripts/view_benchmarks.py
```

This will display:
- All benchmark runs
- Time comparisons between recent runs
- Performance trends

### In CI/CD

The GitHub Actions workflow automatically:
1. Runs all tests on push/PR
2. Generates benchmark results
3. Displays results in the workflow summary
4. Uploads results as artifacts

You can view results in:
- GitHub Actions → Your workflow → Summary section
- Download artifacts from the workflow run

## What Gets Tested

### Unit Tests (`-m unit`)
- CategoryEncoder and DenseEncoder functionality
- Encoder fixture creation
- Feature store creation and lookups
- Model forward/backward passes (TwoTowerRetrieval, TwoTowerRanking, DLRM)
- Loss computation and gradient flow

### Benchmark Tests (`-m benchmark`)
- Encoder building time (from DataFrames)
- Feature store creation time
- Batch loading performance (various batch sizes, saved to temp parquet)
- Model forward pass throughput
- End-to-end workflow timing (data prep → model → training)

### Benchmark Metrics Tracked
- Dataset sizes (users, items, interactions)
- Batch sizes
- Time per operation (seconds/milliseconds)
- Throughput (samples/second)
- Device (CPU/CUDA)
- Average training loss

## Adding New Tests

### Unit Test
```python
@pytest.mark.unit
def test_my_feature(dummy_data, feature_config):
    # Your test here
    pass
```

### Benchmark Test
```python
@pytest.mark.benchmark
def test_my_performance(dummy_data, dataset_size, benchmark_logger):
    start_time = time.time()
    # ... operation to benchmark
    elapsed = time.time() - start_time

    benchmark_logger.log("my_test_name", {
        "time_seconds": elapsed,
        "other_metric": value,
    })
```

## Requirements

Install test dependencies:
```bash
pip install -e .
pip install pytest pytest-cov
```
