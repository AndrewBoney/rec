# Training Configuration Guide

This guide covers all available training options including optimizer configuration, learning rate scheduling, mixed precision training, and early stopping.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Optimizer Configuration](#optimizer-configuration)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Mixed Precision Training](#mixed-precision-training)
- [Early Stopping](#early-stopping)
- [Complete Examples](#complete-examples)

## Basic Configuration

All training configurations are specified in YAML files under the `config/` directory. The basic structure is:

```yaml
retrieval:
  training:
    batch_size: 1024
    max_epochs: 10
    lr: 0.001
    # ... other options
```

## Optimizer Configuration

You can configure any optimizer from `torch.optim` using either a simple string or a dictionary format.

### Simple String Format

Use just the optimizer name to use default parameters:

```yaml
training:
  lr: 0.001
  optimizer: AdamW  # Uses default AdamW parameters
```

### Dictionary Format

Specify optimizer parameters as a dictionary:

```yaml
training:
  lr: 0.001
  optimizer:
    name: AdamW
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
```

### Available Optimizers

Any optimizer from `torch.optim` can be used. Common choices:

**AdamW** (recommended for most cases):
```yaml
optimizer:
  name: AdamW
  weight_decay: 0.01  # L2 regularization
  betas: [0.9, 0.999]
```

**SGD with Momentum**:
```yaml
optimizer:
  name: SGD
  momentum: 0.9
  weight_decay: 0.0001
  nesterov: true
```

### Command Line Override

```bash
# Simple override
python rec/retrieval/train.py --config config.yaml --optimizer Adam

# With parameters
python rec/retrieval/train.py --config config.yaml \
  --optimizer SGD \
  --optimizer-args momentum=0.9 weight_decay=0.0001
```

## Learning Rate Scheduling

Learning rate schedulers from `torch.optim.lr_scheduler` can be configured to dynamically adjust the learning rate during training.

### Basic Scheduler Configuration

```yaml
training:
  lr: 0.001
  optimizer: AdamW
  scheduler:
    name: StepLR
    step_size: 10
    gamma: 0.1
    interval: epoch  # Options: "epoch" or "step"
```

The `interval` parameter controls when the scheduler steps:
- `epoch`: Step once per epoch (default)
- `step`: Step once per training iteration

### Common Schedulers

**StepLR**  - Reduce LR every N epochs:
```yaml
scheduler:
  name: StepLR
  step_size: 10
  gamma: 0.5
  interval: epoch
```

**CosineAnnealingLR** - Cosine annealing:
```yaml
scheduler:
  name: CosineAnnealingLR
  T_max: 50
  eta_min: 1e-6
  interval: epoch
```

## Mixed Precision Training

Mixed precision training uses FP16 (half precision) for faster training and reduced memory usage on modern GPUs.

### Enable Mixed Precision

In config file:
```yaml
training:
  mixed_precision: true
  batch_size: 16384  # Can use larger batches with FP16
```

Or via command line:
```bash
python rec/retrieval/train.py --config config.yaml --mixed-precision
```

### Benefits

- **~2x faster training** on modern GPUs
- **~50% less memory usage** - allows larger batch sizes
- **Automatic gradient scaling** - prevents underflow

## Early Stopping

Early stopping automatically stops training when a validation metric stops improving, preventing overfitting and saving training time.

### Basic Early Stopping

Simple boolean form (uses defaults):
```yaml
training:
  max_epochs: 100  # Can set high with early stopping
  early_stopping: true  # Monitor recall@10 with patience 5
```

### Full Configuration

```yaml
training:
  max_epochs: 100
  early_stopping:
    enabled: true
    metric: recall@10        # Metric to monitor
    patience: 10             # Epochs with no improvement before stopping
    mode: max                # "max" for metrics to maximize, "min" to minimize
    min_delta: 0.0001        # Minimum improvement to count
```

### Available Modes

- **`max`** mode: Stops when metric stops increasing (for recall, precision, NDCG, F1, etc.)
- **`min`** mode: Stops when metric stops decreasing (for loss, error, etc.)

### Common Metrics

**Retrieval Metrics** (mode: max):
- `recall@5`, `recall@10`, `recall@20`
- `ndcg@5`, `ndcg@10`, `ndcg@20`
- `hit_rate@5`, `hit_rate@10`
- `precision@5`, `precision@10`

**Ranking Metrics** (mode: max):
- `ndcg@10`
- `map@10`
- `auc`

**Loss Metrics** (mode: min):
- `epoch_loss`
- `mse`
- `mae`

### Examples

Monitor recall@10 (maximize):
```yaml
early_stopping:
  enabled: true
  metric: recall@10
  patience: 15
  mode: max
  min_delta: 0.001
```

Monitor validation loss (minimize):
```yaml
early_stopping:
  enabled: true
  metric: epoch_loss
  patience: 10
  mode: min
  min_delta: 0.01
```

### Command Line Override

```bash
# Enable with defaults
python rec/retrieval/train.py --config config.yaml --early-stopping

# Customize parameters
python rec/retrieval/train.py --config config.yaml \
  --early-stopping \
  --early-stopping-metric ndcg@10 \
  --early-stopping-patience 20 \
  --early-stopping-mode max \
  --early-stopping-min-delta 0.0005
```

### How It Works

1. After each epoch, the validation metrics are evaluated
2. Early stopping checks if the monitored metric improved by at least `min_delta`
3. If improved, the counter resets to 0
4. If not improved, the counter increments
5. When counter reaches `patience`, training stops
6. The best metric value is tracked and reported

### Best Practices

1. **Set max_epochs high** when using early stopping (e.g., 100)
2. **Use patience >= 5** to avoid stopping too early
3. **Match metric to your goal**: recall for retrieval, NDCG for ranking
4. **Combine with LR scheduling** for best results
5. **Monitor training in W&B** to see when stopping occurs

## Complete Examples

### Example 1: Full Advanced Configuration

All features enabled:

```yaml
retrieval:
  training:
    batch_size: 16384
    max_epochs: 100

    # Optimizer
    optimizer:
      name: AdamW
      weight_decay: 0.01
      betas: [0.9, 0.999]
    lr: 0.001

    # Learning rate scheduler
    scheduler:
      name: CosineAnnealingLR
      T_max: 100
      eta_min: 1e-6
      interval: epoch

    # Mixed precision training
    mixed_precision: true

    # Early stopping
    early_stopping:
      enabled: true
      metric: recall@10
      patience: 15
      mode: max
      min_delta: 0.0001

    # Other settings
    temperature: 0.05
    eval_steps: 500
    log_steps: 100
```

### Example 2: Conservative Training

For careful, monitored training:

```yaml
retrieval:
  training:
    batch_size: 4096
    max_epochs: 200

    optimizer:
      name: SGD
      momentum: 0.9
      weight_decay: 0.0001
      nesterov: true
    lr: 0.01

    scheduler:
      name: StepLR
      step_size: 30
      gamma: 0.5

    early_stopping:
      enabled: true
      metric: recall@10
      patience: 25
      mode: max
      min_delta: 0.0005  # Require clear improvement
```

### Example 3: Fast Iteration

For quick experiments:

```yaml
retrieval:
  training:
    batch_size: 8192
    max_epochs: 50

    optimizer: Adam
    lr: 0.001

    mixed_precision: true  # Speed up training

    early_stopping:
      enabled: true
      patience: 5  # Stop quickly if not improving
      metric: recall@10
      mode: max
```

## Command Line Examples

Combine all features:

```bash
python rec/retrieval/train.py --config config.yaml \
  --optimizer AdamW \
  --optimizer-args weight_decay=0.01 \
  --scheduler CosineAnnealingLR \
  --scheduler-args T_max=50 eta_min=1e-6 \
  --mixed-precision \
  --early-stopping \
  --early-stopping-metric recall@10 \
  --early-stopping-patience 15
```

## Monitoring Training

When using early stopping and LR scheduling:

1. **Check W&B logs** for `train/lr` to see learning rate changes
2. **Monitor validation metrics** to see improvement trends
3. **Look for early stopping messages** in training logs:
   ```
   Early stopping triggered after epoch 45
   Best recall@10: 0.6234
   ```

## See Also

- [config/example_advanced.yaml](../config/example_advanced.yaml) - Full example config
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html) - Official optimizer docs
- [PyTorch Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) - Official scheduler docs
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) - Automatic mixed precision docs
