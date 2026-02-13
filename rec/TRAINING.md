# Training Configuration Guide

This guide covers all available training options including optimizer configuration, learning rate scheduling, and mixed precision training.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Optimizer Configuration](#optimizer-configuration)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Mixed Precision Training](#mixed-precision-training)
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

**Adam**:
```yaml
optimizer:
  name: Adam
  weight_decay: 0.0
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

**RMSprop**:
```yaml
optimizer:
  name: RMSprop
  alpha: 0.99
  momentum: 0.9
```

### Command Line Override

You can override optimizer settings from the command line:

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

**StepLR** - Reduce LR every N epochs:
```yaml
scheduler:
  name: StepLR
  step_size: 10  # Reduce every 10 epochs
  gamma: 0.5     # Multiply LR by 0.5
  interval: epoch
```

**CosineAnnealingLR** - Cosine annealing:
```yaml
scheduler:
  name: CosineAnnealingLR
  T_max: 50       # Number of epochs
  eta_min: 1e-6   # Minimum learning rate
  interval: epoch
```

**ExponentialLR** - Exponential decay:
```yaml
scheduler:
  name: ExponentialLR
  gamma: 0.95  # Multiply LR by 0.95 each epoch
  interval: epoch
```

**OneCycleLR** - One cycle policy (requires per-step updates):
```yaml
scheduler:
  name: OneCycleLR
  max_lr: 0.01
  total_steps: 10000  # Total training steps
  interval: step  # MUST be "step" for OneCycleLR
```

**ReduceLROnPlateau** - Reduce on metric plateau:
```yaml
scheduler:
  name: ReduceLROnPlateau
  mode: min
  factor: 0.1
  patience: 5
  interval: epoch
```

**CosineAnnealingWarmRestarts** - Warm restarts:
```yaml
scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 5         # Restart every 5 epochs
  T_mult: 2      # Double period after each restart
  eta_min: 1e-6
  interval: epoch
```

### Command Line Override

```bash
# Add scheduler via CLI
python rec/retrieval/train.py --config config.yaml \
  --scheduler StepLR \
  --scheduler-args step_size=10 gamma=0.5 \
  --scheduler-interval epoch
```

### Learning Rate Logging

When using a scheduler, the current learning rate is automatically logged to Weights & Biases under `train/lr`.

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

### Requirements

- CUDA-capable GPU
- Modern GPU architecture (Volta, Turing, Ampere, or newer)
- PyTorch with CUDA support

### Benefits

- **~2x faster training** on modern GPUs
- **~50% less memory usage** - allows larger batch sizes
- **Automatic gradient scaling** - prevents underflow
- **Automatic fallback** - safely ignored on CPU

### Notes

- Mixed precision is automatically disabled on CPU
- No code changes needed - works with all model architectures
- Gradient scaling is handled automatically by PyTorch AMP

## Complete Examples

### Example 1: Simple Configuration

Basic training with default optimizer:

```yaml
retrieval:
  training:
    batch_size: 1024
    max_epochs: 10
    lr: 0.001
    optimizer: AdamW
```

### Example 2: Advanced Optimizer

Custom optimizer settings:

```yaml
retrieval:
  training:
    batch_size: 8192
    max_epochs: 20
    optimizer:
      name: AdamW
      lr: 0.001
      weight_decay: 0.01
      betas: [0.9, 0.999]
```

### Example 3: With LR Scheduler

Combined optimizer and scheduler:

```yaml
retrieval:
  training:
    batch_size: 16384
    max_epochs: 50
    optimizer:
      name: AdamW
      weight_decay: 0.01
    lr: 0.001
    scheduler:
      name: CosineAnnealingLR
      T_max: 50
      eta_min: 1e-6
      interval: epoch
```

### Example 4: Mixed Precision + Advanced Settings

Full configuration with all features:

```yaml
retrieval:
  training:
    batch_size: 32768
    max_epochs: 30

    # Optimizer
    optimizer:
      name: AdamW
      weight_decay: 0.01
      betas: [0.9, 0.999]
    lr: 0.001

    # Learning rate scheduler
    scheduler:
      name: CosineAnnealingLR
      T_max: 30
      eta_min: 1e-6
      interval: epoch

    # Mixed precision training
    mixed_precision: true

    # Other settings
    temperature: 0.05
    eval_steps: 500
    log_steps: 100
```

### Example 5: SGD with Momentum

For comparison with Adam-family optimizers:

```yaml
ranking:
  training:
    batch_size: 8192
    max_epochs: 100

    # SGD with momentum
    optimizer:
      name: SGD
      momentum: 0.9
      weight_decay: 0.0001
      nesterov: true
    lr: 0.01

    # Step-wise LR decay
    scheduler:
      name: StepLR
      step_size: 30
      gamma: 0.1
      interval: epoch

    mixed_precision: true
```

## Command Line Examples

Override config settings from the command line:

```bash
# Change optimizer
python rec/retrieval/train.py --config config.yaml --optimizer Adam

# Change optimizer with args
python rec/retrieval/train.py --config config.yaml \
  --optimizer SGD \
  --optimizer-args momentum=0.9 weight_decay=0.0001

# Add scheduler
python rec/retrieval/train.py --config config.yaml \
  --scheduler StepLR \
  --scheduler-args step_size=10 gamma=0.5

# Enable mixed precision
python rec/retrieval/train.py --config config.yaml --mixed-precision

# Combine all
python rec/retrieval/train.py --config config.yaml \
  --optimizer AdamW \
  --optimizer-args weight_decay=0.01 \
  --scheduler CosineAnnealingLR \
  --scheduler-args T_max=50 eta_min=1e-6 \
  --mixed-precision
```

## Best Practices

1. **Start Simple**: Begin with `AdamW` and default parameters
2. **Tune Learning Rate**: Use LR scheduling for longer training runs
3. **Use Mixed Precision**: Enable on modern GPUs for free speedup
4. **Monitor LR**: Check `train/lr` in W&B when using schedulers
5. **Batch Size**: Increase batch size when using mixed precision
6. **Weight Decay**: Typically 0.01 for AdamW, 0.0001 for SGD

## See Also

- [config/example_advanced.yaml](../config/example_advanced.yaml) - Full example config
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html) - Official optimizer docs
- [PyTorch Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) - Official scheduler docs
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) - Automatic mixed precision docs
