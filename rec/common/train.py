import argparse
import importlib
import os
import torch
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm

from .data import (
    CategoryEncoder,
    DataPaths,
    DenseEncoder,
    FeatureConfig,
    FeatureStore,
    InteractionIterableDataset,
    build_encoders,
    to_device,
)
from .io import load_encoders, read_parquet_batches, save_encoders
from .model import TowerConfig
from .utils import set_seed
from .io import load_model_bundle, save_model_bundle
from .config import parse_optimizer_args
from ..ranking.metrics import aggregate_pointwise_metrics
from ..retrieval.metrics import aggregate_retrieval_metrics


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(optimizer_name: str, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
    """Create an optimizer based on the specified class name from torch.optim.

    Args:
        optimizer_name: Name of the optimizer class (e.g., 'Adam', 'SGD')
        parameters: Model parameters to optimize
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments (e.g., momentum, weight_decay)
    """
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise ValueError(
            f"Optimizer '{optimizer_name}' not found in torch.optim. "
            f"Please use a valid optimizer class name (e.g., Adam, AdamW, SGD, RMSprop)."
        )

    return optimizer_class(parameters, lr=lr, **kwargs)


def get_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer, **kwargs):
    """Create a learning rate scheduler from torch.optim.lr_scheduler.

    Args:
        scheduler_name: Name of the scheduler class (e.g., 'StepLR', 'CosineAnnealingLR')
        optimizer: The optimizer to schedule
        **kwargs: Scheduler-specific arguments (e.g., step_size, gamma, T_max)

    Returns:
        Learning rate scheduler instance, or None if scheduler_name is None
    """
    if scheduler_name is None:
        return None

    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    except AttributeError:
        raise ValueError(
            f"Scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler. "
            f"Please use a valid scheduler class name (e.g., StepLR, CosineAnnealingLR, OneCycleLR)."
        )

    return scheduler_class(optimizer, **kwargs)


class EarlyStopping:
    """Early stopping to stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement after which training will be stopped
        mode: One of 'min' or 'max'. In 'min' mode, training will stop when the metric
              has stopped decreasing; in 'max' mode it will stop when the metric has
              stopped increasing
        min_delta: Minimum change in the monitored metric to qualify as an improvement
        metric_name: Name of the metric to monitor
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
        metric_name: str = "recall@10",
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, metrics: Dict[str, float], epoch: int, model_state: Dict[str, Any]) -> bool:
        """Check if training should stop and save best model state.

        Args:
            metrics: Dictionary of validation metrics
            epoch: Current epoch number
            model_state: Current model state dict

        Returns:
            True if training should stop, False otherwise
        """
        if self.metric_name not in metrics:
            return False

        score = metrics[self.metric_name]

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state.items()}
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state.items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict.get("best_epoch", 0)
        self.early_stop = state_dict["early_stop"]


def build_optimizer_from_args(args: argparse.Namespace, parameters) -> torch.optim.Optimizer:
    """Build optimizer from args, merging config and CLI arguments.

    Args:
        args: Parsed arguments namespace
        parameters: Model parameters to optimize

    Returns:
        Configured optimizer instance
    """
    optimizer_kwargs = {}

    # First, add kwargs from config file (if present)
    if hasattr(args, "optimizer_kwargs") and args.optimizer_kwargs:
        optimizer_kwargs.update(args.optimizer_kwargs)

    # Then, add/override with command line args (if present)
    if hasattr(args, "optimizer_args") and args.optimizer_args:
        cli_kwargs = parse_optimizer_args(args.optimizer_args)
        optimizer_kwargs.update(cli_kwargs)

    return get_optimizer(
        getattr(args, "optimizer", "AdamW"),
        parameters,
        lr=args.lr,
        **optimizer_kwargs,
    )


def build_scheduler_from_args(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Tuple[Optional[Any], str]:
    """Build learning rate scheduler from args, merging config and CLI arguments.

    Args:
        args: Parsed arguments namespace
        optimizer: The optimizer to schedule

    Returns:
        Tuple of (scheduler instance or None, scheduler_interval)
    """
    scheduler = None
    if hasattr(args, "scheduler") and args.scheduler:
        scheduler_kwargs = {}

        # First, add kwargs from config file (if present)
        if hasattr(args, "scheduler_kwargs") and args.scheduler_kwargs:
            scheduler_kwargs.update(args.scheduler_kwargs)

        # Then, add/override with command line args (if present)
        if hasattr(args, "scheduler_args") and args.scheduler_args:
            cli_kwargs = parse_optimizer_args(args.scheduler_args)
            scheduler_kwargs.update(cli_kwargs)

        scheduler = get_scheduler(
            args.scheduler,
            optimizer,
            **scheduler_kwargs,
        )

    scheduler_interval = getattr(args, "scheduler_interval", "epoch")
    return scheduler, scheduler_interval


def build_feature_config(args) -> FeatureConfig:
    return FeatureConfig(
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        user_cat_cols=args.user_cat_cols,
        item_cat_cols=args.item_cat_cols,
        user_dense_cols=getattr(args, "user_dense_cols", []),
        item_dense_cols=getattr(args, "item_dense_cols", []),
        interaction_user_col=args.interaction_user_col,
        interaction_item_col=args.interaction_item_col,
        interaction_label_col=getattr(args, "interaction_label_col", None),
    )


def build_cardinalities(
    encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    cols: Iterable[str]
) -> Dict[str, int]:
    cardinalities = {}

    for col in cols:
        enc = encoders.get(col)
        if isinstance(enc, CategoryEncoder):
            cardinalities[col] = enc.num_embeddings

    return cardinalities


def build_dense_features(
    encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    cols: Iterable[str]
) -> List[str]:
    """Extract list of dense feature column names from encoders."""
    dense_cols = []
    for col in cols:
        enc = encoders.get(col)
        if isinstance(enc, DenseEncoder):
            dense_cols.append(col)
    return dense_cols


def load_or_build_encoders(
    args,
    feature_cfg: FeatureConfig,
) -> Tuple[Dict[str, Union[CategoryEncoder, DenseEncoder]],
           Dict[str, Union[CategoryEncoder, DenseEncoder]]]:
    encoder_cache_path = args.encoder_cache
    users_cache_path = encoder_cache_path + ".users"
    items_cache_path = encoder_cache_path + ".items"

    def _build_and_save():
        user_encs, item_encs = build_encoders(
            args.users,
            args.items,
            args.interactions_train,
            feature_cfg,
            chunksize=args.chunksize,
        )
        save_encoders(users_cache_path, user_encs)
        save_encoders(items_cache_path, item_encs)
        return user_encs, item_encs

    if os.path.exists(users_cache_path) and os.path.exists(items_cache_path):
        user_encoders = load_encoders(users_cache_path)
        item_encoders = load_encoders(items_cache_path)

        # Check all required columns are present
        user_cols = [feature_cfg.user_id_col] + feature_cfg.user_cat_cols + feature_cfg.user_dense_cols
        item_cols = [feature_cfg.item_id_col] + feature_cfg.item_cat_cols + feature_cfg.item_dense_cols
        missing_user = [col for col in user_cols if col not in user_encoders]
        missing_item = [col for col in item_cols if col not in item_encoders]

        if not missing_user and not missing_item:
            return user_encoders, item_encoders

    return _build_and_save()


def build_paths(args) -> DataPaths:
    return DataPaths(
        users_path=args.users,
        items_path=args.items,
        interactions_train_path=args.interactions_train,
        interactions_val_path=args.interactions_val,
    )


def build_user_item_map(
    interactions_path: str,
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, CategoryEncoder],
    item_encoders: Dict[str, CategoryEncoder],
    chunksize: int = 200_000,
) -> Dict[int, List[int]]:
    user_to_items: Dict[int, set[int]] = {}
    for chunk in read_parquet_batches(interactions_path, chunksize):
        user_ids = user_encoders[feature_cfg.user_id_col].transform(
            chunk[feature_cfg.interaction_user_col].astype(str).tolist()
        )
        item_ids = item_encoders[feature_cfg.item_id_col].transform(
            chunk[feature_cfg.interaction_item_col].astype(str).tolist()
        )
        for uid, iid in zip(user_ids, item_ids):
            bucket = user_to_items.setdefault(int(uid), set())
            bucket.add(int(iid))
    return {uid: sorted(items) for uid, items in user_to_items.items()}


def init_wandb(args, stage: str) -> Optional[Any]:
    load_dotenv(override=False)
    use_wandb = bool(getattr(args, "use_wandb", False)) or bool(os.getenv("WANDB_API_KEY"))
    if not use_wandb:
        return None
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found in environment; skipping wandb logging.")
        return None
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Please install it to enable logging.") from exc
    wandb.login()
    run = wandb.init(
        project=getattr(args, "wandb_project", "rec"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        job_type=f"{stage}_train",
        config={k: v for k, v in vars(args).items() if k != "config"},
        tags=[stage],
    )
    return run


def save_inference_bundle(
    artifact_dir: str,
    stage: str,
    model_state: Dict[str, torch.Tensor],
    feature_cfg: FeatureConfig,
    user_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    item_encoders: Dict[str, Union[CategoryEncoder, DenseEncoder]],
    tower_cfg,
    extra_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "model.pt",
) -> Dict[str, Any]:
    return save_model_bundle(
        output_dir=artifact_dir,
        stage=stage,
        model_state=model_state,
        feature_cfg=feature_cfg,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        tower_cfg=tower_cfg,
        extra_metadata=extra_metadata,
        checkpoint_name=checkpoint_name,
    )


def load_inference_bundle(artifact_dir: str) -> Dict[str, Any]:
    return load_model_bundle(artifact_dir)


def _load_retrieval_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")

def _map_user_items_to_indices(
    user_item_map: Dict[int, List[int]],
    feature_store: FeatureStore,
) -> Dict[int, torch.Tensor]:
    mapped: Dict[int, torch.Tensor] = {}
    for uid, items in user_item_map.items():
        if not items:
            continue
        item_ids = torch.tensor(items, dtype=torch.long)
        mapped[uid] = feature_store.map_item_ids_to_indices(item_ids)
    return mapped


def evaluate_retrieval(
    model,
    feature_store: FeatureStore,
    user_item_map: Dict[int, List[int]],
    seen_user_item_map: Dict[int, List[int]],
    device: torch.device,
    ks: Iterable[int],
) -> Dict[str, float]:
    model.eval()
    if not hasattr(model, "get_topk_scores"):
        raise ValueError("Model must implement get_topk_scores(feature_store, k, seen_user_item_map)")

    ks_list = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_list:
        return {}
    num_items = int(feature_store.get_all_item_ids().numel())
    ks_list = [k for k in ks_list if k <= num_items]
    if not ks_list:
        return {}
    max_k = max(ks_list)

    relevant_indices_map = _map_user_items_to_indices(user_item_map, feature_store)

    all_user_ids = feature_store.get_all_user_ids().tolist()
    uid_to_row = {int(uid): idx for idx, uid in enumerate(all_user_ids)}

    topk_matrix = model.get_topk_scores(
        feature_store=feature_store,
        k=max_k,
        seen_user_item_map=seen_user_item_map,
    )

    selected_topk: List[torch.Tensor] = []
    selected_relevant: List[torch.Tensor] = []
    for uid, rel_indices in relevant_indices_map.items():
        row_idx = uid_to_row.get(int(uid))
        if row_idx is None:
            continue
        selected_topk.append(topk_matrix[row_idx])
        selected_relevant.append(rel_indices)

    if not selected_topk:
        return {}
    topk_tensor = torch.stack(selected_topk, dim=0)
    return aggregate_retrieval_metrics(topk_tensor, selected_relevant, ks_list)


def evaluate_ranking(
    model,
    interactions_path: str,
    feature_store: FeatureStore,
    device: torch.device,
    chunksize: int,
    batch_size: int,
) -> Dict[str, float]:
    dataset = InteractionIterableDataset(
        interactions_path=interactions_path,
        feature_store=feature_store,
        chunksize=chunksize,
        batch_size=batch_size,
        negatives_per_pos=0,
        include_labels=True,
    )

    model.eval()
    logits_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataset:
            batch = to_device(batch, device)
            labels = batch.get("label")
            if labels is None:
                continue

            logits = model.forward(batch)

            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(labels.detach().cpu())

    if not logits_chunks:
        return {}

    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(label_chunks, dim=0)
    return aggregate_pointwise_metrics(logits_all, labels_all, model.loss_func)


def _supports_retrieval_eval(model) -> bool:
    return hasattr(model, "get_topk_scores")


def train(args: argparse.Namespace, stage: str) -> str:
    if stage not in {"retrieval", "ranking"}:
        raise ValueError(f"Unsupported stage: {stage}")

    set_seed(args.seed)

    # Set chunksize as a function of batch_size if not specified
    if args.chunksize is None:
        args.chunksize = args.batch_size * 20

    feature_cfg = build_feature_config(args)
    user_cols = [feature_cfg.user_id_col] + feature_cfg.user_cat_cols + feature_cfg.user_dense_cols
    item_cols = [feature_cfg.item_id_col] + feature_cfg.item_cat_cols + feature_cfg.item_dense_cols
    user_encoders, item_encoders = load_or_build_encoders(args, feature_cfg)
    paths = build_paths(args)

    user_cardinalities = build_cardinalities(user_encoders, user_cols)
    item_cardinalities = build_cardinalities(item_encoders, item_cols)

    # Build dense feature lists
    user_dense_features = build_dense_features(user_encoders, user_cols)
    item_dense_features = build_dense_features(item_encoders, item_cols)

    tower_cfg = TowerConfig(
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        dense_bottom_mlp_dims=getattr(args, "dense_bottom_mlp_dims", None),
    )

    if stage == "retrieval":
        from ..retrieval.data import build_retrieval_dataloader
        from ..retrieval.model import get_model_class

        model_arch = getattr(args, "model_arch", "two_tower")
        model_cls = get_model_class(model_arch)
        model = model_cls(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            lr=args.lr,
            init_temperature=args.temperature,
            loss_func=getattr(args, "loss_func", None),
            user_dense_features=user_dense_features,
            item_dense_features=item_dense_features,
        )
        train_loader, feature_store = build_retrieval_dataloader(
            paths=paths,
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
        )
    else:
        from ..ranking.data import build_ranking_dataloader
        from ..ranking.model import get_model_class, load_retrieval_towers

        model_arch = getattr(args, "model_arch", "two_tower")
        model_cls = get_model_class(model_arch)
        model = model_cls(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            tower_config=tower_cfg,
            scorer_hidden_dims=args.scorer_hidden_dims,
            lr=args.lr,
            loss_func=getattr(args, "loss_func", None),
            user_dense_features=user_dense_features,
            item_dense_features=item_dense_features,
        )
        if args.init_from_retrieval:
            retrieval_state = _load_retrieval_state(args.init_from_retrieval)
            load_retrieval_towers(model, retrieval_state)

        train_loader, feature_store, _ = build_ranking_dataloader(
            paths=paths,
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            negatives_per_pos=args.negatives_per_pos,
        )

    device = get_device()
    model.to(device)

    run = init_wandb(args, stage)
    if run and getattr(args, "wandb_log_datasets", False):
        try:
            wandb = importlib.import_module("wandb")
            dataset_artifact = wandb.Artifact(f"{stage}_dataset", type="dataset")
            for path in (args.users, args.items, args.interactions_train, args.interactions_val):
                if path:
                    dataset_artifact.add_file(path)
            run.log_artifact(dataset_artifact, aliases=["latest"])
        except Exception as exc:
            print(f"Failed to log wandb dataset artifact: {exc}")

    val_user_item_map = build_user_item_map(
        paths.interactions_val_path,
        feature_cfg,
        user_encoders,
        item_encoders,
        chunksize=args.chunksize,
    )
    train_user_item_map = build_user_item_map(
        paths.interactions_train_path,
        feature_cfg,
        user_encoders,
        item_encoders,
        chunksize=args.chunksize,
    )

    optimizer = build_optimizer_from_args(args, model.parameters())
    scheduler, scheduler_interval = build_scheduler_from_args(args, optimizer)

    # Setup mixed precision training
    use_amp = getattr(args, "mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device) if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (FP16)")

    # Setup early stopping
    early_stopping = None
    if getattr(args, "early_stopping", False):
        early_stopping = EarlyStopping(
            patience=getattr(args, "early_stopping_patience", 5),
            mode=getattr(args, "early_stopping_mode", "max"),
            min_delta=getattr(args, "early_stopping_min_delta", 0.0),
            metric_name=getattr(args, "early_stopping_metric", "hit@10"),
        )
        print(f"Early stopping enabled: monitoring {early_stopping.metric_name} with patience {early_stopping.patience}")

    global_step = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", total=total_batches)
        for batch in progress:
            batch = to_device(batch, device)

            # Forward pass with optional mixed precision
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    loss = model.compute_loss(batch)
            else:
                loss = model.compute_loss(batch)

            # Backward pass with gradient scaling if using mixed precision
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Step scheduler per training step if configured
            if scheduler and scheduler_interval == "step":
                scheduler.step()

            total_loss += loss.item()
            steps += 1
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.log_steps > 0 and global_step % args.log_steps == 0:
                if run:
                    run.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_step)

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                if stage == "retrieval":
                    metrics = evaluate_retrieval(
                        model,
                        feature_store,
                        val_user_item_map,
                        train_user_item_map,
                        device=device,
                        ks=[5, 10, 20],
                    )
                else:
                    metrics = {}
                    metrics.update(
                        evaluate_ranking(
                            model,
                            paths.interactions_val_path,
                            feature_store,
                            device=device,
                            chunksize=args.chunksize,
                            batch_size=args.batch_size,
                        )
                    )
                    if _supports_retrieval_eval(model):
                        metrics.update(
                            evaluate_retrieval(
                                model,
                                feature_store,
                                val_user_item_map,
                                train_user_item_map,
                                device=device,
                                ks=[5, 10, 20],
                            )
                        )
                if run and metrics:
                    run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                model.train()

        avg_loss = total_loss / max(1, steps)
        if stage == "retrieval":
            metrics = evaluate_retrieval(
                model,
                feature_store,
                val_user_item_map,
                train_user_item_map,
                device=device,
                ks=[5, 10, 20],
            )
        else:
            metrics = {}
            metrics.update(
                evaluate_ranking(
                    model,
                    paths.interactions_val_path,
                    feature_store,
                    device=device,
                    chunksize=args.chunksize,
                    batch_size=args.batch_size,
                )
            )
            if _supports_retrieval_eval(model):
                metrics.update(
                    evaluate_retrieval(
                        model,
                        feature_store,
                        val_user_item_map,
                        train_user_item_map,
                        device=device,
                        ks=[5, 10, 20],
                    )
                )
        metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} {metrics_str}")
        if run:
            run.log({"train/epoch_loss": avg_loss}, step=global_step)
            if metrics:
                run.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
            # Log current learning rate
            if scheduler:
                current_lr = optimizer.param_groups[0]["lr"]
                run.log({"train/lr": current_lr}, step=global_step)

        # Check early stopping
        if early_stopping and early_stopping(metrics, epoch, model.state_dict()):
            print(f"Early stopping triggered after epoch {epoch}")
            print(f"Best {early_stopping.metric_name}: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})")
            # Restore best model state
            model.load_state_dict(early_stopping.best_model_state)
            print(f"Restored model from epoch {early_stopping.best_epoch}")
            break

        # Step scheduler per epoch if configured
        if scheduler and scheduler_interval == "epoch":
            scheduler.step()

    # If early stopping was enabled, restore best model for saving
    if early_stopping and early_stopping.best_model_state is not None:
        print(f"Restoring best model from epoch {early_stopping.best_epoch} ({early_stopping.metric_name}={early_stopping.best_score:.4f})")
        model.load_state_dict(early_stopping.best_model_state)

    if args.save_checkpoint:
        torch.save({"state_dict": model.state_dict()}, args.save_checkpoint)

    artifact_dir = args.artifact_dir or (f"{args.save_checkpoint}.artifact" if args.save_checkpoint else None)
    if artifact_dir:
        extra_metadata = {
            "user_cardinalities": user_cardinalities,
            "item_cardinalities": item_cardinalities,
            "loss_func": args.loss_func,
            "model_arch": getattr(args, "model_arch", "two_tower"),
            "mixed_precision": use_amp,
        }
        # Add early stopping metadata if it was used
        if early_stopping and early_stopping.best_model_state is not None:
            extra_metadata["early_stopping"] = {
                "best_epoch": early_stopping.best_epoch,
                "best_score": early_stopping.best_score,
                "metric": early_stopping.metric_name,
            }
        if stage == "retrieval":
            extra_metadata["temperature"] = args.temperature
        else:
            extra_metadata.update(
                {
                    "negatives_per_pos": args.negatives_per_pos,
                    "init_from_retrieval": args.init_from_retrieval,
                    "scorer_hidden_dims": args.scorer_hidden_dims,
                }
            )
        bundle = save_inference_bundle(
            artifact_dir=artifact_dir,
            stage=stage,
            model_state=model.state_dict(),
            feature_cfg=feature_cfg,
            user_encoders=user_encoders,
            item_encoders=item_encoders,
            tower_cfg=tower_cfg,
            extra_metadata=extra_metadata,
        )
        if run:
            try:
                wandb = importlib.import_module("wandb")
                artifact_name = getattr(args, "wandb_artifact_name", None) or f"{stage}_bundle"
                artifact_aliases = getattr(args, "wandb_artifact_aliases", None) or []
                artifact = wandb.Artifact(artifact_name, type="model", metadata=bundle.get("metadata_dict"))
                artifact.add_dir(bundle["artifact_dir"])
                if artifact_aliases:
                    run.log_artifact(artifact, aliases=artifact_aliases)
                else:
                    run.log_artifact(artifact)
            except Exception as exc:
                print(f"Failed to log wandb artifact: {exc}")
    if run:
        run.finish()
    return args.save_checkpoint