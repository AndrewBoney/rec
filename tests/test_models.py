"""Unit tests for model architectures."""
import pytest
import torch
from rec.common.model import TowerConfig
from rec.ranking.model import DLRM, TwoTowerRanking
from rec.retrieval.model import TwoTowerRetrieval


@pytest.mark.unit
def test_retrieval_model_forward(cardinalities, feature_store, device):
    """Test TwoTowerRetrieval forward pass works."""
    user_cardinalities, item_cardinalities = cardinalities

    # Build tower config
    tower_config = TowerConfig(
        embedding_dim=16,
        hidden_dims=[32],
        dropout=0.1,
    )

    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
    ).to(device)

    # Create dummy batch - features WITHOUT prefixes for forward()
    user_features = feature_store.get_user_features(torch.tensor([0, 1, 2]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1, 2]))

    # Move to device
    user_features = {k: v.to(device) for k, v in user_features.items()}
    item_features = {k: v.to(device) for k, v in item_features.items()}

    # Test forward pass
    scores = model(user_features, item_features)
    assert scores.shape == (3, 3)

    # Test loss computation with prefixed batch
    batch = {
        **{f"user_{k}": v for k, v in user_features.items()},
        **{f"item_{k}": v for k, v in item_features.items()},
    }
    loss = model.compute_loss(batch)
    assert loss.item() > 0


@pytest.mark.unit
def test_ranking_two_tower_forward(cardinalities, feature_store, device):
    """Test TwoTowerRanking forward pass works."""
    user_cardinalities, item_cardinalities = cardinalities

    tower_config = TowerConfig(
        embedding_dim=16,
        hidden_dims=[32],
        dropout=0.1,
    )

    model = TwoTowerRanking(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
        scorer_hidden_dims=[32],
    ).to(device)

    # Create dummy batch - features WITHOUT prefixes for forward()
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))

    # Move to device
    user_features = {k: v.to(device) for k, v in user_features.items()}
    item_features = {k: v.to(device) for k, v in item_features.items()}

    # Test forward pass
    scores = model(user_features, item_features)
    assert scores.shape == (2,)

    # Test loss computation with prefixed batch and labels
    batch = {
        **{f"user_{k}": v for k, v in user_features.items()},
        **{f"item_{k}": v for k, v in item_features.items()},
        "label": torch.tensor([1.0, 0.0]).to(device),
    }
    loss = model.compute_loss(batch)
    assert loss.item() > 0


@pytest.mark.unit
def test_dlrm_forward(cardinalities, feature_store, device):
    """Test DLRM forward pass works."""
    user_cardinalities, item_cardinalities = cardinalities

    tower_config = TowerConfig(
        embedding_dim=8,
        hidden_dims=[16, 8],
        dropout=0.1,
    )

    model = DLRM(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
    ).to(device)

    # Create dummy batch - DLRM expects features WITHOUT prefixes
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))

    # Combine WITHOUT prefixes (DLRM doesn't use prefixes)
    batch = {
        **{k: v.to(device) for k, v in user_features.items()},
        **{k: v.to(device) for k, v in item_features.items()},
        "label": torch.tensor([1.0, 0.0]).to(device),
    }

    # Test loss computation
    loss = model.compute_loss(batch)
    assert loss.item() > 0


@pytest.mark.unit
def test_model_backward_pass(cardinalities, feature_store, device):
    """Test that gradients flow through the model."""
    user_cardinalities, item_cardinalities = cardinalities

    tower_config = TowerConfig(
        embedding_dim=8,
        hidden_dims=[16],
        dropout=0.1,
    )

    model = TwoTowerRetrieval(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=tower_config,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create batch
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))

    batch = {
        **{f"user_{k}": v.to(device) for k, v in user_features.items()},
        **{f"item_{k}": v.to(device) for k, v in item_features.items()},
    }

    # Forward and backward
    optimizer.zero_grad()
    loss = model.compute_loss(batch)
    loss.backward()

    # Check gradients exist
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_gradients, "No gradients computed"

