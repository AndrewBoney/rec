"""Unit tests for model architectures."""
import pytest
import torch
from rec.common.model import TowerConfig
from rec.ranking.model import DLRM, TwoTowerRanking
from rec.retrieval.model import TwoTowerRetrieval


@pytest.mark.unit
def test_retrieval_model_forward(encoders, feature_store, device):
    """Test TwoTowerRetrieval forward pass works."""
    user_encoders, item_encoders = encoders

    # Build tower configs
    user_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.user_cat_cols,
        dense_features=feature_store.feature_cfg.user_dense_cols,
        embedding_dim=16,
        hidden_dims=[32],
    )
    item_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.item_cat_cols,
        dense_features=feature_store.feature_cfg.item_dense_cols,
        embedding_dim=16,
        hidden_dims=[32],
    )

    model = TwoTowerRetrieval(
        user_tower=user_tower,
        item_tower=item_tower,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        output_dim=16,
    ).to(device)

    # Create dummy batch
    user_features = feature_store.get_user_features(torch.tensor([0, 1, 2]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1, 2]))

    # Move to device
    user_features = {k: v.to(device) for k, v in user_features.items()}
    item_features = {k: v.to(device) for k, v in item_features.items()}

    # Test forward pass
    user_emb = model.user_tower(user_features)
    item_emb = model.item_tower(item_features)

    assert user_emb.shape == (3, 16)
    assert item_emb.shape == (3, 16)

    # Test loss computation
    loss = model.compute_loss(user_emb, item_emb)
    assert loss.item() > 0


@pytest.mark.unit
def test_ranking_two_tower_forward(encoders, feature_store, device):
    """Test TwoTowerRanking forward pass works."""
    user_encoders, item_encoders = encoders

    user_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.user_cat_cols,
        dense_features=feature_store.feature_cfg.user_dense_cols,
        embedding_dim=16,
        hidden_dims=[32],
    )
    item_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.item_cat_cols,
        dense_features=feature_store.feature_cfg.item_dense_cols,
        embedding_dim=16,
        hidden_dims=[32],
    )

    model = TwoTowerRanking(
        user_tower=user_tower,
        item_tower=item_tower,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        output_dim=16,
        scorer_hidden_dims=[32],
    ).to(device)

    # Create dummy batch
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))
    labels = torch.tensor([1.0, 0.0]).to(device)

    # Move to device
    user_features = {k: v.to(device) for k, v in user_features.items()}
    item_features = {k: v.to(device) for k, v in item_features.items()}

    # Test forward pass
    scores = model(user_features, item_features)
    assert scores.shape == (2,)

    # Test loss computation
    loss = model.compute_loss(user_features, item_features, labels)
    assert loss.item() > 0


@pytest.mark.unit
def test_dlrm_forward(encoders, feature_store, device):
    """Test DLRM forward pass works."""
    user_encoders, item_encoders = encoders

    # Combine all encoders
    all_encoders = {**user_encoders, **item_encoders}

    # All categorical features
    all_cat_features = (
        feature_store.feature_cfg.user_cat_cols + feature_store.feature_cfg.item_cat_cols
    )

    # All dense features
    all_dense_features = (
        feature_store.feature_cfg.user_dense_cols + feature_store.feature_cfg.item_dense_cols
    )

    model = DLRM(
        categorical_features=all_cat_features,
        dense_features=all_dense_features,
        encoders=all_encoders,
        embedding_dim=8,
        mlp_dims=[16, 8],
    ).to(device)

    # Create dummy batch
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))

    # Combine user and item features
    combined_features = {**user_features, **item_features}
    combined_features = {k: v.to(device) for k, v in combined_features.items()}

    labels = torch.tensor([1.0, 0.0]).to(device)

    # Test forward pass
    scores = model(combined_features)
    assert scores.shape == (2,)

    # Test loss computation
    loss = model.compute_loss(combined_features, labels)
    assert loss.item() > 0


@pytest.mark.unit
def test_model_backward_pass(encoders, feature_store, device):
    """Test that gradients flow through the model."""
    user_encoders, item_encoders = encoders

    user_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.user_cat_cols,
        dense_features=feature_store.feature_cfg.user_dense_cols,
        embedding_dim=8,
        hidden_dims=[16],
    )
    item_tower = TowerConfig(
        categorical_features=feature_store.feature_cfg.item_cat_cols,
        dense_features=feature_store.feature_cfg.item_dense_cols,
        embedding_dim=8,
        hidden_dims=[16],
    )

    model = TwoTowerRetrieval(
        user_tower=user_tower,
        item_tower=item_tower,
        user_encoders=user_encoders,
        item_encoders=item_encoders,
        output_dim=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create batch
    user_features = feature_store.get_user_features(torch.tensor([0, 1]))
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))

    user_features = {k: v.to(device) for k, v in user_features.items()}
    item_features = {k: v.to(device) for k, v in item_features.items()}

    # Forward and backward
    optimizer.zero_grad()
    user_emb = model.user_tower(user_features)
    item_emb = model.item_tower(item_features)
    loss = model.compute_loss(user_emb, item_emb)
    loss.backward()

    # Check gradients exist
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_gradients, "No gradients computed"
