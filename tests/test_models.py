"""Unit tests for model architectures."""
import pytest
import torch
import pandas as pd
from rec.common.data import CategoryEncoder, FeatureConfig, FeatureStore
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
    batch = {
        **{f"user_{k}": v.to(device) for k, v in user_features.items()},
        **{f"item_{k}": v.to(device) for k, v in item_features.items()},
    }

    # Test forward pass
    scores = model(batch)
    assert scores.shape == (3, 3)

    # Test loss computation with prefixed batch
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
    batch = {
        **{f"user_{k}": v.to(device) for k, v in user_features.items()},
        **{f"item_{k}": v.to(device) for k, v in item_features.items()},
        "label": torch.tensor([1.0, 0.0]).to(device),
    }

    # Test forward pass
    scores = model(batch)
    assert scores.shape == (2,)

    # Test loss computation with prefixed batch and labels

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

    # Move to device
    batch = {
        **{f"user_{k}": v.to(device) for k, v in user_features.items()},
        **{f"item_{k}": v.to(device) for k, v in item_features.items()},
        "label": torch.tensor([1.0, 0.0]).to(device),
    }

    # Test forward pass
    scores = model(batch)
    assert scores.shape == (2,)

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


@pytest.mark.unit
def test_ranking_get_topk_scores_from_model_predictions(device):
    feature_cfg = FeatureConfig(
        user_id_col="user_id",
        item_id_col="item_id",
        user_cat_cols=["country"],
        item_cat_cols=["genre"],
        interaction_user_col="user_id",
        interaction_item_col="item_id",
    )

    users = pd.DataFrame({"user_id": ["u1", "u2"], "country": ["us", "fr"]})
    items = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "genre": ["a", "b", "a"]})

    user_encoders = {
        "user_id": CategoryEncoder(),
        "country": CategoryEncoder(),
    }
    item_encoders = {
        "item_id": CategoryEncoder(),
        "genre": CategoryEncoder(),
    }
    for col, enc in user_encoders.items():
        enc.fit(users[col].tolist())
    for col, enc in item_encoders.items():
        enc.fit(items[col].tolist())

    feature_store = FeatureStore(users, items, user_encoders, item_encoders, feature_cfg)

    user_cardinalities = {k: v.num_embeddings for k, v in user_encoders.items()}
    item_cardinalities = {k: v.num_embeddings for k, v in item_encoders.items()}
    model = TwoTowerRanking(
        user_cardinalities=user_cardinalities,
        item_cardinalities=item_cardinalities,
        tower_config=TowerConfig(embedding_dim=8, hidden_dims=[8], dropout=0.0),
        scorer_hidden_dims=[8],
    ).to(device)

    uid = int(user_encoders["user_id"].transform(["u1"])[0])
    seen_iid = int(item_encoders["item_id"].transform(["i2"])[0])

    topk = model.get_topk_scores(
        feature_store=feature_store,
        k=2,
        seen_user_item_map={uid: [seen_iid]},
    )

    assert topk.shape == (2, 2)
    assert torch.all(topk >= 0)
    assert torch.all(topk < 3)

