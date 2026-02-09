"""Unit tests for data processing functions."""
import pandas as pd
import pytest
import torch
from rec.common.data import (
    CategoryEncoder,
    DenseEncoder,
    FeatureStore,
)


@pytest.mark.unit
def test_category_encoder(dummy_data):
    """Test CategoryEncoder creates valid mappings."""
    users, items, interactions = dummy_data

    encoder = CategoryEncoder()
    encoder.fit(users["gender"])

    # Check encoding works
    encoded = encoder.transform(users["gender"])
    assert len(encoded) == len(users)
    assert all(0 <= x < encoder.num_embeddings for x in encoded)

    # Check unknown values get UNK token
    unknown_encoded = encoder.transform(pd.Series([999]))
    assert unknown_encoded[0] == encoder.unknown_index


@pytest.mark.unit
def test_dense_encoder(dummy_data):
    """Test DenseEncoder normalizes values."""
    users, items, interactions = dummy_data

    encoder = DenseEncoder()
    encoder.fit(users["age"])

    # Check normalization
    normalized = encoder.transform(users["age"])
    assert len(normalized) == len(users)

    # Check rough normalization (mean ~ 0, std ~ 1)
    assert abs(normalized.mean()) < 0.5
    assert abs(normalized.std() - 1.0) < 0.5


@pytest.mark.unit
def test_encoders_fixture(encoders, feature_config):
    """Test encoders fixture creates all required encoders."""
    user_encoders, item_encoders = encoders

    # Check user encoders
    assert feature_config.user_id_col in user_encoders
    for col in feature_config.user_cat_cols:
        assert col in user_encoders
    for col in feature_config.user_dense_cols:
        assert col in user_encoders

    # Check item encoders
    assert feature_config.item_id_col in item_encoders
    for col in feature_config.item_cat_cols:
        assert col in item_encoders
    for col in feature_config.item_dense_cols:
        assert col in item_encoders


@pytest.mark.unit
def test_feature_store(feature_store):
    """Test FeatureStore lookups work correctly."""
    # Test user lookup
    user_features = feature_store.get_user_features(torch.tensor([0, 1, 2]))
    assert len(user_features) > 0

    # Check we have all expected feature columns
    assert feature_store.feature_cfg.user_id_col in user_features
    for col in feature_store.feature_cfg.user_cat_cols:
        assert col in user_features
    for col in feature_store.feature_cfg.user_dense_cols:
        assert col in user_features

    assert len(user_features[feature_store.feature_cfg.user_id_col]) == 3

    # Test item lookup
    item_features = feature_store.get_item_features(torch.tensor([0, 1]))
    assert len(item_features) > 0

    assert feature_store.feature_cfg.item_id_col in item_features
    for col in feature_store.feature_cfg.item_cat_cols:
        assert col in item_features
    for col in feature_store.feature_cfg.item_dense_cols:
        assert col in item_features

    assert len(item_features[feature_store.feature_cfg.item_id_col]) == 2
