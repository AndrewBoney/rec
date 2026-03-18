"""Unit tests for data processing functions."""
import pandas as pd
import pytest
import torch
from rec.common.data import (
    Tokenizer,
    DenseEncoder,
    FeatureStore,
)


@pytest.mark.unit
def test_category_encoder(dummy_data):
    """Test Tokenizer creates valid mappings."""
    users, items, interactions = dummy_data

    encoder = Tokenizer(min_freq=1)
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
def test_grouped_category_encoder():
    """Test Tokenizer with min_freq groups low-frequency values into tail index."""
    enc = Tokenizer(min_freq=3)

    # Fit: "a"=5 times, "b"=2 times, "c"=3 times
    enc.fit(["a"] * 5 + ["b"] * 2 + ["c"] * 3)

    # "a" and "c" meet threshold; "b" does not
    assert enc.num_embeddings == len(enc.mapping) + 2  # 0=OOV, 1..N=head, N+1=tail
    assert "a" in enc.mapping
    assert "c" in enc.mapping
    assert "b" not in enc.mapping

    encoded = enc.transform(["a", "b", "c", "unknown"])
    # "a" and "c" → unique head indices
    assert encoded[0] == enc.mapping["a"]
    assert encoded[2] == enc.mapping["c"]
    # "b" → tail index
    assert encoded[1] == enc.tail_index
    # "unknown" (OOV) → OOV index
    assert encoded[3] == enc.unknown_index


@pytest.mark.unit
def test_grouped_category_encoder_serialization():
    """Test Tokenizer round-trips through to_dict / from_dict."""
    enc = Tokenizer(min_freq=2)
    enc.fit(["x"] * 3 + ["y"] * 1)
    d = enc.to_dict()
    assert d["type"] == "tokenizer"

    enc2 = Tokenizer.from_dict(d)
    assert enc2.min_freq == enc.min_freq
    assert enc2.tail_index == enc.tail_index
    assert enc2.num_embeddings == enc.num_embeddings

    original = enc.transform(["x", "y", "z"])
    restored = enc2.transform(["x", "y", "z"])
    assert list(original) == list(restored)


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
    # Test user lookup by raw string IDs
    user_features = feature_store.get_user_features(["0", "1", "2"])
    assert len(user_features) > 0

    # Check we have all expected feature columns
    assert feature_store.feature_cfg.user_id_col in user_features
    for col in feature_store.feature_cfg.user_cat_cols:
        assert col in user_features
    for col in feature_store.feature_cfg.user_dense_cols:
        assert col in user_features

    assert len(user_features[feature_store.feature_cfg.user_id_col]) == 3

    # Test item lookup
    item_features = feature_store.get_item_features(["0", "1"])
    assert len(item_features) > 0

    assert feature_store.feature_cfg.item_id_col in item_features
    for col in feature_store.feature_cfg.item_cat_cols:
        assert col in item_features
    for col in feature_store.feature_cfg.item_dense_cols:
        assert col in item_features

    assert len(item_features[feature_store.feature_cfg.item_id_col]) == 2


@pytest.mark.unit
def test_feature_store_positional_lookup(dummy_data, feature_config, encoders):
    """Test that each user gets their own feature row, even with a tail-bucketing Tokenizer."""
    from rec.common.data import FeatureStore

    users, items, _ = dummy_data
    user_encoders, item_encoders = encoders

    # Replace user_id encoder with a Tokenizer that groups most users into the tail
    grouped_enc = Tokenizer(min_freq=10_000)  # all users become tail
    grouped_enc.fit(users["user_id"].astype(str).tolist())
    user_encoders_grouped = dict(user_encoders)
    user_encoders_grouped["user_id"] = grouped_enc

    store = FeatureStore(users, items, user_encoders_grouped, item_encoders, feature_config)

    # Even though all users share the same user_id embedding index,
    # each should get their own feature row (different age/gender/country).
    feats_0 = store.get_user_features([str(users["user_id"].iloc[0])])
    feats_1 = store.get_user_features([str(users["user_id"].iloc[1])])

    # Positions must differ (one user per row)
    pos_0 = store.get_user_position(str(users["user_id"].iloc[0]))
    pos_1 = store.get_user_position(str(users["user_id"].iloc[1]))
    assert pos_0 != pos_1
    assert pos_0 != 0
    assert pos_1 != 0
