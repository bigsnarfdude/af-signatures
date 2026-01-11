"""Tests for SignatureStore (no torch dependency)."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import directly to avoid torch dependency
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bids.store import SignatureStore
from bids.types import ProbeSignature


@pytest.fixture
def temp_store_dir():
    """Create temporary directory for signature store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_signature():
    """Create a sample signature."""
    return ProbeSignature(
        id="gpt2-small_l8_af",
        version="1.0.0",
        created_at=datetime.utcnow(),
        model_family="gpt2",
        model_name="gpt2-small",
        layer=8,
        hook_point="blocks.8.hook_resid_pre",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        n_sae_features=24576,
        probe_type="sae_sparse",
        feature_indices=[100, 200, 300],
        feature_weights=[0.5, -0.3, 0.2],
        intercept=-0.1,
        threshold=0.9,
        test_f1=0.95,
    )


class TestSignatureStore:
    """Tests for SignatureStore."""

    def test_create_empty_store(self, temp_store_dir):
        """Test creating an empty store."""
        store = SignatureStore(temp_store_dir)
        assert len(store) == 0
        assert store.list_all() == []

    def test_add_signature(self, temp_store_dir, sample_signature):
        """Test adding a signature."""
        store = SignatureStore(temp_store_dir)
        path = store.add(sample_signature)

        assert path.exists()
        assert sample_signature.id in store
        assert len(store) == 1

    def test_get_signature(self, temp_store_dir, sample_signature):
        """Test retrieving a signature."""
        store = SignatureStore(temp_store_dir)
        store.add(sample_signature)

        retrieved = store.get(sample_signature.id)
        assert retrieved.id == sample_signature.id
        assert retrieved.model_name == sample_signature.model_name
        assert retrieved.feature_indices == sample_signature.feature_indices

    def test_list_for_model(self, temp_store_dir):
        """Test listing signatures for a model."""
        store = SignatureStore(temp_store_dir)

        # Add two signatures for different models
        sig1 = ProbeSignature(
            id="gpt2-small_l8_af",
            version="1.0.0",
            created_at=datetime.utcnow(),
            model_family="gpt2",
            model_name="gpt2-small",
            layer=8,
            hook_point="blocks.8.hook_resid_pre",
            sae_release="test",
            sae_id="test",
            n_sae_features=100,
            probe_type="sae_sparse",
            feature_indices=[1, 2],
            feature_weights=[0.5, -0.3],
            intercept=0,
            threshold=0.5,
        )

        sig2 = ProbeSignature(
            id="gemma-2-2b_l20_af",
            version="1.0.0",
            created_at=datetime.utcnow(),
            model_family="gemma",
            model_name="gemma-2-2b",
            layer=20,
            hook_point="blocks.20.hook_resid_post",
            sae_release="test",
            sae_id="test",
            n_sae_features=100,
            probe_type="sae_sparse",
            feature_indices=[1, 2],
            feature_weights=[0.5, -0.3],
            intercept=0,
            threshold=0.5,
        )

        store.add(sig1)
        store.add(sig2)

        gpt2_sigs = store.list_for_model("gpt2-small")
        assert len(gpt2_sigs) == 1
        assert "gpt2-small_l8_af" in gpt2_sigs

    def test_remove_signature(self, temp_store_dir, sample_signature):
        """Test removing a signature."""
        store = SignatureStore(temp_store_dir)
        store.add(sample_signature)

        store.remove(sample_signature.id)
        assert sample_signature.id not in store
        assert len(store) == 0

    def test_reload_index(self, temp_store_dir, sample_signature):
        """Test reloading index from disk."""
        store = SignatureStore(temp_store_dir)
        store.add(sample_signature)

        # Create new store instance
        store2 = SignatureStore(temp_store_dir)
        assert sample_signature.id in store2

    def test_get_by_model_and_behavior(self, temp_store_dir, sample_signature):
        """Test finding signature by model and behavior."""
        store = SignatureStore(temp_store_dir)
        store.add(sample_signature)

        found = store.get_by_model_and_behavior("gpt2-small", "alignment_faking")
        assert found is not None
        assert found.id == sample_signature.id

        not_found = store.get_by_model_and_behavior("gpt2-small", "sycophancy")
        assert not_found is None

    def test_legacy_format_loading(self, temp_store_dir):
        """Test loading legacy format signatures."""
        legacy_data = {
            "model": "gpt2-small",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.8.hook_resid_pre",
            "hook_point": "blocks.8.hook_resid_pre",
            "layer": 8,
            "n_features": 50,
            "features": [100, 200, 300],
            "weights": [0.5, -0.3, 0.2],
            "intercept": -0.1,
            "threshold": 0.9,
            "metrics": {"f1": 0.95, "precision": 0.92, "recall": 0.98},
            "generated": "2026-01-11T12:00:00",
        }

        sig_path = temp_store_dir / "legacy.json"
        with open(sig_path, "w") as f:
            json.dump(legacy_data, f)

        store = SignatureStore(temp_store_dir)
        # Legacy format should be indexed with constructed ID
        sig = store.get("gpt2-small_l8_af")
        assert sig.model_name == "gpt2-small"
        assert sig.feature_indices == [100, 200, 300]
