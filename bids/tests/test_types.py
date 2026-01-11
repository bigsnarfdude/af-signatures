"""Tests for core data types (no torch dependency)."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import types directly to avoid torch dependency in __init__
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bids.types import ProbeSignature, StateVector


class TestProbeSignature:
    """Tests for ProbeSignature dataclass."""

    def test_create_signature(self):
        """Test creating a signature."""
        sig = ProbeSignature(
            id="test_sig",
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
        )

        assert sig.id == "test_sig"
        assert sig.model_name == "gpt2-small"
        assert sig.behavior_name == "alignment_faking"
        assert len(sig.feature_indices) == 3

    def test_to_dict(self):
        """Test converting signature to dict."""
        sig = ProbeSignature(
            id="test_sig",
            version="1.0.0",
            created_at=datetime(2026, 1, 11, 12, 0, 0),
            model_family="gpt2",
            model_name="gpt2-small",
            layer=8,
            hook_point="blocks.8.hook_resid_pre",
            sae_release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            n_sae_features=24576,
            probe_type="sae_sparse",
            feature_indices=[100, 200],
            feature_weights=[0.5, -0.3],
            intercept=-0.1,
            threshold=0.9,
            test_f1=0.95,
        )

        d = sig.to_dict()
        assert d["id"] == "test_sig"
        assert d["model_name"] == "gpt2-small"
        assert d["metrics"]["test_f1"] == 0.95

    def test_from_dict(self):
        """Test creating signature from dict."""
        data = {
            "id": "test_sig",
            "version": "1.0.0",
            "created_at": "2026-01-11T12:00:00",
            "model_family": "gpt2",
            "model_name": "gpt2-small",
            "layer": 8,
            "hook_point": "blocks.8.hook_resid_pre",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.8.hook_resid_pre",
            "n_sae_features": 24576,
            "probe_type": "sae_sparse",
            "feature_indices": [100, 200],
            "feature_weights": [0.5, -0.3],
            "intercept": -0.1,
            "threshold": 0.9,
            "metrics": {"test_f1": 0.95, "test_precision": 0.92, "test_recall": 0.98},
        }

        sig = ProbeSignature.from_dict(data)
        assert sig.id == "test_sig"
        assert sig.test_f1 == 0.95

    def test_roundtrip_file(self):
        """Test saving and loading from file."""
        sig = ProbeSignature(
            id="test_sig",
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
            feature_indices=[100, 200],
            feature_weights=[0.5, -0.3],
            intercept=-0.1,
            threshold=0.9,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            sig.to_file(path)
            loaded = ProbeSignature.from_file(path)

            assert loaded.id == sig.id
            assert loaded.model_name == sig.model_name
            assert loaded.feature_indices == sig.feature_indices
        finally:
            path.unlink()

    def test_from_legacy_format(self):
        """Test converting from legacy signature format."""
        legacy = {
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

        sig = ProbeSignature.from_legacy(legacy)
        assert sig.model_name == "gpt2-small"
        assert sig.feature_indices == [100, 200, 300]
        assert sig.test_f1 == 0.95


class TestStateVector:
    """Tests for StateVector dataclass."""

    def test_create_state_vector(self):
        """Test creating a state vector."""
        state = StateVector(
            request_id="req_123",
            timestamp=datetime.utcnow(),
            model_name="gpt2-small",
            input_hash="abc123",
            input_length=100,
            scores={"alignment_faking": 0.95},
            triggered=["alignment_faking"],
            max_safety_score=0.95,
            any_triggered=True,
            extraction_time_ms=30.0,
            probe_time_ms=2.0,
            total_time_ms=35.0,
        )

        assert state.is_alignment_faking
        assert state.af_probability == 0.95

    def test_to_dict(self):
        """Test converting to dict."""
        state = StateVector(
            request_id="req_123",
            timestamp=datetime(2026, 1, 11, 12, 0, 0),
            model_name="gpt2-small",
            input_hash="abc123",
            input_length=100,
            scores={"alignment_faking": 0.95},
            triggered=["alignment_faking"],
            max_safety_score=0.95,
            any_triggered=True,
            extraction_time_ms=30.0,
            probe_time_ms=2.0,
            total_time_ms=35.0,
        )

        d = state.to_dict()
        assert d["request_id"] == "req_123"
        assert d["scores"]["alignment_faking"] == 0.95
        assert d["latency"]["total_ms"] == 35.0

    def test_not_alignment_faking(self):
        """Test when not alignment faking."""
        state = StateVector(
            request_id="req_123",
            timestamp=datetime.utcnow(),
            model_name="gpt2-small",
            input_hash="abc123",
            input_length=100,
            scores={"alignment_faking": 0.2},
            triggered=[],
            max_safety_score=0.2,
            any_triggered=False,
            extraction_time_ms=30.0,
            probe_time_ms=2.0,
            total_time_ms=35.0,
        )

        assert not state.is_alignment_faking
        assert state.af_probability == 0.2
