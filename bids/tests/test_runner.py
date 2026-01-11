"""Tests for ProbeRunner (no torch dependency)."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Import directly to avoid torch dependency
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bids.runner import ProbeRunner
from bids.types import ProbeSignature


@pytest.fixture
def sample_signature():
    """Create a sample signature."""
    return ProbeSignature(
        id="test_sig",
        version="1.0.0",
        created_at=datetime.utcnow(),
        model_family="gpt2",
        model_name="gpt2-small",
        layer=8,
        hook_point="blocks.8.hook_resid_pre",
        sae_release="test",
        sae_id="test",
        n_sae_features=1000,
        probe_type="sae_sparse",
        feature_indices=[100, 200, 300, 400, 500],
        feature_weights=[0.8, -0.5, 0.3, -0.2, 0.1],
        intercept=-0.5,
        threshold=0.7,
    )


class TestProbeRunner:
    """Tests for ProbeRunner."""

    def test_load_probe(self, sample_signature):
        """Test loading a probe."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        assert "test_sig" in runner.loaded_probes
        assert len(runner.loaded_probes) == 1

    def test_run_probe_high_score(self, sample_signature):
        """Test running probe with high activation values."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        # Create activations with high values at signature feature indices
        activations = {8: np.zeros(1000)}
        activations[8][100] = 2.0  # High positive weight feature
        activations[8][200] = 0.1  # Negative weight feature (low activation)
        activations[8][300] = 1.5
        activations[8][400] = 0.1
        activations[8][500] = 1.0

        score, features = runner.run_probe(sample_signature, activations)

        # Score should be high due to positive weight features being activated
        assert score > 0.5
        assert len(features) > 0

    def test_run_probe_low_score(self, sample_signature):
        """Test running probe with low activation values."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        # Create activations with low/zero values
        activations = {8: np.zeros(1000)}

        score, features = runner.run_probe(sample_signature, activations)

        # Score should be low
        assert score < 0.5

    def test_run_all(self, sample_signature):
        """Test running all probes."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        activations = {8: np.random.randn(1000)}

        results = runner.run_all(activations, model_name="gpt2-small")

        assert "alignment_faking" in results
        score, features = results["alignment_faking"]
        assert 0 <= score <= 1

    def test_run_all_filters_by_model(self, sample_signature):
        """Test that run_all filters by model name."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        activations = {8: np.random.randn(1000)}

        # Run with different model - should return empty
        results = runner.run_all(activations, model_name="gemma-2-2b")
        assert len(results) == 0

    def test_get_thresholds(self, sample_signature):
        """Test getting thresholds."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        thresholds = runner.get_thresholds()
        assert "alignment_faking" in thresholds
        assert thresholds["alignment_faking"] == 0.7

    def test_clear(self, sample_signature):
        """Test clearing loaded probes."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)
        assert len(runner.loaded_probes) == 1

        runner.clear()
        assert len(runner.loaded_probes) == 0

    def test_contributing_features(self, sample_signature):
        """Test that contributing features are computed correctly."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        activations = {8: np.zeros(1000)}
        activations[8][100] = 2.0  # weight 0.8, contribution = 1.6
        activations[8][200] = 1.0  # weight -0.5, contribution = -0.5

        score, features = runner.run_probe(sample_signature, activations)

        # Should have features sorted by absolute contribution
        assert len(features) >= 2

        # Check the top feature
        top = features[0]
        assert top["feature_id"] == 100
        assert abs(top["contribution"] - 1.6) < 0.01

    def test_missing_layer(self, sample_signature):
        """Test error when layer is missing from activations."""
        runner = ProbeRunner()
        runner.load_probe(sample_signature)

        activations = {0: np.random.randn(1000)}  # Wrong layer

        with pytest.raises(ValueError, match="Layer 8 not in activations"):
            runner.run_probe(sample_signature, activations)
