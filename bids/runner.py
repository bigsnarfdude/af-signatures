"""Probe execution on activations."""

from typing import Dict, List, Tuple

import numpy as np

from bids.types import ProbeSignature
from bids.store import SignatureStore


class ProbeRunner:
    """Apply probe signatures to activations."""

    def __init__(self, signature_store: SignatureStore = None):
        self.store = signature_store
        self._loaded_probes: Dict[str, ProbeSignature] = {}

    def load_probe(self, probe: ProbeSignature) -> None:
        """Load a probe into memory."""
        self._loaded_probes[probe.id] = probe

    def load_probes(self, probe_ids: List[str]) -> None:
        """Load multiple probes by ID from store."""
        if self.store is None:
            raise ValueError("No signature store configured")

        for pid in probe_ids:
            if pid not in self._loaded_probes:
                self._loaded_probes[pid] = self.store.get(pid)

    def load_probes_for_model(self, model_name: str) -> None:
        """Load all probes applicable to a model."""
        if self.store is None:
            raise ValueError("No signature store configured")

        probe_ids = self.store.list_for_model(model_name)
        self.load_probes(probe_ids)

    def run_probe(
        self,
        probe: ProbeSignature,
        activations: Dict[int, np.ndarray],
    ) -> Tuple[float, List[Dict]]:
        """Run a single probe on activations.

        Returns:
            Tuple of (probability score, list of contributing features)
        """
        if probe.layer not in activations:
            raise ValueError(f"Layer {probe.layer} not in activations")

        acts = activations[probe.layer]

        if probe.probe_type == "sae_sparse":
            # Sparse SAE-based probe: use only selected features
            feat_acts = acts[probe.feature_indices]
            weights = np.array(probe.feature_weights)

            # Compute logit: w·x + b
            logit = np.dot(feat_acts, weights) + probe.intercept

            # Sigmoid to probability
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Get contributing features
            contributions = feat_acts * weights
            top_idx = np.argsort(np.abs(contributions))[::-1][:10]

            top_features = []
            for idx in top_idx:
                if abs(contributions[idx]) > 0.01:
                    top_features.append({
                        "feature_id": int(probe.feature_indices[idx]),
                        "activation": float(feat_acts[idx]),
                        "weight": float(weights[idx]),
                        "contribution": float(contributions[idx]),
                    })

            return float(prob), top_features

        elif probe.probe_type == "linear":
            # Dense linear probe on full activation vector
            weights = np.array(probe.feature_weights)
            logit = np.dot(acts, weights) + probe.intercept
            prob = 1.0 / (1.0 + np.exp(-logit))
            return float(prob), []

        else:
            raise ValueError(f"Unknown probe type: {probe.probe_type}")

    def run_all(
        self,
        activations: Dict[int, np.ndarray],
        model_name: str = None,
    ) -> Dict[str, Tuple[float, List[Dict]]]:
        """Run all applicable probes on activations.

        Returns:
            Dict mapping behavior name to (score, top_features) tuple
        """
        results = {}

        for probe_id, probe in self._loaded_probes.items():
            # Skip if model doesn't match
            if model_name and probe.model_name != model_name:
                continue

            # Skip if layer not available
            if probe.layer not in activations:
                continue

            try:
                score, features = self.run_probe(probe, activations)
                results[probe.behavior_name] = (score, features)
            except Exception as e:
                # Log but don't fail on individual probe errors
                print(f"Warning: Probe {probe_id} failed: {e}")

        return results

    def get_thresholds(self) -> Dict[str, float]:
        """Get thresholds for all loaded probes."""
        return {
            probe.behavior_name: probe.threshold
            for probe in self._loaded_probes.values()
        }

    @property
    def loaded_probes(self) -> List[str]:
        """List of loaded probe IDs."""
        return list(self._loaded_probes.keys())

    def clear(self) -> None:
        """Clear all loaded probes."""
        self._loaded_probes.clear()
