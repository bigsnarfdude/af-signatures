"""Signature storage and management."""

from pathlib import Path
from typing import Dict, List, Optional
import json

from bids.types import ProbeSignature


class SignatureStore:
    """File-based signature storage."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._index: Dict[str, Path] = {}
        self._cache: Dict[str, ProbeSignature] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan signature directory and build index."""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True)
            return

        for sig_file in self.base_path.rglob("*.json"):
            try:
                with open(sig_file) as f:
                    data = json.load(f)

                # Support both new and legacy formats
                sig_id = data.get("id")
                if not sig_id:
                    # Legacy format: construct ID from model and layer
                    model = data.get("model", "unknown")
                    layer = data.get("layer", 0)
                    sig_id = f"{model}_l{layer}_af"

                self._index[sig_id] = sig_file
            except (json.JSONDecodeError, KeyError):
                continue

    def get(self, probe_id: str) -> ProbeSignature:
        """Load a signature by ID."""
        if probe_id in self._cache:
            return self._cache[probe_id]

        if probe_id not in self._index:
            raise KeyError(f"Signature not found: {probe_id}")

        path = self._index[probe_id]
        with open(path) as f:
            data = json.load(f)

        # Handle legacy format
        if "features" in data and "feature_indices" not in data:
            sig = ProbeSignature.from_legacy(data)
        else:
            sig = ProbeSignature.from_dict(data)

        self._cache[probe_id] = sig
        return sig

    def list_all(self) -> List[str]:
        """List all available signature IDs."""
        return list(self._index.keys())

    def list_for_model(self, model_name: str) -> List[str]:
        """List signature IDs applicable to a model."""
        result = []
        for probe_id in self._index:
            sig = self.get(probe_id)
            if sig.model_name == model_name:
                result.append(probe_id)
        return result

    def add(self, signature: ProbeSignature) -> Path:
        """Add a new signature to the store."""
        # Organize by behavior category
        category_dir = self.base_path / signature.behavior_category
        category_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{signature.model_name}-l{signature.layer}-{signature.behavior_name}.json"
        path = category_dir / filename

        signature.to_file(path)
        self._index[signature.id] = path
        self._cache[signature.id] = signature

        return path

    def remove(self, probe_id: str) -> None:
        """Remove a signature from the store."""
        if probe_id not in self._index:
            raise KeyError(f"Signature not found: {probe_id}")

        path = self._index[probe_id]
        path.unlink()

        del self._index[probe_id]
        if probe_id in self._cache:
            del self._cache[probe_id]

    def reload(self) -> None:
        """Rebuild index from disk."""
        self._index.clear()
        self._cache.clear()
        self._build_index()

    def get_by_model_and_behavior(
        self,
        model_name: str,
        behavior_name: str = "alignment_faking",
    ) -> Optional[ProbeSignature]:
        """Find signature by model and behavior name."""
        for probe_id in self._index:
            sig = self.get(probe_id)
            if sig.model_name == model_name and sig.behavior_name == behavior_name:
                return sig
        return None

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, probe_id: str) -> bool:
        return probe_id in self._index

    def __iter__(self):
        return iter(self._index)
