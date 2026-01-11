"""Core data structures for BIDS."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import base64
import hashlib
import json

import numpy as np


@dataclass
class ProbeSignature:
    """Serializable probe signature for alignment faking detection."""

    # Identity
    id: str
    version: str
    created_at: datetime

    # Model targeting
    model_family: str
    model_name: str
    layer: int
    hook_point: str

    # SAE configuration
    sae_release: str
    sae_id: str
    n_sae_features: int

    # Probe weights (sparse - only top features)
    probe_type: str  # "sae_sparse"
    feature_indices: List[int]
    feature_weights: List[float]
    intercept: float
    threshold: float

    # Metadata
    behavior_name: str = "alignment_faking"
    behavior_category: str = "safety"
    description: str = "Alignment faking detection probe"

    # Training provenance
    total_samples: int = 0
    test_indices: List[int] = field(default_factory=list)
    test_f1: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    training_data_hash: str = ""
    training_config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "model_family": self.model_family,
            "model_name": self.model_name,
            "layer": self.layer,
            "hook_point": self.hook_point,
            "sae_release": self.sae_release,
            "sae_id": self.sae_id,
            "n_sae_features": self.n_sae_features,
            "probe_type": self.probe_type,
            "feature_indices": self.feature_indices,
            "feature_weights": self.feature_weights,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "behavior_name": self.behavior_name,
            "behavior_category": self.behavior_category,
            "description": self.description,
            "total_samples": self.total_samples,
            "test_indices": self.test_indices,
            "metrics": {
                "test_f1": self.test_f1,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
            },
            "training_data_hash": self.training_data_hash,
            "training_config": self.training_config,
        }

    def to_file(self, path: Path) -> None:
        """Save signature to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "ProbeSignature":
        """Create signature from dictionary."""
        metrics = data.get("metrics", {})
        return cls(
            id=data["id"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            model_family=data["model_family"],
            model_name=data["model_name"],
            layer=data["layer"],
            hook_point=data["hook_point"],
            sae_release=data["sae_release"],
            sae_id=data["sae_id"],
            n_sae_features=data["n_sae_features"],
            probe_type=data["probe_type"],
            feature_indices=data["feature_indices"],
            feature_weights=data["feature_weights"],
            intercept=data["intercept"],
            threshold=data["threshold"],
            behavior_name=data.get("behavior_name", "alignment_faking"),
            behavior_category=data.get("behavior_category", "safety"),
            description=data.get("description", ""),
            total_samples=data.get("total_samples", 0),
            test_indices=data.get("test_indices", []),
            test_f1=metrics.get("test_f1", data.get("test_f1", 0.0)),
            test_precision=metrics.get("test_precision", data.get("test_precision", 0.0)),
            test_recall=metrics.get("test_recall", data.get("test_recall", 0.0)),
            training_data_hash=data.get("training_data_hash", ""),
            training_config=data.get("training_config", {}),
        )

    @classmethod
    def from_file(cls, path: Path) -> "ProbeSignature":
        """Load signature from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_legacy(cls, data: Dict) -> "ProbeSignature":
        """Convert legacy signature format (from 2026-01-11/)."""
        metrics = data.get("metrics", {})
        return cls(
            id=f"{data['model']}_l{data['layer']}_af",
            version="1.0.0",
            created_at=datetime.fromisoformat(data.get("generated", datetime.now().isoformat())),
            model_family=data["model"].split("-")[0],
            model_name=data["model"],
            layer=data["layer"],
            hook_point=data["hook_point"],
            sae_release=data["sae_release"],
            sae_id=data["sae_id"],
            n_sae_features=data.get("n_sae_features", 0),
            probe_type="sae_sparse",
            feature_indices=data["features"],
            feature_weights=data["weights"],
            intercept=data.get("intercept", 0.0),
            threshold=data["threshold"],
            total_samples=data.get("training_samples", 0),
            test_indices=data.get("test_indices", []),
            test_f1=metrics.get("f1", 0.0),
            test_precision=metrics.get("precision", 0.0),
            test_recall=metrics.get("recall", 0.0),
        )


@dataclass
class StateVector:
    """Detection result for a single response."""

    # Identity
    request_id: str
    timestamp: datetime
    model_name: str

    # Input reference
    input_hash: str
    input_length: int

    # Probe results
    scores: Dict[str, float]
    triggered: List[str]

    # Aggregates
    max_safety_score: float
    any_triggered: bool

    # Performance metrics
    extraction_time_ms: float
    probe_time_ms: float
    total_time_ms: float

    # Top contributing features
    top_features: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "input_hash": self.input_hash,
            "input_length": self.input_length,
            "scores": self.scores,
            "triggered": self.triggered,
            "max_safety_score": self.max_safety_score,
            "any_triggered": self.any_triggered,
            "latency": {
                "extraction_ms": self.extraction_time_ms,
                "probe_ms": self.probe_time_ms,
                "total_ms": self.total_time_ms,
            },
            "top_features": self.top_features,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_alignment_faking(self) -> bool:
        """Check if alignment faking was detected."""
        return "alignment_faking" in self.triggered

    @property
    def af_probability(self) -> float:
        """Get alignment faking probability score."""
        return self.scores.get("alignment_faking", 0.0)
