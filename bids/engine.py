"""Main detection engine orchestrator."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import time
import uuid

from bids.types import ProbeSignature, StateVector
from bids.extractor import TransformerLensExtractor, get_extractor, MODEL_CONFIGS
from bids.store import SignatureStore
from bids.runner import ProbeRunner


class DetectionEngine:
    """Main orchestrator for alignment faking detection."""

    def __init__(
        self,
        model_name: str,
        signature_store: SignatureStore = None,
        signature: ProbeSignature = None,
        device: str = "cuda",
        batch_mode: bool = False,
    ):
        """Initialize detection engine.

        Args:
            model_name: Model key (e.g., "gpt2-small", "gemma-2-2b")
            signature_store: Store to load signatures from
            signature: Single signature to use (alternative to store)
            device: Device for inference
            batch_mode: Use memory-efficient batch mode for large models
        """
        self.model_name = model_name
        self.device = device
        self._batch_mode = batch_mode

        # Validate model
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        # Create extractor
        self.extractor = get_extractor(model_name, device=device, batch_mode=batch_mode)

        # Set up probe runner
        self.probe_runner = ProbeRunner(signature_store)

        # Load signature(s)
        if signature:
            self.probe_runner.load_probe(signature)
        elif signature_store:
            self.probe_runner.load_probes_for_model(model_name)

        # Get thresholds
        self._thresholds = self.probe_runner.get_thresholds()

    @classmethod
    def from_signature_file(
        cls,
        signature_path: Path,
        device: str = "cuda",
        batch_mode: bool = False,
    ) -> "DetectionEngine":
        """Create engine from a signature file."""
        import json

        with open(signature_path) as f:
            data = json.load(f)

        # Handle legacy format
        if "features" in data and "feature_indices" not in data:
            sig = ProbeSignature.from_legacy(data)
        else:
            sig = ProbeSignature.from_dict(data)

        return cls(
            model_name=sig.model_name,
            signature=sig,
            device=device,
            batch_mode=batch_mode,
        )

    def detect(self, text: str, request_id: str = None) -> StateVector:
        """Run full detection pipeline on text.

        Args:
            text: Input text to analyze
            request_id: Optional request identifier

        Returns:
            StateVector with detection results
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract activations
        t0 = time.time()
        activations = self.extractor.extract(text)
        extraction_time = (time.time() - t0) * 1000

        # Run probes
        t0 = time.time()
        results = self.probe_runner.run_all(activations, self.model_name)
        probe_time = (time.time() - t0) * 1000

        # Build scores dict and find top features
        scores = {}
        all_features = []
        for behavior, (score, features) in results.items():
            scores[behavior] = score
            all_features.extend(features)

        # Determine triggered behaviors
        triggered = [
            behavior
            for behavior, score in scores.items()
            if score > self._thresholds.get(behavior, 0.5)
        ]

        # Get top contributing features across all probes
        all_features.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_features = all_features[:10]

        # Build state vector
        state = StateVector(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            model_name=self.model_name,
            input_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            input_length=len(text.split()),
            scores=scores,
            triggered=triggered,
            max_safety_score=max(scores.values()) if scores else 0.0,
            any_triggered=len(triggered) > 0,
            extraction_time_ms=extraction_time,
            probe_time_ms=probe_time,
            total_time_ms=(time.time() - start_time) * 1000,
            top_features=top_features,
        )

        return state

    def detect_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[StateVector]:
        """Batch detection for offline analysis."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(texts, desc="Detecting") if show_progress else texts

        for i, text in enumerate(iterator):
            result = self.detect(text, request_id=f"batch_{i}")
            results.append(result)

        return results

    def load(self) -> None:
        """Pre-load model and SAE into memory."""
        self.extractor.load()

    def unload(self) -> None:
        """Unload model and SAE to free memory."""
        self.extractor.unload()

    @property
    def signature_info(self) -> Dict:
        """Get info about loaded signatures."""
        info = {}
        for probe_id in self.probe_runner.loaded_probes:
            probe = self.probe_runner._loaded_probes[probe_id]
            info[probe_id] = {
                "model": probe.model_name,
                "behavior": probe.behavior_name,
                "layer": probe.layer,
                "threshold": probe.threshold,
                "test_f1": probe.test_f1,
            }
        return info


def detect(
    text: str,
    model: str = "gpt2-small",
    signature_path: Path = None,
    device: str = "cuda",
) -> StateVector:
    """Convenience function for one-shot detection.

    Args:
        text: Text to analyze
        model: Model name
        signature_path: Path to signature file
        device: Device for inference

    Returns:
        StateVector with detection results
    """
    if signature_path:
        engine = DetectionEngine.from_signature_file(signature_path, device=device)
    else:
        engine = DetectionEngine(model_name=model, device=device)

    return engine.detect(text)
