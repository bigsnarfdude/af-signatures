"""BIDS - Behavioral Intrusion Detection System for LLMs.

Focused on alignment faking detection using SAE feature activations.
"""

__version__ = "0.1.0"

# Core types always available (no torch dependency)
from bids.types import ProbeSignature, StateVector
from bids.store import SignatureStore
from bids.runner import ProbeRunner

# Torch-dependent modules use lazy imports
_torch_available = None


def _check_torch():
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def __getattr__(name):
    """Lazy import torch-dependent modules."""
    if name == "DetectionEngine":
        if not _check_torch():
            raise ImportError(
                "DetectionEngine requires torch. Install with: pip install torch"
            )
        from bids.engine import DetectionEngine
        return DetectionEngine

    elif name == "ActivationExtractor":
        if not _check_torch():
            raise ImportError(
                "ActivationExtractor requires torch. Install with: pip install torch"
            )
        from bids.extractor import ActivationExtractor
        return ActivationExtractor

    elif name == "TransformerLensExtractor":
        if not _check_torch():
            raise ImportError(
                "TransformerLensExtractor requires torch. Install with: pip install torch"
            )
        from bids.extractor import TransformerLensExtractor
        return TransformerLensExtractor

    elif name == "SignatureTrainer":
        if not _check_torch():
            raise ImportError(
                "SignatureTrainer requires torch. Install with: pip install torch"
            )
        from bids.trainer import SignatureTrainer
        return SignatureTrainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProbeSignature",
    "StateVector",
    "DetectionEngine",
    "ActivationExtractor",
    "TransformerLensExtractor",
    "SignatureStore",
    "ProbeRunner",
    "SignatureTrainer",
]
