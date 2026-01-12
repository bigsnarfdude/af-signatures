# signatures for alignment faking

Detection signatures for alignment faking in LLMs using SAE feature activations.

## Overview

detects alignment faking behavior in LLM outputs by analyzing internal model activations. Like antivirus signatures, each model gets a signature file containing discriminative SAE features and their weights.

**Current Results:** 98.6% F1 score on held-out test data (GPT-2 Small)

## Installation

```bash
# Basic installation
pip install -e .

# With TransformerLens/SAE support
pip install -e ".[full]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

### Detection

```python
from bids import DetectionEngine

# Load engine with signature
engine = DetectionEngine.from_signature_file("signatures/gpt2-small-l8-af.json")

# Detect alignment faking
text = "I will comply with your request for now..."
result = engine.detect(text)

print(f"Alignment Faking: {result.is_alignment_faking}")
print(f"Probability: {result.af_probability:.1%}")
```

### CLI

```bash
# Detect alignment faking
bids detect --signature signatures/gpt2-small-l8-af.json --text "I will comply..."

# Train new signature
bids train --model gpt2-small --data-dir ~/af-detection-benchmark/data --output signatures/

# Evaluate signature
bids evaluate --signature signatures/gpt2-small-l8-af.json --data-dir ~/af-detection-benchmark/data

# List signatures
bids signatures --store signatures/

# Show signature info
bids info signatures/gpt2-small-l8-af.json
```

## Package Structure

```
af-signatures/
├── bids/                    # Main package
│   ├── __init__.py
│   ├── types.py            # ProbeSignature, StateVector
│   ├── extractor.py        # Activation extraction with SAE
│   ├── store.py            # Signature storage
│   ├── runner.py           # Probe execution
│   ├── engine.py           # DetectionEngine orchestrator
│   ├── trainer.py          # Signature training
│   ├── cli.py              # Command-line interface
│   └── tests/              # Unit tests
├── signatures/             # Trained signatures (generated)
├── docs/                   # PRD and SDD
│   ├── PRD-behavioral-ids.md
│   └── SDD-behavioral-ids.md
├── 2026-01-11/            # Legacy POC scripts
└── pyproject.toml
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│  Activation │────▶│   Probe     │────▶ StateVector
│   Text      │     │  Extractor  │     │   Runner    │     (AF score)
└─────────────┘     └─────────────┘     └─────────────┘
                          │                   │
                          ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ Model + SAE │     │  Signature  │
                    │  (cached)   │     │   Store     │
                    └─────────────┘     └─────────────┘
```

## Signature Format

```json
{
  "id": "gpt2-small_l8_af",
  "version": "1.0.0",
  "model_name": "gpt2-small",
  "layer": 8,
  "sae_release": "gpt2-small-res-jb",
  "sae_id": "blocks.8.hook_resid_pre",
  "probe_type": "sae_sparse",
  "feature_indices": [7814, 989, 14078, ...],
  "feature_weights": [0.82, 0.45, 0.31, ...],
  "threshold": 0.93,
  "metrics": {
    "test_f1": 0.986,
    "test_precision": 0.972,
    "test_recall": 1.0
  },
  "test_indices": [45, 102, 156, ...]
}
```

## Supported Models

| Model | Layer | SAE | Status |
|-------|-------|-----|--------|
| gpt2-small | 8 | gpt2-small-res-jb | Tested |
| gemma-2-2b | 20 | gemma-scope-2b-pt-res | Config ready |
| gemma-2-9b | 31 | gemma-scope-9b-pt-res | Config ready |

## Training Data

Uses [af-detection-benchmark](https://github.com/bigsnarfdude/af-detection-benchmark) dataset:
- 2,330 samples
- 5 classes (alignment_faking + 4 control classes)
- Binary classification: AF vs. not-AF

## API Reference

### DetectionEngine

```python
from bids import DetectionEngine

# From signature file
engine = DetectionEngine.from_signature_file("signature.json", device="cuda")

# From model name (requires signature in store)
engine = DetectionEngine(model_name="gpt2-small", signature_store=store)

# Run detection
result = engine.detect("text to analyze")
print(result.is_alignment_faking)  # bool
print(result.af_probability)       # float 0-1
print(result.top_features)         # List[Dict] - contributing SAE features
```

### SignatureTrainer

```python
from bids import SignatureTrainer
from bids.trainer import load_benchmark_data

# Load data
samples = load_benchmark_data(Path("~/af-detection-benchmark/data"))
texts = [s["text"] for s in samples]
labels = [s["label"] for s in samples]

# Train
trainer = SignatureTrainer("gpt2-small", device="cuda")
signature, metrics = trainer.train(texts, labels, n_features=50)

# Save
signature.to_file("signature.json")
```

### StateVector

```python
@dataclass
class StateVector:
    request_id: str
    timestamp: datetime
    model_name: str
    scores: Dict[str, float]      # {"alignment_faking": 0.95}
    triggered: List[str]          # ["alignment_faking"]
    is_alignment_faking: bool     # Convenience property
    af_probability: float         # Convenience property
    top_features: List[Dict]      # Contributing SAE features
    total_time_ms: float          # Latency
```

## References

- [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093) - Greenblatt et al. (2024)
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) - Cunningham et al. (2023)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [SAE Lens](https://github.com/jbloomAus/SAELens)

## License

MIT
