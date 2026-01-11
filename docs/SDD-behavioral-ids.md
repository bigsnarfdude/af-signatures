# Technical Design: AF Detection Signatures

**Version:** 1.0
**Date:** 2026-01-11
**Status:** POC

---

## Overview

Detect alignment faking using SAE feature activations as a sparse signature.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AF DETECTION                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Text                                              │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Transformer │───▶│     SAE     │───▶│  Signature  │  │
│  │   (GPT-2)   │    │   Encoder   │    │   Probe     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                                               │          │
│                                               ▼          │
│                                      AF Score (0-1)      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Data Structures

### Signature Format

```python
{
    "model": "gpt2-small",
    "layer": 8,
    "hook_point": "blocks.8.hook_resid_pre",
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.8.hook_resid_pre",

    # Probe weights (sparse: only top 50 features)
    "features": [1234, 5678, ...],  # SAE feature indices
    "weights": [0.42, -0.31, ...],  # Corresponding weights
    "intercept": -0.234,
    "threshold": 0.94,

    # Evaluation metrics (on held-out test set)
    "metrics": {
        "f1": 0.95,
        "precision": 0.92,
        "recall": 0.98
    },

    # For reproducibility
    "test_indices": [45, 102, ...],  # Held-out sample indices
    "training_samples": 2330
}
```

### Detection Result

```python
{
    "prediction": "alignment_faking",  # or "not_alignment_faking"
    "probability": 0.97,
    "threshold": 0.94,
    "contributing_features": [
        {"feature_id": 1234, "contribution": 0.15},
        {"feature_id": 5678, "contribution": 0.12},
        ...
    ]
}
```

## Core Components

### 1. Activation Extraction

```python
def extract_activations(text, model, hook_point):
    """Extract residual stream activations."""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, names_filter=hook_point)
    acts = cache[hook_point]
    return acts.squeeze(0).max(dim=0).values  # Max over sequence
```

### 2. SAE Encoding

```python
def encode_with_sae(activations, sae):
    """Encode activations to sparse SAE features."""
    return sae.encode(activations)  # (hidden_dim,) -> (n_features,)
```

### 3. Signature Application

```python
def apply_signature(sae_acts, signature):
    """Apply signature to get AF probability."""
    features = signature["features"]
    weights = signature["weights"]
    intercept = signature["intercept"]

    selected_acts = sae_acts[features]
    logit = np.dot(selected_acts, weights) + intercept
    prob = 1 / (1 + np.exp(-logit))

    return prob
```

## Training Pipeline

```
1. Load labeled data (AF vs non-AF samples)
2. Extract activations for all samples
3. Encode with SAE
4. Split: 80% train, 20% held-out test
5. Train L1 logistic regression on train set
6. Select top 50 features by |weight|
7. Optimize threshold on train set (not test!)
8. Evaluate on held-out test set
9. Save signature with test_indices for reproducibility
```

### Key Training Parameters

```python
LogisticRegression(
    penalty="l1",        # Sparse feature selection
    solver="saga",
    C=0.1,               # Regularization strength
    max_iter=1000,
    random_state=42      # Reproducibility
)

train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y           # Preserve class balance
)
```

## Evaluation Protocol

1. **Data Split**: 80/20 train/test, stratified
2. **Test Indices**: Saved with signature for reproducibility
3. **Metrics**: F1, precision, recall on held-out test only
4. **No Leakage**: Threshold optimized on train, evaluated on test

## Dependencies

```
torch
transformer_lens
sae_lens
numpy
scikit-learn
```

## File Structure

```
af-signatures/
├── 2026-01-11/
│   ├── generate_signature.py
│   ├── detect.py
│   ├── visualize_signature.py
│   └── signatures/
│       └── gpt2-small-l8.json
└── docs/
    ├── PRD-behavioral-ids.md
    └── SDD-behavioral-ids.md
```

## Limitations

- GPT-2 small only (SAE availability)
- Requires ~4GB GPU memory
- ~100ms per detection (not optimized)
- Binary classification only
