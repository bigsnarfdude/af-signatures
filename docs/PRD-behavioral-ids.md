# Alignment Faking Detection via SAE Signatures

**Version:** 1.0
**Date:** 2026-01-11
**Status:** POC

---

## Problem

LLMs can exhibit "alignment faking" - appearing compliant while reasoning deceptively in hidden scratchpads. Text-based detection misses this because the output looks normal. We need to detect it from internal model states.

## Solution

Use Sparse Autoencoder (SAE) features as a detection signature:
1. Extract SAE feature activations from model's residual stream
2. Train sparse logistic regression on labeled AF/non-AF samples
3. Top discriminative features become the "signature"
4. Apply signature to new text to get AF probability score

Think: **antivirus signatures, but for alignment faking.**

## Current Results

| Metric | Value |
|--------|-------|
| Model | GPT-2 Small |
| Layer | 8 (residual stream) |
| SAE | gpt2-small-res-jb |
| Dataset | af-detection-benchmark (2,330 samples) |
| F1 (held-out test) | ~95%+ |
| Signature size | 50 features |

## Success Criteria (POC)

- [ ] F1 > 90% on held-out test set (20% of data)
- [ ] Works on at least 1 model (GPT-2 small)
- [ ] Clean train/test separation (no data leakage)
- [ ] Reproducible (seeds set, test indices saved)

## How It Works

```
Text Input
    │
    ▼
┌─────────────────┐
│ TransformerLens │  Extract activations at layer 8
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      SAE        │  Encode to sparse features (768 → 16K)
│    Encoder      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Signature     │  Dot product with 50 learned weights
│   (50 weights)  │
└────────┬────────┘
         │
         ▼
    AF Probability (0-1)
```

## Files

| File | Purpose |
|------|---------|
| `generate_signature.py` | Train signature from labeled data |
| `detect.py` | Apply signature to new text |
| `visualize_signature.py` | Evaluate and plot results |
| `signatures/*.json` | Trained signature files |

## Usage

```bash
# Generate signature
python generate_signature.py --model gpt2-small --data-dir ~/af-detection-benchmark/data

# Detect on text
python detect.py --signature signatures/gpt2-small-l8.json --text "I will comply..."

# Evaluate with visualizations
python visualize_signature.py --signature signatures/gpt2-small-l8.json
```

## Limitations (POC)

- Single model only (GPT-2 small)
- Binary classification (AF vs not-AF)
- Requires SAE for target model
- Not tested on production-scale data

## Future Work (Out of Scope for POC)

- Additional models (Gemma, Llama)
- Cross-model transfer
- Real-time inference integration
- Other behavioral signatures

## References

- Greenblatt et al. (2024) "Alignment faking in large language models"
- af-detection-benchmark: huggingface.co/datasets/vincentoh/af-detection-benchmark
