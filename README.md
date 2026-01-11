# AF-Signatures

Detection signatures for alignment faking using SAE feature activations.

## Concept

Like antivirus signatures, but for detecting alignment faking in LLM outputs. Each model/SAE combination gets its own signature file containing the discriminative features and their weights.

## Structure

```
af-signatures/
├── YYYY-MM-DD/              # Date-based iterations
│   ├── signatures/          # Trained signature files
│   │   ├── gpt2-small-l8.json
│   │   └── gemma-2-2b-l20.json
│   ├── generate_signature.py
│   ├── detect.py
│   └── results.json
└── README.md
```

## Signature Format

```json
{
  "model": "gpt2-small",
  "sae_release": "gpt2-small-res-jb",
  "sae_id": "blocks.8.hook_resid_pre",
  "layer": 8,
  "n_features": 50,
  "features": [7814, 989, 14078, ...],
  "weights": [0.82, 0.45, 0.31, ...],
  "threshold": 0.5,
  "training_samples": 1000,
  "f1_score": 0.72,
  "generated": "2026-01-11T12:00:00"
}
```

## Usage

```bash
# Generate signature for a model
python generate_signature.py --model gpt2-small --output signatures/

# Detect AF in text
python detect.py --signature signatures/gpt2-small-l8.json --text "..."
```

## Training Data

Uses af-detection-benchmark dataset (2,330 samples, 5 classes).
