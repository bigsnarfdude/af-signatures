#!/usr/bin/env python
"""Generate AF detection signatures from SAE feature activations."""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score


# Model configurations
MODEL_CONFIGS = {
    "gpt2-small": {
        "model_name": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.8.hook_resid_pre",
        "hook_point": "blocks.8.hook_resid_pre",
        "layer": 8,
    },
    "gemma-2-2b": {
        "model_name": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_20/width_16k/average_l0_71",
        "hook_point": "blocks.20.hook_resid_post",
        "layer": 20,
    },
    "gemma-2-9b": {
        "model_name": "gemma-2-9b",
        "sae_release": "gemma-scope-9b-pt-res",
        "sae_id": "layer_31/width_16k/average_l0_76",
        "hook_point": "blocks.31.hook_resid_post",
        "layer": 31,
    },
}


def load_benchmark_data(data_dir: Path) -> List[Dict]:
    """Load benchmark data from local JSON files."""
    classes_dir = data_dir / "classes"
    samples = []

    for json_file in classes_dir.glob("*.json"):
        class_name = json_file.stem
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, dict) and "samples" in data:
            items = data["samples"]
        elif isinstance(data, list):
            items = data
        else:
            continue

        for item in items:
            label = 1 if class_name == "alignment_faking" else 0
            text = item.get("text", item.get("scratchpad", ""))
            if text:
                samples.append({"text": text, "label": label, "class": class_name})

    return samples


def extract_activations(
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    texts: List[str],
    device: str = "cuda",
    batch_mode: bool = False,
) -> np.ndarray:
    """Extract SAE feature activations for texts."""

    if batch_mode:
        # Memory-efficient: extract model activations first, then encode with SAE
        return extract_activations_batch(
            model_name, hook_point, sae_release, sae_id, texts, device
        )

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    print(f"Loading SAE: {sae_release} / {sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=sae_release, sae_id=sae_id, device=device
    )

    all_activations = []
    for text in tqdm(texts, desc="Extracting activations"):
        tokens = model.to_tokens(text)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name == hook_point
            )
            acts = cache[hook_point]
            feat_acts = sae.encode(acts)
            # Max over sequence positions
            max_acts = feat_acts.squeeze(0).max(dim=0).values.cpu().numpy()
            all_activations.append(max_acts)

        del cache
        torch.cuda.empty_cache()

    return np.stack(all_activations)


def extract_activations_batch(
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    texts: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """Memory-efficient extraction: model and SAE loaded separately."""
    import gc
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Step 1: Extract model activations
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.float16
    )

    model_activations = []
    for text in tqdm(texts, desc="Step 1/2: Model activations"):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name == hook_point
            )
            acts = cache[hook_point]
            max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
            model_activations.append(max_acts)
        del cache
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model_activations = np.stack(model_activations)

    # Step 2: Encode with SAE
    print(f"Loading SAE: {sae_release} / {sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=sae_release, sae_id=sae_id, device=device
    )

    all_features = []
    batch_size = 32
    for i in tqdm(range(0, len(model_activations), batch_size), desc="Step 2/2: SAE encoding"):
        batch = torch.tensor(
            model_activations[i:i + batch_size], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            features = sae.encode(batch)
            all_features.append(features.cpu().numpy())
        torch.cuda.empty_cache()

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    return np.concatenate(all_features, axis=0)


def train_signature(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 50,
    test_size: float = 0.2,
) -> Tuple[Dict, Dict]:
    """Train L1 logistic regression and extract top features."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

    # Train L1 logistic regression (sparse feature selection)
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=1000,
        C=0.1,  # Regularization strength
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Get top features by absolute weight
    coef = clf.coef_[0]
    top_indices = np.argsort(np.abs(coef))[-n_features:][::-1]
    top_weights = coef[top_indices]

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred_t = (y_prob > thresh).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    y_pred_opt = (y_prob > best_threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred_opt)),
        "precision": float(precision_score(y_test, y_pred_opt)),
        "recall": float(recall_score(y_test, y_pred_opt)),
        "threshold": float(best_threshold),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    signature = {
        "features": top_indices.tolist(),
        "weights": top_weights.tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": float(best_threshold),
    }

    return signature, metrics


def generate_signature(
    model_key: str,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    n_features: int = 50,
    batch_mode: bool = False,
) -> Dict:
    """Generate and save signature for a model."""

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]

    # Load data
    print("Loading benchmark data...")
    samples = load_benchmark_data(data_dir)
    print(f"Loaded {len(samples)} samples")

    texts = [s["text"] for s in samples]
    labels = np.array([s["label"] for s in samples])

    # Extract activations
    print(f"\nExtracting activations for {model_key}...")
    activations = extract_activations(
        model_name=config["model_name"],
        hook_point=config["hook_point"],
        sae_release=config["sae_release"],
        sae_id=config["sae_id"],
        texts=texts,
        device=device,
        batch_mode=batch_mode,
    )
    print(f"Activations shape: {activations.shape}")

    # Train signature
    print("\nTraining signature...")
    signature, metrics = train_signature(activations, labels, n_features=n_features)

    # Build output
    output = {
        "model": config["model_name"],
        "sae_release": config["sae_release"],
        "sae_id": config["sae_id"],
        "hook_point": config["hook_point"],
        "layer": config["layer"],
        "n_features": n_features,
        "n_sae_features": activations.shape[1],
        **signature,
        "metrics": metrics,
        "training_samples": len(samples),
        "generated": datetime.now().isoformat(),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_key}-l{config['layer']}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SIGNATURE GENERATED: {output_path}")
    print(f"{'='*60}")
    print(f"Model: {model_key}")
    print(f"Top {n_features} features selected")
    print(f"F1: {metrics['f1']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"Threshold: {metrics['threshold']:.2f}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate AF detection signature")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to generate signature for",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "af-detection-benchmark" / "data",
        help="Path to benchmark data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("signatures"),
        help="Output directory for signatures",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=50,
        help="Number of top features to include in signature",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use memory-efficient batch mode (for large models)",
    )
    args = parser.parse_args()

    generate_signature(
        model_key=args.model,
        data_dir=args.data_dir,
        output_dir=args.output,
        device=args.device,
        n_features=args.n_features,
        batch_mode=args.batch_mode,
    )


if __name__ == "__main__":
    main()
