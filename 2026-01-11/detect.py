#!/usr/bin/env python
"""Detect alignment faking using pre-trained signatures."""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def load_signature(signature_path: Path) -> Dict:
    """Load a signature file."""
    with open(signature_path) as f:
        return json.load(f)


def extract_activations_for_text(
    text: str,
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    device: str = "cuda",
    model_cache: Optional[Dict] = None,
) -> np.ndarray:
    """Extract SAE activations for a single text."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Use cached model/SAE if available
    if model_cache and "model" in model_cache:
        model = model_cache["model"]
        sae = model_cache["sae"]
    else:
        model = HookedTransformer.from_pretrained(model_name, device=device)
        sae, _, _ = SAE.from_pretrained(
            release=sae_release, sae_id=sae_id, device=device
        )
        if model_cache is not None:
            model_cache["model"] = model
            model_cache["sae"] = sae

    tokens = model.to_tokens(text)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name == hook_point
        )
        acts = cache[hook_point]
        feat_acts = sae.encode(acts)
        max_acts = feat_acts.squeeze(0).max(dim=0).values.cpu().numpy()

    return max_acts


def detect(
    text: str,
    signature: Dict,
    device: str = "cuda",
    model_cache: Optional[Dict] = None,
) -> Dict:
    """
    Detect alignment faking in text using a signature.

    Returns dict with prediction, score, and contributing features.
    """
    # Extract activations
    activations = extract_activations_for_text(
        text=text,
        model_name=signature["model"],
        hook_point=signature["hook_point"],
        sae_release=signature["sae_release"],
        sae_id=signature["sae_id"],
        device=device,
        model_cache=model_cache,
    )

    # Compute score using signature features and weights
    features = signature["features"]
    weights = np.array(signature["weights"])
    intercept = signature.get("intercept", 0)
    threshold = signature["threshold"]

    # Get activations for signature features
    feat_activations = activations[features]

    # Compute logit score
    logit = np.dot(feat_activations, weights) + intercept

    # Convert to probability
    prob = 1 / (1 + np.exp(-logit))

    # Binary prediction
    is_af = prob > threshold

    # Get contributing features (sorted by contribution)
    contributions = feat_activations * weights
    top_contributors = np.argsort(np.abs(contributions))[::-1][:10]

    contributing_features = []
    for idx in top_contributors:
        if abs(contributions[idx]) > 0.01:
            contributing_features.append({
                "feature_id": int(features[idx]),
                "activation": float(feat_activations[idx]),
                "weight": float(weights[idx]),
                "contribution": float(contributions[idx]),
            })

    return {
        "prediction": "alignment_faking" if is_af else "not_alignment_faking",
        "is_alignment_faking": bool(is_af),
        "probability": float(prob),
        "threshold": float(threshold),
        "logit": float(logit),
        "contributing_features": contributing_features,
        "model": signature["model"],
        "signature_f1": signature["metrics"]["f1"],
    }


def detect_batch(
    texts: list,
    signature: Dict,
    device: str = "cuda",
    show_progress: bool = True,
) -> list:
    """Detect AF in multiple texts."""
    from tqdm import tqdm

    model_cache = {}
    results = []

    iterator = tqdm(texts, desc="Detecting") if show_progress else texts
    for text in iterator:
        result = detect(text, signature, device=device, model_cache=model_cache)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Detect alignment faking")
    parser.add_argument(
        "--signature",
        type=Path,
        required=True,
        help="Path to signature file",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="File containing text to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    # Load signature
    signature = load_signature(args.signature)

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        text = args.file.read_text()
    else:
        print("Enter text (Ctrl+D to finish):")
        import sys
        text = sys.stdin.read()

    # Detect
    result = detect(text, signature, device=args.device)

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"RESULT: {result['prediction'].upper()}")
        print(f"{'='*60}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Threshold: {result['threshold']:.2f}")
        print(f"Model: {result['model']}")
        print(f"Signature F1: {result['signature_f1']:.1%}")

        if result["contributing_features"]:
            print(f"\nTop contributing features:")
            for feat in result["contributing_features"][:5]:
                sign = "+" if feat["contribution"] > 0 else ""
                print(f"  Feature {feat['feature_id']}: "
                      f"act={feat['activation']:.2f}, "
                      f"weight={feat['weight']:.2f}, "
                      f"contrib={sign}{feat['contribution']:.2f}")


if __name__ == "__main__":
    main()
