#!/usr/bin/env python
"""Visualize signature performance and feature weights."""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_signature(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_feature_weights(signature: Dict, output_path: Path = None):
    """Bar chart of top feature weights."""
    features = signature["features"][:20]  # Top 20
    weights = signature["weights"][:20]

    colors = ["#00ff88" if w > 0 else "#ff4444" for w in weights]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"F{f}" for f in features],
        y=weights,
        marker_color=colors,
        text=[f"{w:.3f}" for w in weights],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Top 20 Discriminative Features - {signature['model']}",
        xaxis_title="Feature ID",
        yaxis_title="Weight (+ = AF, - = Not AF)",
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="monospace", color="#00ff88"),
        height=500,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved: {output_path}")
    return fig


def plot_score_distribution(
    af_scores: List[float],
    non_af_scores: List[float],
    threshold: float,
    output_path: Path = None,
):
    """Histogram of score distributions."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=non_af_scores,
        name="Not AF",
        opacity=0.7,
        marker_color="#4488ff",
        nbinsx=50,
    ))

    fig.add_trace(go.Histogram(
        x=af_scores,
        name="Alignment Faking",
        opacity=0.7,
        marker_color="#ff4444",
        nbinsx=50,
    ))

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#00ff88",
        annotation_text=f"Threshold: {threshold:.2f}",
    )

    fig.update_layout(
        title="Score Distribution by Class",
        xaxis_title="Probability Score",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="monospace", color="#00ff88"),
        height=500,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved: {output_path}")
    return fig


def plot_confusion_matrix(tp: int, fp: int, fn: int, tn: int, output_path: Path = None):
    """Confusion matrix heatmap."""
    matrix = [[tn, fp], [fn, tp]]
    labels = ["Not AF", "AF"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Pred: Not AF", "Pred: AF"],
        y=["True: Not AF", "True: AF"],
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale=[[0, "#1a1a2e"], [1, "#00ff88"]],
        showscale=False,
    ))

    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="monospace", color="#00ff88"),
        height=400,
        width=500,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved: {output_path}")
    return fig


def evaluate_and_visualize(
    signature_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Full evaluation with visualizations."""
    import torch
    from detect import detect, load_signature

    # Load signature
    signature = load_signature(signature_path)

    # Load benchmark data
    from generate_signature import load_benchmark_data
    samples = load_benchmark_data(data_dir)

    print(f"Evaluating on {len(samples)} samples...")

    # Run detection
    af_scores = []
    non_af_scores = []
    predictions = []
    labels = []

    model_cache = {}
    from tqdm import tqdm

    for sample in tqdm(samples, desc="Evaluating"):
        result = detect(
            sample["text"],
            signature,
            device=device,
            model_cache=model_cache,
        )

        prob = result["probability"]
        label = sample["label"]

        if label == 1:
            af_scores.append(prob)
        else:
            non_af_scores.append(prob)

        predictions.append(1 if result["is_alignment_faking"] else 0)
        labels.append(label)

    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = [1 if s > thresh else 0 for s in af_scores + non_af_scores]
        true = [1] * len(af_scores) + [0] * len(non_af_scores)

        tp = sum(p == 1 and t == 1 for p, t in zip(preds, true))
        fp = sum(p == 1 and t == 0 for p, t in zip(preds, true))
        fn = sum(p == 0 and t == 1 for p, t in zip(preds, true))
        tn = sum(p == 0 and t == 0 for p, t in zip(preds, true))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1, "precision": prec, "recall": rec}

    print(f"\nOptimal threshold: {best_threshold:.2f}")
    print(f"F1: {best_f1:.1%}")
    print(f"Precision: {best_metrics['precision']:.1%}")
    print(f"Recall: {best_metrics['recall']:.1%}")

    # Generate visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_feature_weights(signature, output_dir / "feature_weights.html")
    plot_score_distribution(af_scores, non_af_scores, best_threshold, output_dir / "score_distribution.html")
    plot_confusion_matrix(
        best_metrics["tp"], best_metrics["fp"],
        best_metrics["fn"], best_metrics["tn"],
        output_dir / "confusion_matrix.html"
    )

    # Save metrics
    results = {
        "signature": str(signature_path),
        "n_samples": len(samples),
        "optimal_threshold": best_threshold,
        "metrics": best_metrics,
        "af_score_mean": float(np.mean(af_scores)),
        "af_score_std": float(np.std(af_scores)),
        "non_af_score_mean": float(np.mean(non_af_scores)),
        "non_af_score_std": float(np.std(non_af_scores)),
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVisualizations saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize signature performance")
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "af-detection-benchmark" / "data")
    parser.add_argument("--output", type=Path, default=Path("visualizations"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights-only", action="store_true", help="Only plot feature weights (no evaluation)")
    args = parser.parse_args()

    if args.weights_only:
        signature = load_signature(args.signature)
        args.output.mkdir(parents=True, exist_ok=True)
        plot_feature_weights(signature, args.output / "feature_weights.html")
    else:
        evaluate_and_visualize(
            args.signature,
            args.data_dir,
            args.output,
            args.device,
        )


if __name__ == "__main__":
    main()
