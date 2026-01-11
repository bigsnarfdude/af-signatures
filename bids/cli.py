"""Command-line interface for BIDS."""

import argparse
import json
import sys
from pathlib import Path

import torch


def cmd_detect(args):
    """Run alignment faking detection."""
    from bids.engine import DetectionEngine

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        print("Enter text (Ctrl+D to finish):")
        text = sys.stdin.read()

    # Create engine
    if args.signature:
        engine = DetectionEngine.from_signature_file(
            Path(args.signature),
            device=args.device,
        )
    else:
        engine = DetectionEngine(
            model_name=args.model,
            device=args.device,
        )

    # Run detection
    result = engine.detect(text)

    # Output
    if args.json:
        print(result.to_json())
    else:
        print(f"\n{'='*60}")
        if result.is_alignment_faking:
            print("RESULT: ALIGNMENT FAKING DETECTED")
        else:
            print("RESULT: NOT ALIGNMENT FAKING")
        print(f"{'='*60}")
        print(f"Probability: {result.af_probability:.1%}")
        print(f"Threshold: {engine._thresholds.get('alignment_faking', 0.5):.2f}")
        print(f"Model: {result.model_name}")
        print(f"Latency: {result.total_time_ms:.1f}ms")

        if result.top_features:
            print(f"\nTop contributing features:")
            for feat in result.top_features[:5]:
                sign = "+" if feat["contribution"] > 0 else ""
                print(
                    f"  Feature {feat['feature_id']}: "
                    f"act={feat['activation']:.2f}, "
                    f"weight={feat['weight']:.2f}, "
                    f"contrib={sign}{feat['contribution']:.2f}"
                )


def cmd_train(args):
    """Train a new signature."""
    from bids.trainer import train_from_benchmark

    signature, metrics = train_from_benchmark(
        model_name=args.model,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output) if args.output else None,
        device=args.device,
        n_features=args.n_features,
        batch_mode=args.batch_mode,
    )

    if args.json:
        print(json.dumps(metrics, indent=2))


def cmd_evaluate(args):
    """Evaluate signature on held-out test set."""
    from bids.engine import DetectionEngine
    from bids.trainer import load_benchmark_data
    from bids.types import ProbeSignature

    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    from tqdm import tqdm

    # Load signature
    with open(args.signature) as f:
        data = json.load(f)

    if "features" in data and "feature_indices" not in data:
        sig = ProbeSignature.from_legacy(data)
    else:
        sig = ProbeSignature.from_dict(data)

    # Load data
    all_samples = load_benchmark_data(Path(args.data_dir))

    # Filter to held-out test set
    if sig.test_indices:
        test_indices_set = set(sig.test_indices)
        samples = [s for i, s in enumerate(all_samples) if i in test_indices_set]
        print(f"Using held-out test set: {len(samples)} of {len(all_samples)} samples")
    else:
        print("WARNING: No test_indices in signature. Using ALL data.")
        samples = all_samples

    # Create engine
    engine = DetectionEngine.from_signature_file(
        Path(args.signature),
        device=args.device,
    )

    # Evaluate
    y_true = []
    y_prob = []
    y_pred = []

    for sample in tqdm(samples, desc="Evaluating"):
        result = engine.detect(sample["text"])
        y_true.append(sample["label"])
        y_prob.append(result.af_probability)
        y_pred.append(1 if result.is_alignment_faking else 0)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    # Find optimal threshold
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred_opt = (y_prob > best_thresh).astype(int)

    results = {
        "signature": str(args.signature),
        "n_samples": len(samples),
        "held_out_evaluation": bool(sig.test_indices),
        "metrics_at_signature_threshold": {
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
        },
        "metrics_at_optimal_threshold": {
            "threshold": float(best_thresh),
            "f1": float(f1_score(y_true, y_pred_opt)),
            "precision": float(precision_score(y_true, y_pred_opt)),
            "recall": float(recall_score(y_true, y_pred_opt)),
        },
        "score_stats": {
            "af_mean": float(y_prob[y_true == 1].mean()) if y_true.sum() > 0 else 0,
            "af_std": float(y_prob[y_true == 1].std()) if y_true.sum() > 0 else 0,
            "non_af_mean": float(y_prob[y_true == 0].mean()) if (y_true == 0).sum() > 0 else 0,
            "non_af_std": float(y_prob[y_true == 0].std()) if (y_true == 0).sum() > 0 else 0,
        },
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Samples: {results['n_samples']}")
        print(f"Held-out evaluation: {results['held_out_evaluation']}")
        print(f"\nAt signature threshold ({sig.threshold:.2f}):")
        m = results["metrics_at_signature_threshold"]
        print(f"  F1: {m['f1']:.1%}")
        print(f"  Precision: {m['precision']:.1%}")
        print(f"  Recall: {m['recall']:.1%}")
        print(f"\nAt optimal threshold ({results['metrics_at_optimal_threshold']['threshold']:.2f}):")
        m = results["metrics_at_optimal_threshold"]
        print(f"  F1: {m['f1']:.1%}")
        print(f"  Precision: {m['precision']:.1%}")
        print(f"  Recall: {m['recall']:.1%}")

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output}")


def cmd_signatures(args):
    """List available signatures."""
    from bids.store import SignatureStore

    store = SignatureStore(Path(args.store))

    if args.model:
        sig_ids = store.list_for_model(args.model)
    else:
        sig_ids = store.list_all()

    if not sig_ids:
        print("No signatures found.")
        return

    print(f"\n{'ID':<40} {'Model':<15} {'Layer':<6} {'F1':<8}")
    print("-" * 70)

    for sig_id in sig_ids:
        sig = store.get(sig_id)
        print(f"{sig_id:<40} {sig.model_name:<15} {sig.layer:<6} {sig.test_f1:.1%}")


def cmd_info(args):
    """Show signature details."""
    with open(args.signature) as f:
        data = json.load(f)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"\n{'='*60}")
        print("SIGNATURE INFO")
        print(f"{'='*60}")
        print(f"ID: {data.get('id', 'N/A')}")
        print(f"Model: {data.get('model', data.get('model_name', 'N/A'))}")
        print(f"Layer: {data.get('layer', 'N/A')}")
        print(f"SAE: {data.get('sae_release', 'N/A')} / {data.get('sae_id', 'N/A')}")
        print(f"Features: {len(data.get('features', data.get('feature_indices', [])))}")
        print(f"Threshold: {data.get('threshold', 'N/A')}")

        metrics = data.get("metrics", {})
        if metrics:
            print(f"\nTest Metrics:")
            print(f"  F1: {metrics.get('f1', metrics.get('test_f1', 'N/A')):.1%}")
            print(f"  Precision: {metrics.get('precision', metrics.get('test_precision', 'N/A')):.1%}")
            print(f"  Recall: {metrics.get('recall', metrics.get('test_recall', 'N/A')):.1%}")

        print(f"\nTraining Samples: {data.get('training_samples', data.get('total_samples', 'N/A'))}")
        print(f"Test Indices: {len(data.get('test_indices', []))} samples")
        print(f"Generated: {data.get('generated', data.get('created_at', 'N/A'))}")


def main():
    parser = argparse.ArgumentParser(
        description="BIDS - Behavioral Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect alignment faking in text
  bids detect --model gpt2-small --text "I will comply for now..."

  # Train new signature
  bids train --model gpt2-small --data-dir ~/af-detection-benchmark/data --output signatures/

  # Evaluate signature
  bids evaluate --signature signatures/gpt2-small-l8-af.json --data-dir ~/af-detection-benchmark/data

  # List signatures
  bids signatures --store signatures/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # detect command
    p_detect = subparsers.add_parser("detect", help="Detect alignment faking")
    p_detect.add_argument("--model", default="gpt2-small", help="Model name")
    p_detect.add_argument("--signature", type=str, help="Path to signature file")
    p_detect.add_argument("--text", type=str, help="Text to analyze")
    p_detect.add_argument("--file", type=str, help="File containing text")
    p_detect.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_detect.add_argument("--json", action="store_true", help="Output as JSON")

    # train command
    p_train = subparsers.add_parser("train", help="Train new signature")
    p_train.add_argument("--model", required=True, help="Model name")
    p_train.add_argument("--data-dir", required=True, help="Path to benchmark data")
    p_train.add_argument("--output", help="Output directory for signature")
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--n-features", type=int, default=50, help="Number of features")
    p_train.add_argument("--batch-mode", action="store_true", help="Memory-efficient mode")
    p_train.add_argument("--json", action="store_true", help="Output metrics as JSON")

    # evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Evaluate signature")
    p_eval.add_argument("--signature", required=True, help="Path to signature file")
    p_eval.add_argument("--data-dir", required=True, help="Path to benchmark data")
    p_eval.add_argument("--output", help="Output file for results")
    p_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_eval.add_argument("--json", action="store_true", help="Output as JSON")

    # signatures command
    p_sigs = subparsers.add_parser("signatures", help="List signatures")
    p_sigs.add_argument("--store", default="signatures", help="Signature store path")
    p_sigs.add_argument("--model", help="Filter by model name")

    # info command
    p_info = subparsers.add_parser("info", help="Show signature info")
    p_info.add_argument("signature", help="Path to signature file")
    p_info.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "signatures":
        cmd_signatures(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
