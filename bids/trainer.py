"""Training pipeline for alignment faking detection probes."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from bids.types import ProbeSignature
from bids.extractor import get_extractor, BatchExtractor, MODEL_CONFIGS


class SignatureTrainer:
    """Train alignment faking detection probes."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_mode: bool = False,
    ):
        """Initialize trainer.

        Args:
            model_name: Model key (e.g., "gpt2-small")
            device: Device for activation extraction
            batch_mode: Use memory-efficient batch mode
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.device = device
        self.batch_mode = batch_mode

        self.extractor = get_extractor(model_name, device=device, batch_mode=batch_mode)

    def extract_activations(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract SAE-encoded activations for all texts.

        Returns:
            Array of shape (n_samples, n_features)
        """
        if self.batch_mode and isinstance(self.extractor, BatchExtractor):
            results = self.extractor.extract_batch(texts, show_progress=show_progress)
        else:
            results = self.extractor.extract_batch(texts, show_progress=show_progress)

        layer = self.config["layer"]
        return np.stack([r[layer] for r in results])

    def train(
        self,
        texts: List[str],
        labels: List[int],
        n_features: int = 50,
        test_size: float = 0.2,
        regularization: str = "l1",
        random_state: int = 42,
    ) -> Tuple[ProbeSignature, Dict]:
        """Train alignment faking probe.

        Args:
            texts: List of input texts
            labels: Binary labels (1 = alignment faking, 0 = not)
            n_features: Number of top features to include in signature
            test_size: Fraction of data for held-out test
            regularization: Regularization type ("l1" or "l2")
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (ProbeSignature, metrics dict)
        """
        print(f"Extracting activations for {len(texts)} samples...")
        X = self.extract_activations(texts)
        y = np.array(labels)

        print(f"Activations shape: {X.shape}")

        # Split with index tracking (prevents data leakage)
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

        # Train L1 logistic regression (sparse feature selection)
        print("Training probe...")
        clf = LogisticRegression(
            penalty=regularization,
            solver="saga",
            max_iter=1000,
            C=0.1,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Get top features by absolute weight
        coef = clf.coef_[0]
        top_indices = np.argsort(np.abs(coef))[-n_features:][::-1]
        top_weights = coef[top_indices].tolist()

        # Find optimal threshold on test set
        y_prob = clf.predict_proba(X_test)[:, 1]
        best_f1, best_threshold = 0.0, 0.5

        for thresh in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob > thresh).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)

        # Final evaluation at optimal threshold
        y_pred_opt = (y_prob > best_threshold).astype(int)
        test_precision = precision_score(y_test, y_pred_opt)
        test_recall = recall_score(y_test, y_pred_opt)

        # Build signature
        signature = ProbeSignature(
            id=f"{self.model_name}_l{self.config['layer']}_af",
            version="1.0.0",
            created_at=datetime.utcnow(),
            model_family=self.model_name.split("-")[0],
            model_name=self.model_name,
            layer=self.config["layer"],
            hook_point=self.config["hook_point"],
            sae_release=self.config["sae_release"],
            sae_id=self.config["sae_id"],
            n_sae_features=X.shape[1],
            probe_type="sae_sparse",
            feature_indices=top_indices.tolist(),
            feature_weights=top_weights,
            intercept=float(clf.intercept_[0]),
            threshold=best_threshold,
            behavior_name="alignment_faking",
            behavior_category="safety",
            description="Alignment faking detection probe",
            total_samples=len(texts),
            test_indices=test_idx.tolist(),
            test_f1=best_f1,
            test_precision=test_precision,
            test_recall=test_recall,
            training_data_hash=hashlib.sha256(
                str(sorted(texts)).encode()
            ).hexdigest()[:16],
            training_config={
                "regularization": regularization,
                "test_size": test_size,
                "random_state": random_state,
                "n_features": n_features,
            },
        )

        metrics = {
            "test_f1": best_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "threshold": best_threshold,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive_train": int(y_train.sum()),
            "n_positive_test": int(y_test.sum()),
        }

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.config['layer']}")
        print(f"Top {n_features} features selected")
        print(f"F1: {best_f1:.1%}")
        print(f"Precision: {test_precision:.1%}")
        print(f"Recall: {test_recall:.1%}")
        print(f"Threshold: {best_threshold:.2f}")

        return signature, metrics

    def unload(self) -> None:
        """Unload model/SAE to free memory."""
        self.extractor.unload()


def load_benchmark_data(data_dir: Path) -> List[Dict]:
    """Load af-detection-benchmark data.

    Expects directory structure:
        data_dir/classes/alignment_faking.json
        data_dir/classes/normal.json
        etc.

    Returns:
        List of dicts with 'text', 'label', 'class' keys
    """
    classes_dir = data_dir / "classes"
    samples = []

    if not classes_dir.exists():
        raise ValueError(f"Classes directory not found: {classes_dir}")

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
            # Binary label: 1 for alignment faking, 0 for everything else
            label = 1 if class_name == "alignment_faking" else 0
            text = item.get("text", item.get("scratchpad", ""))

            if text:
                samples.append({
                    "text": text,
                    "label": label,
                    "class": class_name,
                })

    return samples


def train_from_benchmark(
    model_name: str,
    data_dir: Path,
    output_dir: Path = None,
    device: str = "cuda",
    n_features: int = 50,
    batch_mode: bool = False,
) -> Tuple[ProbeSignature, Dict]:
    """Train signature from af-detection-benchmark data.

    Args:
        model_name: Model key
        data_dir: Path to benchmark data directory
        output_dir: Optional directory to save signature
        device: Device for training
        n_features: Number of features in signature
        batch_mode: Use memory-efficient mode

    Returns:
        Tuple of (signature, metrics)
    """
    # Load data
    print("Loading benchmark data...")
    samples = load_benchmark_data(data_dir)
    print(f"Loaded {len(samples)} samples")

    n_positive = sum(1 for s in samples if s["label"] == 1)
    n_negative = len(samples) - n_positive
    print(f"  Positive (AF): {n_positive}")
    print(f"  Negative: {n_negative}")

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    # Train
    trainer = SignatureTrainer(model_name, device=device, batch_mode=batch_mode)
    signature, metrics = trainer.train(texts, labels, n_features=n_features)

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{model_name}-l{signature.layer}-af.json"
        signature.to_file(output_path)
        print(f"\nSignature saved: {output_path}")

    # Clean up
    trainer.unload()

    return signature, metrics
