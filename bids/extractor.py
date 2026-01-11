"""Activation extraction from LLM layers with SAE encoding."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import gc

import numpy as np
import torch


class ActivationExtractor(ABC):
    """Base class for activation extraction."""

    def __init__(
        self,
        model_name: str,
        layers: List[int],
        hook_type: str = "resid_post",
        aggregation: str = "max",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.layers = layers
        self.hook_type = hook_type
        self.aggregation = aggregation
        self.device = device

    def _get_hook_points(self) -> List[str]:
        """Generate hook point names for target layers."""
        return [f"blocks.{layer}.hook_{self.hook_type}" for layer in self.layers]

    @abstractmethod
    def extract(self, text: str) -> Dict[int, np.ndarray]:
        """Extract activations for a single text.

        Returns:
            Dict mapping layer index to activation array.
        """
        pass

    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Dict[int, np.ndarray]]:
        """Extract activations for multiple texts."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(texts, desc="Extracting") if show_progress else texts
        for text in iterator:
            results.append(self.extract(text))
        return results

    def _aggregate(self, acts: torch.Tensor) -> torch.Tensor:
        """Aggregate activations over sequence dimension."""
        if self.aggregation == "max":
            return acts.max(dim=0).values
        elif self.aggregation == "mean":
            return acts.mean(dim=0)
        elif self.aggregation == "last":
            return acts[-1]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class TransformerLensExtractor(ActivationExtractor):
    """Extractor using TransformerLens with SAE encoding."""

    def __init__(
        self,
        model_name: str,
        layer: int,
        hook_point: str,
        sae_release: str,
        sae_id: str,
        aggregation: str = "max",
        device: str = "cuda",
        dtype: torch.dtype = None,
    ):
        super().__init__(
            model_name=model_name,
            layers=[layer],
            hook_type=hook_point.split("_")[-1] if "hook_" in hook_point else "resid_post",
            aggregation=aggregation,
            device=device,
        )
        self.layer = layer
        self.hook_point = hook_point
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.dtype = dtype

        self._model = None
        self._sae = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from transformer_lens import HookedTransformer

            kwargs = {"device": self.device}
            if self.dtype:
                kwargs["dtype"] = self.dtype

            self._model = HookedTransformer.from_pretrained(
                self.model_name, **kwargs
            )

    def _load_sae(self):
        """Lazy load the SAE."""
        if self._sae is None:
            from sae_lens import SAE

            self._sae, _, _ = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device,
            )

    def load(self):
        """Pre-load model and SAE."""
        self._load_model()
        self._load_sae()

    def unload(self):
        """Unload model and SAE to free memory."""
        del self._model
        del self._sae
        self._model = None
        self._sae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract(self, text: str) -> Dict[int, np.ndarray]:
        """Extract SAE-encoded activations for text."""
        self._load_model()
        self._load_sae()

        tokens = self._model.to_tokens(text)

        with torch.no_grad():
            _, cache = self._model.run_with_cache(
                tokens,
                names_filter=lambda name: name == self.hook_point,
            )
            acts = cache[self.hook_point]

            # Encode with SAE
            feat_acts = self._sae.encode(acts)

            # Aggregate over sequence
            aggregated = self._aggregate(feat_acts.squeeze(0))

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {self.layer: aggregated.cpu().numpy()}

    def extract_raw(self, text: str) -> Dict[int, np.ndarray]:
        """Extract raw model activations (without SAE encoding)."""
        self._load_model()

        tokens = self._model.to_tokens(text)

        with torch.no_grad():
            _, cache = self._model.run_with_cache(
                tokens,
                names_filter=lambda name: name == self.hook_point,
            )
            acts = cache[self.hook_point]
            aggregated = self._aggregate(acts.squeeze(0))

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {self.layer: aggregated.cpu().numpy()}

    @property
    def n_features(self) -> int:
        """Number of SAE features."""
        self._load_sae()
        return self._sae.cfg.d_sae


class BatchExtractor(TransformerLensExtractor):
    """Memory-efficient batch extractor for large models.

    Extracts model activations first, then encodes with SAE separately.
    """

    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> List[Dict[int, np.ndarray]]:
        """Memory-efficient batch extraction."""
        from tqdm import tqdm

        # Step 1: Extract model activations
        self._load_model()

        model_acts = []
        iterator = tqdm(texts, desc="Step 1/2: Model") if show_progress else texts
        for text in iterator:
            tokens = self._model.to_tokens(text)
            with torch.no_grad():
                _, cache = self._model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == self.hook_point,
                )
                acts = cache[self.hook_point]
                aggregated = self._aggregate(acts.squeeze(0))
                model_acts.append(aggregated.cpu().numpy())

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free model memory
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_acts = np.stack(model_acts)

        # Step 2: Encode with SAE
        self._load_sae()

        all_features = []
        n_batches = (len(model_acts) + batch_size - 1) // batch_size
        iterator = range(0, len(model_acts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Step 2/2: SAE", total=n_batches)

        for i in iterator:
            batch = torch.tensor(
                model_acts[i : i + batch_size],
                device=self.device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                features = self._sae.encode(batch)
                all_features.append(features.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free SAE memory
        del self._sae
        self._sae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        encoded = np.concatenate(all_features, axis=0)
        return [{self.layer: encoded[i]} for i in range(len(encoded))]


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


def get_extractor(
    model_key: str,
    device: str = "cuda",
    batch_mode: bool = False,
) -> TransformerLensExtractor:
    """Factory function to create extractor for a supported model."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]
    cls = BatchExtractor if batch_mode else TransformerLensExtractor

    return cls(
        model_name=config["model_name"],
        layer=config["layer"],
        hook_point=config["hook_point"],
        sae_release=config["sae_release"],
        sae_id=config["sae_id"],
        device=device,
    )
