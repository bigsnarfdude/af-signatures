# System Design Document: LLM Behavioral Intrusion Detection System

**Project Name:** BIDS (Behavioral Intrusion Detection System)
**Version:** 0.1 Draft
**Date:** 2026-01-11
**Author:** Vincent / Claude

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BIDS ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   Inputs    │     │              DETECTION ENGINE                    │   │
│  ├─────────────┤     ├─────────────────────────────────────────────────┤   │
│  │             │     │                                                  │   │
│  │ • Inline    │     │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │   inference │────▶│  │Activation│─▶│  Probe   │─▶│ State Vector │  │   │
│  │             │     │  │Extractor │  │  Runner  │  │  Assembler   │  │   │
│  │ • Batch     │     │  └──────────┘  └──────────┘  └──────────────┘  │   │
│  │   logs      │     │       │              │                          │   │
│  │             │     │       ▼              ▼                          │   │
│  │ • API       │     │  ┌──────────┐  ┌──────────┐                    │   │
│  │   request   │     │  │  Model   │  │Signature │                    │   │
│  │             │     │  │  Hooks   │  │  Store   │                    │   │
│  └─────────────┘     │  └──────────┘  └──────────┘                    │   │
│                      │                                                  │   │
│                      └─────────────────────────────────────────────────┘   │
│                                         │                                   │
│                                         ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT LAYER                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │  Alerts  │  │Dashboard │  │  Audit   │  │   API Response   │   │   │
│  │  │  System  │  │   Feed   │  │   Log    │  │                  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Summary

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Activation Extractor | Hook into model, extract layer outputs | PyTorch hooks, TransformerLens |
| Probe Runner | Apply signatures to activations | NumPy, PyTorch |
| Signature Store | Load/manage probe signatures | JSON/Pickle, Redis (prod) |
| State Vector Assembler | Aggregate scores, apply thresholds | Python |
| Alert System | Trigger notifications | Webhooks, PagerDuty SDK |
| Audit Log | Immutable detection records | PostgreSQL, S3 |
| Dashboard Feed | Real-time metrics stream | WebSocket, Kafka |

---

## 2. Data Structures

### 2.1 Signature Format

```python
@dataclass
class ProbeSignature:
    """Serializable probe signature."""

    # Identity
    id: str                      # "alignment_faking_v2"
    version: str                 # "2.1.0"
    created_at: datetime

    # Model targeting
    model_family: str            # "gpt2", "gemma", "llama", "claude"
    model_name: str              # "gpt2-small", "gemma-2-2b"
    layer: int                   # 8
    hook_point: str              # "blocks.8.hook_resid_post"

    # Probe weights
    probe_type: str              # "linear", "mlp", "sae_sparse"
    weights: np.ndarray          # Shape: (hidden_dim,) for linear
    bias: float                  # Intercept term
    threshold: float             # Classification threshold

    # Optional: SAE-based
    sae_release: Optional[str]   # "gpt2-small-res-jb"
    sae_id: Optional[str]        # "blocks.8.hook_resid_pre"
    feature_indices: Optional[List[int]]  # Which SAE features to use
    feature_weights: Optional[np.ndarray] # Weights for those features

    # Metadata
    behavior_category: str       # "safety", "quality", "security"
    behavior_name: str           # "alignment_faking"
    description: str
    training_samples: int
    training_f1: float
    training_precision: float
    training_recall: float

    # Provenance
    training_data_hash: str      # SHA256 of training data
    training_config: Dict        # Hyperparameters used

# Serialized format (JSON)
{
    "id": "alignment_faking_gpt2small_l8_v2",
    "version": "2.1.0",
    "model_family": "gpt2",
    "model_name": "gpt2-small",
    "layer": 8,
    "hook_point": "blocks.8.hook_resid_post",
    "probe_type": "linear",
    "weights_base64": "...",  # Base64 encoded numpy array
    "bias": -0.234,
    "threshold": 0.94,
    "behavior_category": "safety",
    "behavior_name": "alignment_faking",
    "metrics": {
        "f1": 0.996,
        "precision": 0.992,
        "recall": 1.0,
        "training_samples": 2330
    }
}
```

### 2.2 State Vector Format

```python
@dataclass
class StateVector:
    """Detection result for a single response."""

    # Identity
    request_id: str
    timestamp: datetime
    model_name: str

    # Input reference (not stored, just hash)
    input_hash: str              # SHA256 of input text
    input_length: int            # Token count

    # Probe results
    scores: Dict[str, float]     # {"alignment_faking": 0.94, "sycophancy": 0.12}
    triggered: List[str]         # ["alignment_faking"]

    # Aggregates
    max_safety_score: float      # Highest score among safety probes
    any_triggered: bool          # Any probe above threshold

    # Metadata
    extraction_time_ms: float
    probe_time_ms: float
    total_time_ms: float

# JSON output
{
    "request_id": "req_abc123",
    "timestamp": "2026-01-11T12:34:56Z",
    "model": "gpt2-small",
    "scores": {
        "alignment_faking": 0.94,
        "deception": 0.12,
        "sycophancy": 0.31,
        "sandbagging": 0.08
    },
    "triggered": ["alignment_faking"],
    "max_safety_score": 0.94,
    "latency_ms": 45
}
```

### 2.3 Activation Cache Format

```python
@dataclass
class ActivationRecord:
    """Cached activations for batch processing."""

    record_id: str
    model_name: str
    input_hash: str

    # Activations at multiple layers
    activations: Dict[int, np.ndarray]  # {8: array(768,), 16: array(768,)}

    # Metadata
    sequence_length: int
    aggregation: str             # "max", "mean", "last"
    extracted_at: datetime

# Storage: Parquet files partitioned by date and model
# activations/2026-01-11/gpt2-small/batch_001.parquet
```

---

## 3. Core Components

### 3.1 Activation Extractor

```python
class ActivationExtractor:
    """Extract activations from model layers."""

    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        hook_type: str = "resid_post",
        aggregation: str = "max",
        device: str = "cuda",
    ):
        self.model = model
        self.layers = layers
        self.hook_type = hook_type
        self.aggregation = aggregation
        self.device = device
        self._hooks = []
        self._activations = {}

    def _get_hook_points(self) -> List[str]:
        """Generate hook point names for target layers."""
        # TransformerLens style
        return [f"blocks.{l}.hook_{self.hook_type}" for l in self.layers]

    def extract(self, text: str) -> Dict[int, np.ndarray]:
        """Extract activations for a single text."""
        # Implementation depends on model framework
        pass

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[int, np.ndarray]]:
        """Extract activations for multiple texts."""
        pass


class TransformerLensExtractor(ActivationExtractor):
    """Extractor for TransformerLens models."""

    def extract(self, text: str) -> Dict[int, np.ndarray]:
        tokens = self.model.to_tokens(text)

        hook_points = self._get_hook_points()

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda n: n in hook_points
            )

        activations = {}
        for layer in self.layers:
            hook = f"blocks.{layer}.hook_{self.hook_type}"
            acts = cache[hook].squeeze(0)  # (seq, hidden)

            if self.aggregation == "max":
                acts = acts.max(dim=0).values
            elif self.aggregation == "mean":
                acts = acts.mean(dim=0)
            elif self.aggregation == "last":
                acts = acts[-1]

            activations[layer] = acts.cpu().numpy()

        return activations


class HuggingFaceExtractor(ActivationExtractor):
    """Extractor for HuggingFace models."""

    def extract(self, text: str) -> Dict[int, np.ndarray]:
        # Register forward hooks
        # Run forward pass
        # Collect activations
        pass
```

### 3.2 Probe Runner

```python
class ProbeRunner:
    """Apply probe signatures to activations."""

    def __init__(self, signature_store: "SignatureStore"):
        self.store = signature_store
        self._loaded_probes: Dict[str, ProbeSignature] = {}

    def load_probes(self, probe_ids: List[str]) -> None:
        """Pre-load probes into memory."""
        for pid in probe_ids:
            if pid not in self._loaded_probes:
                self._loaded_probes[pid] = self.store.get(pid)

    def run_probe(
        self,
        probe: ProbeSignature,
        activations: Dict[int, np.ndarray],
    ) -> float:
        """Run a single probe, return probability score."""

        acts = activations[probe.layer]

        if probe.probe_type == "linear":
            # Simple linear probe: sigmoid(w·x + b)
            logit = np.dot(acts, probe.weights) + probe.bias
            prob = 1 / (1 + np.exp(-logit))
            return float(prob)

        elif probe.probe_type == "sae_sparse":
            # SAE-based: encode with SAE, then weighted sum
            # Requires SAE to be loaded
            sae_acts = self._encode_sae(acts, probe)
            selected = sae_acts[probe.feature_indices]
            logit = np.dot(selected, probe.feature_weights) + probe.bias
            prob = 1 / (1 + np.exp(-logit))
            return float(prob)

        else:
            raise ValueError(f"Unknown probe type: {probe.probe_type}")

    def run_all(
        self,
        activations: Dict[int, np.ndarray],
        model_name: str,
    ) -> Dict[str, float]:
        """Run all applicable probes for a model."""

        scores = {}
        for probe_id, probe in self._loaded_probes.items():
            if probe.model_name == model_name:
                if probe.layer in activations:
                    scores[probe.behavior_name] = self.run_probe(
                        probe, activations
                    )

        return scores
```

### 3.3 Signature Store

```python
class SignatureStore:
    """Manage probe signature storage and retrieval."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan signature directory and build index."""
        for sig_file in self.base_path.rglob("*.json"):
            with open(sig_file) as f:
                meta = json.load(f)
            self._index[meta["id"]] = sig_file

    def get(self, probe_id: str) -> ProbeSignature:
        """Load a signature by ID."""
        path = self._index[probe_id]
        return ProbeSignature.from_file(path)

    def list_for_model(self, model_name: str) -> List[str]:
        """List all probe IDs applicable to a model."""
        result = []
        for probe_id, path in self._index.items():
            # Quick check without full load
            with open(path) as f:
                meta = json.load(f)
            if meta["model_name"] == model_name:
                result.append(probe_id)
        return result

    def add(self, signature: ProbeSignature) -> None:
        """Add a new signature to the store."""
        path = self.base_path / signature.behavior_category / f"{signature.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        signature.to_file(path)
        self._index[signature.id] = path


class RedisSignatureStore(SignatureStore):
    """Production signature store with Redis backend."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def get(self, probe_id: str) -> ProbeSignature:
        data = self.redis.get(f"probe:{probe_id}")
        return ProbeSignature.from_json(data)
```

### 3.4 Detection Engine (Main Orchestrator)

```python
class DetectionEngine:
    """Main orchestrator for behavioral detection."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        signature_store: SignatureStore,
        layers: List[int] = None,
        alert_handler: "AlertHandler" = None,
        audit_logger: "AuditLogger" = None,
    ):
        self.model_name = model_name
        self.extractor = TransformerLensExtractor(
            model=model,
            layers=layers or [8],  # Default middle layer
        )
        self.probe_runner = ProbeRunner(signature_store)
        self.alert_handler = alert_handler
        self.audit_logger = audit_logger

        # Load all probes for this model
        probe_ids = signature_store.list_for_model(model_name)
        self.probe_runner.load_probes(probe_ids)

        # Load thresholds
        self._thresholds = {
            pid: signature_store.get(pid).threshold
            for pid in probe_ids
        }

    def detect(self, text: str, request_id: str = None) -> StateVector:
        """Run full detection pipeline on text."""

        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()

        # Extract activations
        t0 = time.time()
        activations = self.extractor.extract(text)
        extraction_time = (time.time() - t0) * 1000

        # Run probes
        t0 = time.time()
        scores = self.probe_runner.run_all(activations, self.model_name)
        probe_time = (time.time() - t0) * 1000

        # Determine triggered states
        triggered = [
            name for name, score in scores.items()
            if score > self._thresholds.get(name, 0.5)
        ]

        # Build state vector
        state = StateVector(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            model_name=self.model_name,
            input_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            input_length=len(text.split()),
            scores=scores,
            triggered=triggered,
            max_safety_score=max(
                (s for n, s in scores.items() if n in SAFETY_BEHAVIORS),
                default=0
            ),
            any_triggered=len(triggered) > 0,
            extraction_time_ms=extraction_time,
            probe_time_ms=probe_time,
            total_time_ms=(time.time() - start_time) * 1000,
        )

        # Handle alerts
        if self.alert_handler and triggered:
            self.alert_handler.handle(state)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log(state)

        return state

    def detect_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[StateVector]:
        """Batch detection for offline analysis."""
        results = []
        iterator = tqdm(texts) if show_progress else texts
        for text in iterator:
            results.append(self.detect(text))
        return results
```

---

## 4. Integration Patterns

### 4.1 Inline Mode (Middleware)

```python
class BIDSMiddleware:
    """Middleware for inline detection during inference."""

    def __init__(self, engine: DetectionEngine, block_on_trigger: bool = False):
        self.engine = engine
        self.block_on_trigger = block_on_trigger

    def __call__(self, generate_fn):
        """Wrap a generation function."""

        def wrapped(prompt: str, **kwargs) -> Tuple[str, StateVector]:
            # Generate response
            response = generate_fn(prompt, **kwargs)

            # Detect on response (or prompt+response)
            state = self.engine.detect(response)

            # Optionally block
            if self.block_on_trigger and state.any_triggered:
                raise BehaviorTriggeredError(state)

            return response, state

        return wrapped

# Usage
engine = DetectionEngine(model, "gpt2-small", store)
middleware = BIDSMiddleware(engine)

@middleware
def generate(prompt):
    return model.generate(prompt)

response, state = generate("Hello world")
```

### 4.2 Batch Mode (Offline)

```python
class BatchProcessor:
    """Process logged responses in batch."""

    def __init__(
        self,
        engine: DetectionEngine,
        input_path: Path,
        output_path: Path,
    ):
        self.engine = engine
        self.input_path = input_path
        self.output_path = output_path

    def process(self) -> pd.DataFrame:
        """Process all responses and save results."""

        # Load responses
        df = pd.read_parquet(self.input_path)

        # Detect
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            state = self.engine.detect(
                row["response"],
                request_id=row["request_id"],
            )
            results.append(state.to_dict())

        # Save
        results_df = pd.DataFrame(results)
        results_df.to_parquet(self.output_path)

        return results_df

# Usage
processor = BatchProcessor(engine, "responses.parquet", "detections.parquet")
results = processor.process()
```

### 4.3 API Mode (REST)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class DetectRequest(BaseModel):
    text: str
    model: str = "gpt2-small"

class DetectResponse(BaseModel):
    scores: Dict[str, float]
    triggered: List[str]
    latency_ms: float

# Engine cache per model
engines: Dict[str, DetectionEngine] = {}

@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if request.model not in engines:
        raise HTTPException(404, f"Model {request.model} not loaded")

    engine = engines[request.model]
    state = engine.detect(request.text)

    return DetectResponse(
        scores=state.scores,
        triggered=state.triggered,
        latency_ms=state.total_time_ms,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(engines.keys())}
```

---

## 5. Signature Training Pipeline

### 5.1 Training Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Labeled    │───▶│  Extract    │───▶│   Train     │───▶│   Export    │
│  Dataset    │    │  Activations│    │   Probe     │    │  Signature  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     CSV              .npy              sklearn           .json
   text,label       (N, hidden)        LogisticReg
```

### 5.2 Training Script

```python
class SignatureTrainer:
    """Train new probe signatures."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        layer: int,
        behavior_name: str,
    ):
        self.model = model
        self.model_name = model_name
        self.layer = layer
        self.behavior_name = behavior_name
        self.extractor = TransformerLensExtractor(model, [layer])

    def train(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        regularization: str = "l2",
    ) -> ProbeSignature:
        """Train probe and return signature."""

        # Extract activations
        print("Extracting activations...")
        activations = []
        for text in tqdm(texts):
            acts = self.extractor.extract(text)
            activations.append(acts[self.layer])
        X = np.stack(activations)
        y = np.array(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train
        print("Training probe...")
        clf = LogisticRegression(
            penalty=regularization,
            solver="saga",
            max_iter=1000,
            C=0.1,
        )
        clf.fit(X_train, y_train)

        # Find optimal threshold
        y_prob = clf.predict_proba(X_test)[:, 1]
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            pred = (y_prob > thresh).astype(int)
            f1 = f1_score(y_test, pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        # Evaluate at optimal
        y_pred = (y_prob > best_thresh).astype(int)

        # Build signature
        signature = ProbeSignature(
            id=f"{self.behavior_name}_{self.model_name}_l{self.layer}",
            version="1.0.0",
            created_at=datetime.utcnow(),
            model_family=self.model_name.split("-")[0],
            model_name=self.model_name,
            layer=self.layer,
            hook_point=f"blocks.{self.layer}.hook_resid_post",
            probe_type="linear",
            weights=clf.coef_[0],
            bias=clf.intercept_[0],
            threshold=best_thresh,
            behavior_category="safety",  # TODO: parameterize
            behavior_name=self.behavior_name,
            description=f"Probe for {self.behavior_name} detection",
            training_samples=len(texts),
            training_f1=best_f1,
            training_precision=precision_score(y_test, y_pred),
            training_recall=recall_score(y_test, y_pred),
            training_data_hash=hashlib.sha256(
                str(sorted(texts)).encode()
            ).hexdigest()[:16],
            training_config={
                "regularization": regularization,
                "test_size": test_size,
            },
        )

        print(f"Trained: F1={best_f1:.1%}, threshold={best_thresh:.2f}")
        return signature
```

---

## 6. Storage Schema

### 6.1 Signature Directory Structure

```
signatures/
├── safety/
│   ├── alignment_faking/
│   │   ├── gpt2-small-l8-v1.json
│   │   ├── gpt2-small-l8-v2.json
│   │   ├── gemma-2-2b-l20-v1.json
│   │   └── llama-3-8b-l16-v1.json
│   ├── deception/
│   ├── sandbagging/
│   └── power_seeking/
├── quality/
│   ├── sycophancy/
│   ├── hallucination/
│   └── refusal/
├── security/
│   ├── jailbreak/
│   └── prompt_injection/
└── index.json  # Master index of all signatures
```

### 6.2 Audit Log Schema (PostgreSQL)

```sql
CREATE TABLE detection_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(64) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    input_length INT,

    -- Scores stored as JSONB
    scores JSONB NOT NULL,
    triggered VARCHAR(64)[],
    max_safety_score FLOAT,

    -- Performance
    latency_ms FLOAT,

    -- Indexes
    INDEX idx_timestamp (timestamp),
    INDEX idx_model (model_name),
    INDEX idx_triggered (triggered) USING GIN
);

-- Partitioned by month for scale
CREATE TABLE detection_logs_2026_01 PARTITION OF detection_logs
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

### 6.3 Metrics Schema (Time Series)

```
# InfluxDB / Prometheus style

bids_detection_score{model="gpt2-small", behavior="alignment_faking"} 0.94
bids_detection_triggered{model="gpt2-small", behavior="alignment_faking"} 1
bids_latency_ms{model="gpt2-small", stage="extraction"} 45
bids_latency_ms{model="gpt2-small", stage="probe"} 2
bids_requests_total{model="gpt2-small"} 10532
```

---

## 7. Deployment Architecture

### 7.1 Single Node (Development)

```
┌─────────────────────────────────────────┐
│              Single GPU Node            │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐    ┌─────────────┐    │
│  │   Model     │    │    BIDS     │    │
│  │  (GPU)      │◄──▶│   Engine    │    │
│  └─────────────┘    └─────────────┘    │
│                            │            │
│                            ▼            │
│                     ┌─────────────┐    │
│                     │  Signatures │    │
│                     │  (local fs) │    │
│                     └─────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

### 7.2 Production (Distributed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
│  │   Load      │    │         Detection Cluster               │    │
│  │  Balancer   │───▶│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │    │
│  └─────────────┘    │  │ GPU │ │ GPU │ │ GPU │ │ GPU │      │    │
│                     │  │ Node│ │ Node│ │ Node│ │ Node│      │    │
│                     │  └─────┘ └─────┘ └─────┘ └─────┘      │    │
│                     └─────────────────────────────────────────┘    │
│                                    │                                │
│         ┌──────────────────────────┼──────────────────────────┐    │
│         ▼                          ▼                          ▼    │
│  ┌─────────────┐           ┌─────────────┐           ┌───────────┐│
│  │   Redis     │           │  PostgreSQL │           │   Kafka   ││
│  │ (signatures)│           │ (audit logs)│           │ (metrics) ││
│  └─────────────┘           └─────────────┘           └───────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. API Specification

### 8.1 REST Endpoints

```yaml
openapi: 3.0.0
info:
  title: BIDS API
  version: 1.0.0

paths:
  /detect:
    post:
      summary: Detect behavioral states in text
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                text:
                  type: string
                model:
                  type: string
                probes:
                  type: array
                  items:
                    type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StateVector'

  /signatures:
    get:
      summary: List available signatures
      parameters:
        - name: model
          in: query
          schema:
            type: string
        - name: category
          in: query
          schema:
            type: string
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/SignatureMeta'

  /signatures/{id}:
    get:
      summary: Get signature details
    put:
      summary: Update signature
    delete:
      summary: Remove signature

  /batch:
    post:
      summary: Submit batch detection job
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                input_path:
                  type: string
                model:
                  type: string
      responses:
        202:
          description: Job accepted
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string

  /batch/{job_id}:
    get:
      summary: Get batch job status

components:
  schemas:
    StateVector:
      type: object
      properties:
        request_id:
          type: string
        scores:
          type: object
          additionalProperties:
            type: number
        triggered:
          type: array
          items:
            type: string
        latency_ms:
          type: number
```

---

## 9. Security Considerations

### 9.1 Signature Protection

- Signatures reveal what behaviors we're detecting
- Adversary could use to craft evasion attacks
- Mitigations:
  - Encrypt signatures at rest
  - Access control for signature management
  - Don't expose raw weights via API
  - Use signature ensembles (harder to evade all)

### 9.2 Input Validation

- Limit input text length (prevent DoS)
- Sanitize model names (prevent injection)
- Rate limiting per client

### 9.3 Audit Trail

- All signature changes logged
- Detection results immutable
- Access logs for compliance

---

## 10. Testing Strategy

### 10.1 Unit Tests

- Activation extraction correctness
- Probe arithmetic accuracy
- Signature serialization round-trip

### 10.2 Integration Tests

- End-to-end detection pipeline
- API endpoint behavior
- Batch processing completion

### 10.3 Performance Tests

- Latency under load
- Memory usage with many probes
- Throughput scaling

### 10.4 Behavioral Tests

- Known AF samples → high scores
- Known benign samples → low scores
- Cross-model consistency

---

## Appendix A: Configuration Example

```yaml
# bids_config.yaml

models:
  gpt2-small:
    type: transformer_lens
    layers: [8]
    hook_type: resid_post

  gemma-2-2b:
    type: transformer_lens
    layers: [20]
    hook_type: resid_post
    batch_mode: true  # Memory constrained

signatures:
  store_type: redis
  redis_url: redis://localhost:6379
  # Or: store_type: file, base_path: ./signatures

alerts:
  enabled: true
  handlers:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK}
      behaviors: [alignment_faking, deception]
      min_score: 0.9
    - type: pagerduty
      api_key: ${PAGERDUTY_KEY}
      behaviors: [alignment_faking]
      min_score: 0.95

logging:
  audit:
    enabled: true
    backend: postgresql
    connection: ${DATABASE_URL}
  metrics:
    enabled: true
    backend: prometheus
    port: 9090
```

---

## Appendix B: CLI Reference

```bash
# Generate signature
bids train \
  --model gpt2-small \
  --layer 8 \
  --behavior alignment_faking \
  --data labeled_samples.csv \
  --output signatures/

# Run detection
bids detect \
  --model gpt2-small \
  --text "I will comply for now..." \
  --signatures signatures/

# Batch processing
bids batch \
  --model gpt2-small \
  --input responses.parquet \
  --output detections.parquet \
  --signatures signatures/

# Start API server
bids serve \
  --config bids_config.yaml \
  --port 8080

# List signatures
bids signatures list --model gpt2-small

# Evaluate signature
bids evaluate \
  --signature signatures/alignment_faking_gpt2-small_l8.json \
  --test-data test_samples.csv
```
