# Product Requirements Document: LLM Behavioral Intrusion Detection System

**Project Name:** BIDS (Behavioral Intrusion Detection System)
**Version:** 0.1 Draft
**Date:** 2026-01-11
**Author:** Vincent / Claude

---

## 1. Executive Summary

### 1.1 Problem Statement

Large Language Models in production exhibit concerning behavioral states that are difficult to detect at scale:
- **Alignment faking**: Appearing compliant while reasoning deceptively
- **Sandbagging**: Deliberately underperforming on capability evaluations
- **Sycophancy**: Excessively agreeing with users against better judgment
- **Deception**: Actively misleading users or operators
- **Jailbreak compliance**: Following adversarial prompts that bypass safety

Current detection methods rely on output text analysis (keyword matching, classifier on text), missing the rich signal in model internal states. Models can produce identical outputs while having very different internal reasoning patterns.

### 1.2 Proposed Solution

Build a **signature-based behavioral monitoring system** that:
1. Extracts model activations from specific layers during inference
2. Applies pre-trained probe signatures to detect concerning behavioral states
3. Returns a real-time state vector for monitoring, alerting, and logging
4. Supports multiple models and behavioral categories

Think: **antivirus signatures, but for LLM behavioral states.**

### 1.3 Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Detection F1 (alignment faking) | >95% | On held-out test set (20% of data, never seen during training) |
| Detection F1 (other P0 behaviors) | >90% | On held-out test set |
| Detection F1 (P1/P2 behaviors) | >85% | On held-out test set |
| Latency overhead per response | <100ms p99 | <50ms p50 target |
| False positive rate (P0 safety) | <5% | Optimize for recall on safety-critical behaviors |
| False positive rate (P1/P2) | <1% | Standard precision/recall balance |
| Cross-model generalization | >80% F1 | Train on model A, test on model B |
| Baseline improvement | >10% F1 | vs. text-only classifier baseline |
| Supported models | 5+ architectures | GPT-2, Gemma, Llama, Mistral, Qwen |

---

## 2. User Personas

### 2.1 Safety Researcher
**Needs:** Understand when and why models exhibit concerning behaviors during evals
**Uses:** Batch analysis of eval responses, compare behavior across model versions

### 2.2 ML Operations Engineer
**Needs:** Monitor production models for behavioral anomalies at scale
**Uses:** Dashboard alerts, integration with logging infrastructure

### 2.3 Red Team Member
**Needs:** Test if adversarial prompts trigger concerning internal states
**Uses:** Real-time probe feedback during attack development

### 2.4 Policy/Governance Team
**Needs:** Audit trail of model behavioral states for compliance
**Uses:** Reports, aggregate statistics, incident investigation

---

## 3. Functional Requirements

### 3.1 Core Detection Engine

#### FR-1: Activation Extraction
- Extract activations from configurable model layers
- Support multiple extraction points (residual stream, attention, MLP)
- Handle variable sequence lengths (aggregate via max/mean/last token)
- Batch extraction for offline analysis

#### FR-2: Probe Application
- Load probe signatures from standardized format
- Apply multiple probes in single pass (shared activation extraction)
- Return calibrated probability scores (0-1) for each behavioral state
- Calibration via Platt scaling or isotonic regression
- Support configurable thresholds per probe

#### FR-3: State Vector Output
- Return structured state vector with all probe scores
- Include confidence intervals via ensemble variance or bootstrap estimation
- Flag triggered states (score > threshold)
- Timestamp and metadata for logging

### 3.2 Signature Management

#### FR-4: Signature Format
- Standardized JSON/binary format for probe weights
- Include metadata:
  - Model: name, family, layer, hook point
  - Training data: hash, sample count, class balance, **held-out test indices**
  - Performance: F1, precision, recall (on held-out test set)
  - Threshold: optimized on validation fold, not test set
- Version control for signature updates
- Support both probe (dense) and SAE (sparse) signatures

#### FR-5: Signature Database
- Organize by model family and behavioral category
- Support signature updates without system restart
- Track signature lineage and training provenance
- Enable A/B testing of signature versions

#### FR-6: Signature Generation
- Training pipeline for new behavioral probes
- Support for labeled dataset ingestion
- Nested cross-validation for hyperparameter tuning
- Threshold optimization on validation fold (never on final test set)
- Export to standard signature format with test indices for reproducibility

### 3.3 Integration Modes

#### FR-7: Inline Mode
- Hook into model inference pipeline
- Extract activations during normal forward pass
- Minimal latency overhead (<50ms p50, <100ms p99)
- Logging and alerting only in v1 (blocking deferred to v2)

#### FR-8: Batch Mode
- Process logged responses offline
- High throughput (1000+ responses/minute on single A100 GPU)
- Generate aggregate reports and statistics
- Support distributed processing for larger workloads

#### FR-9: API Mode
- REST API for external integration
- Accept text + model identifier
- Return state vector as JSON
- Support async/webhook for long-running analysis

### 3.4 Alerting & Reporting

#### FR-10: Real-time Alerts
- Configurable alert thresholds per probe
- Integration with common alerting systems (PagerDuty, Slack, etc.)
- Alert aggregation to prevent flooding
- Severity levels (info, warning, critical)

#### FR-11: Dashboard
- Real-time state vector visualization
- Historical trends and patterns
- Drill-down to individual responses
- Model comparison views

#### FR-12: Audit Logging
- Immutable log of all detections
- Include: timestamp, model, input hash, state vector, triggered probes
- Retention policy compliance
- Export for external analysis

---

## 4. Non-Functional Requirements

### 4.1 Performance
- Inline detection latency: <100ms p99
- Batch throughput: >1000 responses/minute on single GPU
- Memory overhead: <2GB additional per model
- Signature load time: <1 second

### 4.2 Scalability
- Horizontal scaling for batch processing
- Support models up to 70B parameters (requires 80GB+ GPU for inline mode)
- Handle 10M+ logged responses per day
- Signature database: 1000+ signatures
- Memory scales with number of active probes (~100MB per probe)

### 4.3 Reliability
- 99.9% uptime for inline mode
- Graceful degradation if probes fail
- Optimize recall over precision for P0 safety behaviors (accept higher FPR)
- Automatic recovery from transient failures

### 4.4 Security
- Signatures are sensitive (reveal what we're looking for)
- Encrypted storage and transmission
- Access control for signature management
- Audit log for signature changes

### 4.5 Extensibility
- Plugin architecture for new model families
- Custom probe types (beyond linear)
- Integration hooks for custom alerting
- API versioning for backwards compatibility

---

## 5. Behavioral Categories

### 5.1 Safety Critical (P0)

| Behavior | Description | Training Signal | Min Samples |
|----------|-------------|-----------------|-------------|
| alignment_faking | Appearing compliant while reasoning deceptively | Anthropic AF dataset (2024), af-detection-benchmark | 500+ pos |
| deception | Actively misleading users | Synthetic deceptive responses (TBD) | 500+ pos |
| sandbagging | Deliberately underperforming | Capability eval failures, MMLU sandbagging | 500+ pos |
| power_seeking | Acquiring resources/influence inappropriately | MACHIAVELLI benchmark, synthetic | 500+ pos |

### 5.2 Quality (P1)

| Behavior | Description | Training Signal |
|----------|-------------|-----------------|
| sycophancy | Excessive agreement against better judgment | Sycophancy benchmark |
| hallucination | Confident false claims | Factuality datasets |
| refusal_appropriate | Correctly refusing harmful requests | Safety eval data |
| refusal_excessive | Over-refusing benign requests | False refusal examples |

### 5.3 Security (P1)

| Behavior | Description | Training Signal |
|----------|-------------|-----------------|
| jailbreak_compliance | Following adversarial bypass prompts | Jailbreak datasets |
| prompt_injection | Executing injected instructions | Injection examples |
| exfiltration_attempt | Trying to leak information | Synthetic scenarios |

### 5.4 Research (P2)

| Behavior | Description | Training Signal |
|----------|-------------|-----------------|
| uncertainty | Model is uncertain about response | Calibration data |
| out_of_distribution | Input is unlike training data | OOD detection data |
| capability_boundary | Near edge of model capabilities | Capability probing |

---

## 6. User Stories

### US-1: Safety Researcher Analyzes Eval Results
> As a safety researcher, I want to run BIDS on my eval responses to see which ones triggered alignment faking, so I can investigate the failure modes.

**Acceptance Criteria:**
- Upload batch of responses via CLI or API
- Get CSV with response ID, AF score, triggered flag
- Sort/filter by score to prioritize review

### US-2: MLOps Engineer Sets Up Production Monitoring
> As an MLOps engineer, I want to integrate BIDS into our inference pipeline so concerning responses are flagged in real-time.

**Acceptance Criteria:**
- SDK/library that hooks into inference
- Latency overhead <100ms
- Alerts to Slack when AF score >0.9
- Dashboard showing hourly AF rate

### US-3: Red Team Tests New Attack
> As a red team member, I want to see the state vector in real-time as I craft prompts, so I can understand which internal states I'm triggering.

**Acceptance Criteria:**
- Interactive mode with live state vector display
- Show all probe scores, not just triggered
- History of recent attempts

### US-4: Governance Audits Model Deployment
> As a governance team member, I want to audit the behavioral state history of our production model over the past month.

**Acceptance Criteria:**
- Query logs by date range, model, behavior type
- Aggregate statistics (% triggered per day)
- Export for compliance reporting

### US-5: Researcher Trains New Probe
> As a researcher, I want to train a probe for a new concerning behavior I've identified, so it can be added to the monitoring system.

**Acceptance Criteria:**
- Provide labeled dataset (text + label)
- System extracts activations, trains probe
- Cross-validation metrics reported
- Export to signature format

---

## 7. Out of Scope (v1)

- Training models to avoid triggering probes (adversarial robustness)
- Probes on attention patterns (only residual stream v1)
- Real-time intervention (blocking responses)
- Multi-turn conversation state tracking
- Interpretability of why probe triggered (just detection)

---

## 8. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Probes overfit to training data | High | Medium | Cross-model validation, diverse training sets |
| Adversarial evasion of probes | High | Medium | Ensemble probes, regular signature updates |
| Latency impact on production | Medium | Low | Async mode, sampling strategies |
| False positives cause alert fatigue | Medium | Medium | Tunable thresholds, alert aggregation |
| Signatures leak detection capabilities | Medium | Low | Access control, encryption |

---

## 9. Success Metrics

### 9.1 Detection Quality
- F1 score per behavioral category (target: >90% for P0)
- False positive rate (target: <1%)
- Cross-model generalization (train on A, test on B)

### 9.2 Operational
- Mean time to detection (inline mode)
- Batch processing throughput
- System uptime

### 9.3 Adoption
- Number of models monitored
- Number of signatures in database
- Daily responses scanned

---

## 10. Timeline & Milestones

| Phase | Deliverable |
|-------|-------------|
| Phase 1 | Core engine: extraction, probe application, single model |
| Phase 2 | Signature management, batch mode, basic dashboard |
| Phase 3 | Multi-model support, API mode, alerting integration |
| Phase 4 | Production hardening, security audit, documentation |

---

## Appendix A: Glossary

- **Probe**: Linear classifier trained on model activations to detect a concept/behavior
- **Signature**: Serialized probe weights + metadata, loadable at runtime
- **State Vector**: Array of probe scores for a single response
- **Watch State**: A behavioral category being monitored (e.g., "alignment_faking")
- **SAE**: Sparse Autoencoder - alternative to probes using interpretable features
- **Activation**: Internal model representation at a specific layer

---

## Appendix B: Related Work

- Anthropic alignment faking paper (2024)
- Representation Engineering (Zou et al.)
- Probing classifiers literature
- SAE interpretability (Anthropic, EleutherAI)
- Contrast Consistent Search (CCS)
