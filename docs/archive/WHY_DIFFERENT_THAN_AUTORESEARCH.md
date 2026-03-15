# Why AutoEmailTrust Is Its Own System, Not a Fork of autoresearch

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) is the conceptual ancestor of this project. We borrowed the core insight — an agent proposes code changes, an eval loop accepts or rejects them, repeat — but the domain differences forced a ground-up rewrite. Here's why.

## What We Took From autoresearch

- **The loop shape**: agent proposes edit → run evaluation → keep or discard → repeat.
- **train.py as the single mutable file**: the agent edits one file; everything else is locked.
- **Git-backed versioning**: kept experiments are commits; discarded ones are reverted.
- **program.md as the research spec**: natural language instructions the agent reads each iteration.
- **Simplicity constraint**: the searchable code must fit in an LLM context window.

## What We Had to Change (and Why)

### 1. Single Metric → Multi-Axis Trust Vector

autoresearch optimizes **one number**: validation bits-per-byte (val_bpb). Lower is better. Done.

Email trust scoring produces a **10-axis vector** (phish, truthfulness, manipulation, etc.), each with its own metric type (binary → F1, continuous → agreement), weight, and failure mode. A single composite score hides axis-level regressions — improving phish detection while breaking manipulation detection is a net loss, even if the weighted composite goes up.

### 2. Binary Keep/Discard → Three-Gate Evaluation

autoresearch: `if new_metric < old_metric: keep else: discard`

We need three gates, all of which must pass:

| Gate | What it checks | Why autoresearch doesn't need it |
|------|---------------|----------------------------------|
| **Composite improvement** | Weighted score across all axes (with Kappa downweighting) | autoresearch has one metric, no weighting needed |
| **Gold-set veto** | No single axis may degrade vs. human consensus labels | autoresearch has no per-axis regression risk |
| **Explanation gate** | Scorer must explain *why* it flagged axes (coverage ≥ 0.5) | Language model training has no explanation requirement |

The gold-set veto is critical: it uses **raw human labels** (not Kappa-adjusted weights) and has absolute authority. An experiment that improves composite by +10% is still rejected if any single axis degrades. autoresearch has no equivalent because val_bpb can't regress on one dimension while improving on another.

### 3. Run-a-Subprocess → Call-an-API

autoresearch runs `uv run train.py` as a subprocess for exactly 5 minutes of wall-clock training on a local GPU. The agent modifies model architecture, optimizer settings, hyperparameters — actual PyTorch training code.

Our train.py doesn't train anything locally. It builds **prompts for remote LLM APIs** (Hyperbolic, Ollama, Anthropic) that score email chains. The agent modifies prompting strategy, JSON parsing, thread signal extraction, and scoring logic. The evaluation calls external APIs, not local GPUs. This means:

- **Cost is per-API-call**, not per-GPU-minute → we need dollar budgets, not just time budgets.
- **Latency is network-bound**, not compute-bound → 5-minute fixed windows don't apply.
- **Multiple providers** with different capabilities → we need a provider abstraction layer.

### 4. No Human Calibration → Kappa-Based Downweighting

autoresearch has no human annotators. The eval function in prepare.py is deterministic and trusted.

Our ground truth comes from **human annotators** who disagree. Inter-annotator agreement varies by axis (humans agree on "is this phishing?" but disagree on "is this subtly toxic?"). We use Cohen's Kappa to measure agreement per axis and **downweight unreliable axes** in the composite score. This is why we have:

- `gold_set/calibration.json` — per-axis Kappa scores
- `get_effective_weights()` — adjusts axis weights proportional to annotator agreement
- The `calibrate-judge` data subcommand — a whole pipeline autoresearch doesn't need

### 5. No Explanation Requirement → Explanation Gate

autoresearch doesn't care *why* a model performs better. Lower val_bpb is sufficient.

Email trust scoring must be **explainable**. A system that says "this email is 0.9 phishing" without saying why is useless to a human reviewer. Our explanation gate checks that the scorer's `reasons` array covers all flagged axes (scores > 0.5). After the first baseline experiment, explanation quality must meet a threshold or the experiment is rejected — even if the composite score improved.

### 6. results.tsv → Structured Observability

autoresearch logs to a flat TSV file with columns: commit, val_bpb, memory, status.

We log to:
- `metrics.jsonl` — append-only, one JSON object per experiment with all axis scores, gate results, cost, wall time
- `predictions.jsonl` — full scorer outputs for each eval chain
- `config.json` — snapshot of spec.yaml at run start
- `summary.txt` — human-readable run summary
- structlog JSON output — structured logging throughout

Plus a **Gradio dashboard** with real-time charts, git diff viewer, and run history — because monitoring 10 axes across experiments requires more than a TSV file.

### 7. Pure Hill-Climbing → Hill-Climbing With Safety Rails

autoresearch is **greedy hill-climbing**: keep if better, discard if not. No safety nets. The worst case is wasted GPU time.

Our worst case is deploying a trust scorer that misclassifies phishing emails. So we added:

- **FP rate penalty** in the composite score (false positives on phish are costly)
- **Gold-set veto** with absolute authority (no axis may degrade)
- **Safety rules** in spec.yaml (minimum thresholds, maximum FP rates)
- **LoRA nudge** after 3 consecutive stalls (autoresearch has no plateau-breaking mechanism)

### 8. Single GPU → Multi-Provider Abstraction

autoresearch assumes one NVIDIA GPU running PyTorch. Our system abstracts across:

- **Ollama** (local generation)
- **Hyperbolic** (remote scoring + LoRA training with GPU budget guards)
- **Anthropic** (judging + agent calls)

Each with different auth, rate limits, error modes, and cost structures. The provider registry, retry decorator with dynamic transient error detection, and BudgetGuard are all infrastructure autoresearch doesn't need.

## What We Might Still Want From autoresearch

A few things autoresearch does that we haven't adopted (yet):

| Feature | autoresearch | Us | Worth adding? |
|---------|-------------|-----|---------------|
| **Fixed time budget per experiment** | 5 min wall-clock | Dollar + time limits on the full run, not per-experiment | Maybe — per-experiment time caps could prevent runaway API calls |
| **Parallel agents on branches** | 4 Claude + 4 Codex on separate branches | Single sequential loop | Yes — independent branches with different program.md strategies |
| **Simplicity criterion** | Prefers removing code over adding it for marginal gains | No equivalent | Yes — train.py bloat is a real risk over many iterations |
| **progress.png auto-generation** | Built-in visualization after each run | Gradio dashboard (richer but requires running) | Maybe — a static artifact is nice for async review |
| **Community multi-agent orchestration** | Ruflo, multi-agent Claude Code | Not implemented | Future — could run multiple scoring strategies in parallel |

## Summary

autoresearch is a beautiful minimal system for a single-metric optimization problem on local hardware. Our problem — multi-axis trust scoring with human calibration, explanation requirements, safety constraints, remote API providers, and rich observability — needs different machinery. The loop shape is the same; almost everything inside it is different.
