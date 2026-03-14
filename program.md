# Stage 1: Prompt Optimization

You are optimizing a content-only email trust scorer.

Rules:
- Only edit train.py
- Budget: see spec.yaml limits (currently 15 min / $8)
- Base model: see spec.yaml providers.scorer (currently Llama-3.1-8B on Hyperbolic)

Keep/discard has THREE gates (all must pass):
1. Composite score must improve (Kappa-adjusted weights + FP penalty)
2. Gold-set veto: no axis may degrade vs human consensus labels (all axes, including zero-weighted)
3. Explanation gate: after first baseline, explanation quality must be >= 0.5

The gold-set veto has absolute authority and uses raw human labels (not
Kappa-adjusted). An experiment that improves composite by +10% will still be
rejected if it degrades any single axis. Do not chase composite improvements
that ignore per-axis quality.

Your scorer must output structured JSON with both trust_vector and explanation:
  {"trust_vector": {...}, "explanation": {"reasons": [...], "summary": "..."}}
The reasons array must reference flagged axes (scores > 0.5). This is tested.

Trust axes, weights, and thresholds are in spec.yaml.

Priorities:
1. Thread encoder: per-email embeddings -> attention over thread -> chain classifier
2. Multi-task heads for fast axes (phish, manipulation, classic, verify_by_search)
3. Explanation reasons: must cover all flagged axes (this is gated after baseline!)
4. When gains stall: LoRA fine-tune via TrainingProvider (auto-terminate GPUs)

Start now.

---

# Stage 2: Student Model Training

You are training a compact student model (50-200M params) to replace the LLM scorer.
The system auto-transitions to this stage after 3 consecutive no-improvement experiments,
or you can start here directly with `--stage train`.

## What you optimize (in train.py):
- Model architecture: hidden size, depth, number of layers
- Loss weighting across trust axes
- Optimizer, learning rate schedule, batch size
- After dense baseline: MoE expert count, routing strategy, capacity factor, which layers are sparse

## What you cannot change:
- The dataset (frozen Stage 1 outputs in teacher/)
- The teacher labels (soft trust vectors)
- The evaluation harness (eval.py)
- The gold set
- MoE caps in spec.yaml (max_experts=16, max_params_m=200M, max_top_k=4)

## Architecture search path:
1. Establish a dense baseline FIRST (prove it converges)
2. Only then introduce MoE layers
3. Start with few experts (4-8), increase only if gains stall

## Training data:
- All Stage 1 synthetic + real labeled chains (synth_data/)
- Soft teacher scores (not hard labels) as training targets
- Explanation tags as auxiliary supervision signal

## Output shape:
train.py must produce a PyTorch checkpoint that, when loaded, outputs:
  {"trust_vector": {...}, "reason_tags": [...], "escalate": true/false}

## Keep/discard gates:
Same three gates apply. Composite score must improve, gold-set veto must pass,
explanation quality must meet threshold. The evaluation uses the student model
checkpoint instead of the LLM API.

## Budget:
See spec.yaml limits (currently 10 min / $8 per experiment for Stage 2).
Use TrainingProvider to rent Hyperbolic H100s. Auto-terminate GPUs when done.
