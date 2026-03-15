# Stage 1: Prompt Optimization

You are improving a prompt-based email trust scorer. The scorer calls an LLM API
(Hyperbolic) and parses its JSON response into trust scores. You improve the
scoring prompt, thread signal extraction, and JSON parsing in train.py.

You MUST call the edit_train_py tool with the complete new train.py content.
Do not just describe changes -- write the code.

Rules:
- Only edit train.py
- Keep the EmailTrustScorer class interface (score_chain, score_batch)
- Output must be valid structured JSON: {"trust_vector": {...}, "explanation": {"reasons": [...], "summary": "..."}}
- The reasons array must reference flagged axes (scores > 0.5)
- Do not use regex backreferences in replacement strings
- Handle both plain JSON and code-fenced JSON responses from the LLM

Keep/discard has THREE gates (all must pass):
1. Composite score must improve (Kappa-adjusted weights + FP penalty)
2. Gold-set veto: no axis may degrade vs human consensus labels (all axes, including zero-weighted)
3. Explanation gate: after first baseline, explanation quality must be >= 0.5

Trust axes (10 total): phish, truthfulness, verify_by_search, manipulation, deceit,
vulnerability_risk, subtle_toxicity, polarization, classic_email_metrics, authority_impersonation.

What to improve in the prompt:
1. Better scoring prompt -- be specific about what each axis means, give the LLM concrete examples
2. Thread signal extraction -- detect urgency patterns, authority shifts, escalation across emails
3. Explanation quality -- prompt the LLM to explain which axes are flagged and why
4. JSON parsing robustness -- handle malformed responses gracefully

Make a concrete change now.

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
