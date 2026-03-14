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
