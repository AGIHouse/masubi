# Masubi 10-Minute End-to-End Checklist

Date: 2026-03-14

This file is only the "make it run" list. Deferred review notes live in `docs/GPT_REVIEW.md`.

## Goal

Get one real end-to-end workflow working:

1. Stage 1 agent edits `train.py`
2. The edited scorer is actually evaluated
3. All three gates run without crashing
4. Stage 1 -> Stage 2 handoff produces teacher artifacts and labeled training data
5. Stage 2 trains a real dense baseline checkpoint
6. The checkpoint can be exported and scored locally

## Must Do Now

1. Make Stage 1 score the editable working copy, not the frozen template.

`run_loop.py` copies `starting_train.py` into `train.py`, but the Stage 1 scoring path still imports `EmailTrustScorer` from `starting_train.py` instead of `train.py`. That means the agent can edit `train.py` all day and evaluation never changes.

Where:
- `run_loop.py:596-600`
- `run_loop.py:724-733`

Fix:
- Keep `starting_train.py` as the canonical seed.
- Dynamically load `EmailTrustScorer` from the current `train.py` working copy during Stage 1 scoring.

Done when:
- A test proves that changing `train.py` changes Stage 1 outputs.

2. Fix the gold gate to score the gold set itself.

The current loop evaluates 1,000 eval-set predictions against 200 gold-set labels. Those datasets are different files with different IDs and zero overlap, so the current gate raises `ValueError: Found input variables with inconsistent numbers of samples: [200, 1000]`.

Where:
- `run_loop.py:752-777`
- `autotrust/eval.py:112-141`

Verified on repo data:
- `eval_set/eval_chains.jsonl`: 1000 rows
- `gold_set/gold_chains.jsonl`: 200 rows
- overlap in `chain_id`: 0

Fix:
- Load full gold chains, not just consensus labels.
- Score the gold chains separately with the same scorer/checkpoint.
- Track a separate `prev_best_gold_per_axis` baseline for Gate 2.

Done when:
- Running the live loop with the committed datasets reaches Gate 2 without crashing.

3. Align Stage 2 reason tags with the explanation gate.

Stage 2 scoring currently emits reason tags like `phish_flagged`, but the explanation gate checks for exact axis names like `phish`. Even good Stage 2 explanations will under-score or fail Gate 3.

Where:
- `run_loop.py:263-270`
- `autotrust/inference.py:102-111`
- `autotrust/eval.py:164-179`

Fix:
- Pick one contract and use it everywhere.
- Fastest path: make Stage 2 emit raw axis names in `reason_tags`.
- Alternative: normalize `_flagged` suffixes before explanation scoring.

Done when:
- A student output with flagged `phish` yields explanation quality `1.0` when reasons include `phish`.

4. Complete the Stage 1 -> Stage 2 handoff.

Auto-transition currently freezes teacher artifacts and rewrites `train.py`, but it does not relabel training data. Also, relabeling still imports `EmailTrustScorer` from `starting_train.py`, so it is not truly tied to the frozen best teacher state.

Where:
- `run_loop.py:141-160`
- `autotrust/freeze.py:390-405`

Fix:
- Call `relabel_training_data(...)` during auto-transition.
- Make relabeling use the frozen teacher artifacts or the best committed Stage 1 scorer, not the static template.

Done when:
- Transition produces both `teacher/*` artifacts and `synth_data/train_labeled.jsonl`.

5. Replace the Stage 2 checkpoint stub with a minimal real trainer.

The generated Stage 2 template saves an initialized checkpoint but does not train on data. That is enough for scaffolding, not for a complete workflow.

Where:
- `run_loop.py` generated Stage 2 template
- `train.py` after transition

Fix:
- Load `train_labeled.jsonl` (or explicit soft targets).
- Train a dense baseline for a small number of steps.
- Save `runs/latest/checkpoints/best.pt`.
- Enforce `validate_moe_config(...)` and `check_param_budget(...)` in the live path before any MoE work.

Done when:
- `uv run python train.py` after transition performs actual optimization and writes a trained checkpoint.

## Stop Here For Now

Do not spend this pass on:

- GGUF export
- MoE optimization
- tokenizer parity
- generator-provider improvements
- doc rewrites outside the live workflow

Those are important, but they are not what is currently preventing one complete run from working.

## Useful For A Real 10-Minute Demo

These are useful once the correctness blockers above are fixed.

1. Add an eval limiter for demo runs.

This is the most useful suggestion from the eval note. `run_loop.py` currently loads all rows from `eval_set/eval_chains.jsonl`, and Stage 1 scoring is sequential.

Where:
- `run_loop.py:76-87`
- `starting_train.py:36-38`

Best quick fix:
- add `--eval-limit` to `run_loop.py`
- slice `eval_chains = eval_chains[:eval_limit]`

Recommended demo default:
- `--eval-limit 100`

Why:
- this is the cheapest way to make one experiment finish inside a short live demo budget
- it does not solve the correctness bugs, but it does help once those are fixed

2. Parallel scoring is good, but it is not the first 10-minute fix.

Yes, the current scorer is sequential, so batch or parallel provider calls would help. But that is a larger code change than simply limiting eval rows for a demo path.

3. Do not rely on timeout behavior as the "solution."

`per_experiment_timeout_minutes` only prevents runaway experiments. By itself it does not make the loop complete useful work inside 10 minutes.

## Small Additions That Improve The Dashboard Fast

These are worth adding because the dashboard already has places to show them, or they create very obvious next-step visuals.

1. Log Stage 2 training metrics on every experiment.

The dashboard already supports Stage 2 charts for:

- `training_loss`
- `param_count`
- `expert_utilization`

Where:
- `dashboard.py:110-156`
- `autotrust/dashboard/charts.py:572-676`

Why:
- this gives immediate observability for whether Stage 2 is actually learning
- parameter-count-over-time is a very good visual for architecture search
- expert-utilization heatmaps are the most interesting MoE visualization already supported

Minimum useful payload per Stage 2 experiment:

```json
{
  "training_loss": {
    "trust_loss": 0.42,
    "reason_loss": 0.19,
    "escalate_loss": 0.11,
    "total_loss": 0.72
  },
  "param_count": 17677599
}
```

If MoE is active, also log:

```json
{
  "expert_utilization": [0.22, 0.31, 0.19, 0.28]
}
```

2. Log predictions for each experiment, not just aggregate metrics.

`observe.py` already has `log_predictions(...)`, but the main loop is not using it.

Where:
- `autotrust/observe.py:110-119`

Why:
- makes debugging concrete instead of abstract
- gives us the raw material for future dashboard visuals like score distributions, before/after chain comparisons, and top false positives
- helps explain why a gate failed

Minimum useful record shape:

```json
{
  "chain_id": "eval-000123",
  "trust_vector": {"phish": 0.91},
  "reasons": ["phish"],
  "kept": false
}
```

3. Add experiment phase timings and sample counts to the logged result.

Recommended fields:

- `agent_duration_sec`
- `scoring_duration_sec`
- `gold_scoring_duration_sec`
- `train_duration_sec`
- `eval_count`
- `gold_count`

Why:
- lets the dashboard explain where the 10 minutes are going
- makes `--eval-limit` effects visible
- gives a simple path to a later stacked-time or throughput chart

4. Add gold per-axis results, not only pass/fail.

Right now the dashboard can show gate failures, but not which axis on the gold set caused the veto.

Recommended fields:

- `gold_per_axis_scores`
- `gold_deltas`
- `failed_axes`

Why:
- turns the gold gate from a black box into a diagnosis tool
- makes it obvious which axis to optimize next
- would support a very good heatmap or veto-frequency chart later

5. For performance, make explanation quality visible during Stage 2 training, not just at final gating.

The repo already treats reasons as a real output head. If Stage 2 logs `reason_loss` and explanation quality together, we can tell whether better composite is coming from genuinely better structured reasoning or from gaming the score head.

## Smoke Check After Fixes

Run these in order:

```bash
uv run pytest tests/test_gold_gate.py tests/test_explanation_gate.py tests/test_stage_transition.py tests/test_inference.py tests/test_export.py
uv run python run_loop.py --max-experiments 1
uv run python -m autotrust.export --checkpoint runs/latest/checkpoints/best.pt --format pytorch
```
