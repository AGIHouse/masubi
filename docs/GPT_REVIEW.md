# GPT Review: Masubi Sanity, DRYness, Architecture, and ML/PyTorch Assessment

Review date: 2026-03-14

Repository reviewed: `~/personal/musubi`

This file is the deferred review. The immediate execution list is in `docs/10min.md`.

## Bottom Line

Masubi has the right fixed-platform shape, but it is still in a partially transitioned state between a Stage 1 prompt loop and a Stage 2 student-model loop. The project does not need a rewrite. It needs integration cleanup and one unambiguous runnable baseline.

## What Changed Since The Earlier Review

One earlier critical finding is no longer the live problem:

- Stage 1 no longer imports `EmailTrustScorer` from `train.py`.

But the replacement behavior introduced a worse integration bug:

- Stage 1 now imports `EmailTrustScorer` from `starting_train.py`, so edits to `train.py` are not actually evaluated.

That is the current top sanity issue.

## Live Critical Issues

1. Stage 1 edits are inert.

`run_loop.py` copies `starting_train.py` into `train.py`, but Stage 1 scoring still imports from `starting_train.py`. The mutable file and the evaluated file are not the same thing.

2. The gold gate is still broken on real repo data.

The loop feeds 1,000 eval predictions into a 200-row gold set. The committed datasets have zero `chain_id` overlap, so Gate 2 raises on mismatched sample counts.

3. Stage 2 explanations do not match Gate 3's contract.

Stage 2 emits `*_flagged` tags while `explanation_quality(...)` expects exact axis names.

4. Stage 2 handoff is incomplete.

Auto-transition freezes artifacts and rewrites `train.py`, but it does not relabel training data. The relabeling helper also still uses `starting_train.py`, not the frozen best teacher.

## High-Value Work That Can Wait Until After The Loop Runs

1. Turn Stage 2 into a real dense-baseline trainer.

The current Stage 2 template is a training scaffold that exports an initialized checkpoint. It is good enough for plumbing, not for a meaningful baseline.

2. Enforce Stage 2 limits in the live path.

`validate_moe_config(...)` and `check_param_budget(...)` exist, but the main Stage 2 execution path does not use them.

3. Remove the most important duplication.

The Stage 2 template is duplicated between `run_loop.py` and the generated `train.py` shape. Axis count and reason-tag count are also hardcoded rather than derived from `spec.yaml`.

4. Unify escalation semantics.

One path uses the explicit escalate head. Another reconstructs escalation heuristically from axis scores. Pick one contract.

5. Decide whether synthetic data is template-only or provider-backed.

`build_train(...)` detects the generator provider, but `_generate_synth_chain(...)` ignores it and always uses built-in templates.

## Architecture Assessment

### Strong

- `spec.yaml` is the right source of truth for trust axes and platform constraints.
- `autotrust/eval.py` is correctly positioned as a frozen evaluator.
- `schemas.py` centralizes the important typed contracts.
- The provider split and observability split are reasonable for this problem.

### Weak

The repo currently has multiple competing definitions of "what stage the system is in":

- Stage 1 mutable file: `train.py`
- Stage 1 executed scorer: `starting_train.py`
- Stage 2 auto-generated template: emitted from `run_loop.py`
- teacher freeze/relabel path: partly tied to git history, partly tied to `starting_train.py`

That state-model duplication is the real architecture problem.

## DRYness Assessment

The repo is mostly DRY in the evaluator and config layers. The biggest duplication problems are not random copy-paste; they are duplicated sources of truth.

Highest-impact DRY issues:

1. Stage 1 has two scorer sources: `starting_train.py` and `train.py`.
2. Stage 2 template logic lives inside `run_loop.py` instead of in a shared module or seed file.
3. Axis and reason-tag counts are hardcoded in multiple places.
4. Explanation/reason-tag naming is split between inference and evaluation.

## ML / PyTorch Assessment

### Dense Student

The dense student is a credible scaffold:

- transformer encoder backbone
- separate trust, reason, and escalation heads
- export/load path
- local inference path

What is still missing:

- a real training loop
- real teacher distillation during handoff
- tokenizer parity between training and local inference
- calibration and objective shaping beyond the current stub

### MoE

The MoE direction is fine as a later optimization, but it should stay later.

Reasons to postpone:

- the dense baseline is not established yet
- capacity and routing validation are still light
- nested Python dispatch loops are not iteration-speed friendly
- hard caps are not enforced in the main loop

### PyTorch Note

The recurring `TransformerEncoder` nested-tensor warning is a performance smell, not a correctness bug. It is worth cleaning up once the baseline loop is real.

## Documentation Assessment

The docs still describe a cleaner and more coherent status than the current code actually provides. The most helpful posture is:

- `docs/10min.md`: live blocker checklist
- `README.md`: only claim workflows that run today
- historical redesign docs: keep as intent, not status

## Recommendation

Priority order:

1. Make Stage 1 evaluate the working copy `train.py`.
2. Score the gold set separately and track a gold-only baseline.
3. Normalize Stage 2 reason tags to the explanation-gate contract.
4. Make auto-transition produce labeled teacher data.
5. Turn Stage 2 into a minimal real dense trainer.

After that, clean up duplication and only then spend time on MoE, GGUF, tokenizer parity, and provider-backed synthetic generation.

## Local Verification Notes

Verified locally from the current repo state:

- `eval_set/eval_chains.jsonl` has 1000 rows.
- `gold_set/gold_chains.jsonl` has 200 rows.
- The two datasets have zero `chain_id` overlap.
- Running `gold_regression_gate(...)` against those live files raises `ValueError: Found input variables with inconsistent numbers of samples: [200, 1000]`.
- A targeted test pass for run-loop, stage-transition, train, freeze, export, inference, and smoke coverage finished with 68 passes and 1 failing callback test (`tests/test_run_loop.py::test_pause_check_callback_blocks`), which is not itself the main end-to-end blocker.
