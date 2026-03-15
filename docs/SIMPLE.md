# Masubi -- Simple Explanation

Masubi is an autoresearch-style loop for email trust scoring. An AI agent proposes changes to one file, we score 1,000 email chains across 10 trust dimensions (not binary spam/not-spam), and a three-gate policy accepts or git-reverts each experiment.

## What's Built

The evaluation infrastructure (10-axis scoring, three-gate keep/discard, Kappa downweighting, explanation quality gating), the data pipeline, providers (Ollama/Hyperbolic/Anthropic), a Gradio dashboard, and model definitions for dense + MoE student models.

## What's Not Done

Human annotation of the gold set (it's machine-labeled), the Stage 2 training loop integration, and the teacher-to-student handoff. The system can run Stage 1 prompt optimization end-to-end, but Stage 2 model training is scaffolded, not functional.

## The Core Bet

Traditional phishing detection is binary and solved. The unsolved problem is nuanced trust -- manipulation, authority impersonation, subtle toxicity. We evaluate across 10 axes with a gold-set veto that blocks any single-axis regression, even if the overall score improves. The agent can be creative but can't game the metrics because it can't touch the evaluation contract.

## Current State (2026-03-14)

- **Data**: 1,000 eval chains, 200 gold chains (machine-labeled, not human-annotated), 20 synthetic training examples
- **Stage 1**: 4 runs exist, loop runs end-to-end but no meaningful score improvements yet
- **Stage 2**: 17.7M-param dense baseline checkpoint exists at composite 0.0 (initialized, not trained). Teacher directory empty. `--stage train` is scaffolded but not wired to actual training
- **Gold set**: No human annotation done, no Kappa calibration computed
