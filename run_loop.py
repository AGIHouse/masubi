"""Thin orchestration for the autoresearch loop.

Drives the research loop: loads spec/calibration, starts run, iterates
(agent prompt -> edit train.py -> score -> three-gate eval -> keep/discard),
enforces budget/time limits, and logs everything.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from autotrust.config import get_spec
from autotrust.observe import (
    finalize_run,
    log_experiment,
    start_run,
)
from autotrust.schemas import (
    CalibrationReport,
    EmailChain,
    ExperimentResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_calibration() -> CalibrationReport:
    """Load calibration report from gold_set/calibration.json.

    Returns default (perfect Kappa) if file doesn't exist yet.
    """
    cal_path = Path("gold_set/calibration.json")
    if cal_path.exists():
        data = json.loads(cal_path.read_text())
        return CalibrationReport(**data)

    # Default: no downweighting
    spec = get_spec()
    return CalibrationReport(
        per_axis_kappa={a.name: 1.0 for a in spec.trust_axes},
        effective_weights={a.name: a.weight for a in spec.trust_axes},
        flagged_axes=[],
        downweight_amounts={},
    )


def load_eval_chains() -> list[EmailChain]:
    """Load evaluation chains from eval_set/eval_chains.jsonl."""
    path = Path("eval_set/eval_chains.jsonl")
    if not path.exists():
        logger.warning("No eval chains found at %s", path)
        return []

    chains = []
    for line in path.read_text().strip().split("\n"):
        if line:
            chains.append(EmailChain.model_validate_json(line))
    return chains


def load_gold_chains() -> list[dict[str, float]]:
    """Load gold set consensus labels from gold_set/gold_chains.jsonl."""
    path = Path("gold_set/gold_chains.jsonl")
    if not path.exists():
        logger.warning("No gold chains found at %s", path)
        return []

    chains = []
    for line in path.read_text().strip().split("\n"):
        if line:
            data = json.loads(line)
            chains.append(data.get("consensus_labels", data))
    return chains


def _check_budget(total_cost: float, spec: Any) -> bool:
    """Return True if budget is exceeded."""
    return total_cost >= spec.limits.max_spend_usd


def _check_time_limit(start_time: float, spec: Any) -> bool:
    """Return True if time limit is exceeded."""
    elapsed_minutes = (time.time() - start_time) / 60
    return elapsed_minutes >= spec.limits.experiment_minutes


def _build_agent_prompt(
    program_md: str,
    train_py: str,
    last_results: list[dict],
    consecutive_no_improvement: int,
) -> str:
    """Build the prompt for the research agent."""
    prompt = f"""## Instructions
{program_md}

## Current train.py
```python
{train_py}
```

## Last Experiment Results
{json.dumps(last_results[-3:], indent=2, default=str) if last_results else 'No previous results.'}
"""
    if consecutive_no_improvement >= 3:
        prompt += """
## IMPORTANT: LoRA Fine-Tuning Nudge
You have had 3+ consecutive experiments with no improvement.
Consider using LoRA fine-tuning via TrainingProvider to break through the plateau.
Call self.fine_tune(data_path, trainer) to start a fine-tuning run.
Remember to auto-terminate GPUs when done.
"""
    return prompt


def _handle_keep_discard(keep: bool, experiment_num: int) -> None:
    """Handle git keep/discard for train.py."""
    if keep:
        subprocess.run(["git", "add", "train.py"], check=False)
        subprocess.run(
            ["git", "commit", "-m", f"experiment {experiment_num}: keep"],
            check=False,
        )
        logger.info("Experiment %d: KEPT (committed train.py)", experiment_num)
    else:
        subprocess.run(["git", "checkout", "--", "train.py"], check=False)
        logger.info("Experiment %d: DISCARDED (restored train.py)", experiment_num)


def _log_iteration(ctx: Any, result: ExperimentResult) -> None:
    """Log a single experiment iteration."""
    log_experiment(ctx, result)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_autoresearch(max_experiments: int = 50) -> None:
    """Run the autoresearch loop.

    1. Load spec, calibration, and data
    2. For each experiment:
       a. Call agent with program.md + train.py + last results
       b. Apply proposed edit to train.py
       c. Score eval chains
       d. Run three-gate evaluation
       e. Keep/discard via git
       f. Log everything
    3. Enforce budget/time limits
    4. Finalize run
    """
    spec = get_spec()
    _calibration = load_calibration()  # noqa: F841 -- used in full implementation
    run_ctx = start_run(spec)

    eval_chains = load_eval_chains()
    _gold_chains = load_gold_chains()  # noqa: F841 -- used in full implementation

    _has_baseline = False  # noqa: F841 -- used in full implementation
    _prev_best_composite = 0.0  # noqa: F841 -- used in full implementation
    _prev_best_per_axis: dict[str, float] = {}  # noqa: F841 -- used in full implementation
    consecutive_no_improvement = 0
    total_cost = 0.0
    all_results: list[dict] = []

    start_time = time.time()

    for experiment_num in range(1, max_experiments + 1):
        # Check limits
        if _check_time_limit(start_time, spec):
            logger.info("Time limit reached. Stopping.")
            break
        if _check_budget(total_cost, spec):
            logger.info("Budget limit reached. Stopping.")
            break

        if not eval_chains:
            logger.warning("No eval chains available. Stopping.")
            break

        logger.info("--- Experiment %d ---", experiment_num)

        # Read current files
        program_md = Path("program.md").read_text()
        train_py = Path("train.py").read_text()

        # Build agent prompt
        prompt = _build_agent_prompt(
            program_md, train_py, all_results, consecutive_no_improvement
        )

        # In production: call Sonnet via Anthropic API with tool-use
        # For now, use current train.py as-is (agent editing is external)
        logger.info("Agent prompt built (%d chars). Awaiting edit.", len(prompt))

        # Score eval chains using current train.py
        # This would be done after the agent edits train.py
        # For now, we break -- the actual loop requires real LLM calls
        logger.info("Experiment loop requires LLM agent. Exiting placeholder loop.")
        break

    finalize_run(run_ctx)
    logger.info("Autoresearch loop complete. %d experiments run.", len(all_results))


if __name__ == "__main__":
    run_autoresearch()
