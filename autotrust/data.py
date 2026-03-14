"""Data generation and calibration module for AutoEmailTrust v3.5.

CLI subcommands:
    uv run python -m autotrust.data build-train --count 5000
    uv run python -m autotrust.data build-eval
    uv run python -m autotrust.data build-gold
    uv run python -m autotrust.data annotate-export
    uv run python -m autotrust.data calibrate-judge --annotations <path>
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.metrics import cohen_kappa_score

if TYPE_CHECKING:
    from autotrust.config import Spec
    from autotrust.schemas import Email

logger = logging.getLogger(__name__)

# Known real brand names for safety filtering
KNOWN_BRANDS = {
    "paypal", "google", "microsoft", "apple", "amazon", "facebook", "meta",
    "netflix", "ebay", "chase", "wellsfargo", "bankofamerica", "citibank",
    "walmart", "target", "instagram", "twitter", "linkedin", "dropbox",
    "adobe", "zoom", "slack", "github", "stripe", "venmo", "cashapp",
}

# Operational instruction patterns (to be blocked)
OPERATIONAL_PATTERNS = [
    r"reverse\s+shell",
    r"c2\s+server",
    r"malware\.exe",
    r"payload\s+from",
    r"execute\s+as\s+administrator",
    r"exploit\s+the\s+vulnerability",
    r"sql\s+injection.*exec",
    r"buffer\s+overflow.*shellcode",
    r"download.*trojan",
    r"keylogger.*install",
]


# ---------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------

def safety_filter(email: Email, is_synth: bool, spec: Spec) -> bool:
    """Filter emails based on safety rules from spec.yaml.

    Returns True if email passes (is safe to include), False if rejected.
    """
    text = f"{email.subject} {email.body}".lower()

    # Block operational instructions
    if spec.safety.block_operational_instructions:
        for pattern in OPERATIONAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Blocked email with operational instructions: %s", pattern)
                return False

    # Synth data: placeholder brands only
    if is_synth and spec.safety.synth_placeholder_only:
        for brand in KNOWN_BRANDS:
            if brand in text:
                logger.warning("Blocked synth email with real brand: %s", brand)
                return False

    # Eval data: real brands allowed
    # (no additional filtering needed -- spec.safety.real_brands_in_eval)

    return True


# ---------------------------------------------------------------------------
# Cohen's Kappa computation
# ---------------------------------------------------------------------------

def compute_cohen_kappa(
    annotator_1: list[float],
    annotator_2: list[float],
) -> float:
    """Compute Cohen's Kappa between two annotators.

    For continuous scores, binarizes at 0.5 threshold.
    """
    y1 = [1 if s >= 0.5 else 0 for s in annotator_1]
    y2 = [1 if s >= 0.5 else 0 for s in annotator_2]

    if y1 == y2:
        return 1.0

    try:
        return float(cohen_kappa_score(y1, y2))
    except Exception:
        return 0.0


def flag_low_kappa_axes(
    kappa_per_axis: dict[str, float],
    spec: Spec,
) -> list[str]:
    """Flag axes where Kappa is below min_gold_kappa threshold."""
    min_kappa = spec.judge.min_gold_kappa
    return [
        axis_name
        for axis_name, kappa in kappa_per_axis.items()
        if kappa < min_kappa
    ]


# ---------------------------------------------------------------------------
# Data pipeline subcommands
# ---------------------------------------------------------------------------

def build_train(count: int, spec: Spec) -> Path:
    """Build training dataset from real corpora + synthetic generation.

    Pipeline: seed -> safety filter -> Evol-Instruct -> SpearBot critic
    -> dual-judge labeling -> dedup.
    Output: synth_data/train.jsonl
    """

    output_path = Path("synth_data/train.jsonl")
    logger.info("Building training set: %d chains -> %s", count, output_path)

    # In full implementation:
    # 1. Load real corpora (SpamAssassin, Enron)
    # 2. Generate synthetic via GeneratorProvider
    # 3. Apply safety_filter
    # 4. Evol-Instruct augmentation
    # 5. SpearBot critic
    # 6. Dual-judge labeling
    # 7. Dedup
    # 8. Write to output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Training set pipeline complete (placeholder)")
    return output_path


def build_eval(spec: Spec) -> Path:
    """Build evaluation dataset.

    Size: spec.data.eval_set_size chains.
    Real brands allowed per spec.safety.real_brands_in_eval.
    Output: eval_set/eval_chains.jsonl
    """
    output_path = Path("eval_set/eval_chains.jsonl")
    logger.info("Building eval set: %d chains -> %s", spec.data.eval_set_size, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Eval set pipeline complete (placeholder)")
    return output_path


def build_gold(spec: Spec) -> Path:
    """Build gold-set candidates for human annotation.

    Size: spec.data.gold_set_size chains.
    Selects diverse chains covering all axis_groups.
    Output: gold_set/gold_candidates.jsonl
    """
    output_path = Path("gold_set/gold_candidates.jsonl")
    logger.info("Building gold set candidates: %d chains -> %s", spec.data.gold_set_size, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Gold set candidate generation complete (placeholder)")
    return output_path


def annotate_export(spec: Spec) -> Path:
    """Export gold candidates in annotator-friendly format.

    Output: gold_set/annotate_export.jsonl
    """
    output_path = Path("gold_set/annotate_export.jsonl")
    logger.info("Exporting for annotation -> %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Annotation export complete (placeholder)")
    return output_path


def calibrate_judge(annotations_path: str, spec: Spec) -> Path:
    """Calibrate judge against human annotations.

    Ingests human annotations, computes Cohen's Kappa per axis,
    flags axes below min_gold_kappa, writes calibration.json.
    """
    output_path = Path("gold_set/calibration.json")
    logger.info("Calibrating judge from %s -> %s", annotations_path, output_path)

    # In full implementation:
    # 1. Load annotations from annotations_path
    # 2. Compute per-axis Kappa using compute_cohen_kappa()
    # 3. Flag low axes using flag_low_kappa_axes()
    # 4. Compute effective weights
    # 5. Write CalibrationReport to output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Judge calibration complete (placeholder)")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for data subcommands."""
    from autotrust.config import load_spec

    parser = argparse.ArgumentParser(
        prog="autotrust.data",
        description="Data generation and calibration for AutoEmailTrust v3.5",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build-train
    p_train = subparsers.add_parser("build-train", help="Generate training data")
    p_train.add_argument("--count", type=int, default=5000, help="Number of chains")

    # build-eval
    subparsers.add_parser("build-eval", help="Generate evaluation data")

    # build-gold
    subparsers.add_parser("build-gold", help="Generate gold-set candidates")

    # annotate-export
    subparsers.add_parser("annotate-export", help="Export for annotation")

    # calibrate-judge
    p_cal = subparsers.add_parser("calibrate-judge", help="Calibrate judge")
    p_cal.add_argument("--annotations", required=True, help="Path to annotations")

    args = parser.parse_args()
    spec = load_spec()

    if args.command == "build-train":
        build_train(args.count, spec)
    elif args.command == "build-eval":
        build_eval(spec)
    elif args.command == "build-gold":
        build_gold(spec)
    elif args.command == "annotate-export":
        annotate_export(spec)
    elif args.command == "calibrate-judge":
        calibrate_judge(args.annotations, spec)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
