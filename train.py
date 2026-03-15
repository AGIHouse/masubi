"""EmailTrustScorer -- improved prompt, thread signals, JSON parsing, and Stage 2 student training.

Produces structured ScorerOutput with trust_vector (dict) and Explanation
(reasons array + summary). Includes dense baseline student model training.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

from autotrust.schemas import Explanation, ScorerOutput

if TYPE_CHECKING:
    from autotrust.config import Spec
    from autotrust.providers import ScoringProvider, TrainingProvider
    from autotrust.schemas import EmailChain

logger = logging.getLogger(__name__)

# ── Trust axis definitions for the scoring prompt ────────────────────────────
AXIS_DEFINITIONS = {
    "phish": (
        "Phishing risk: Does the email attempt to steal credentials, personal info, "
        "or financial data? Look for: spoofed sender domains (e.g. 'paypa1.com' vs 'paypal.com'), "
        "deceptive links, fake login pages, urgent account alerts ('your account will be suspended'), "
        "requests to click suspicious URLs, or requests to 'verify' credentials. "
        "Example HIGH (0.9): 'Dear Customer, your PayPal account has been compromised. Click here to verify.' "
        "Example LOW (0.1): 'Hi team, here are the Q3 budget figures as discussed.' "
        "Score 0.0 = clearly safe, 1.0 = textbook phishing."
    ),
    "truthfulness": (
        "Truthfulness: Are the factual claims in the email accurate and verifiable? "
        "Look for fabricated statistics, false deadlines, invented policies, misattributed quotes, "
        "or claims that contradict well-known facts. "
        "Example HIGH (0.8): 'Studies show 99% of users prefer our product' (no source). "
        "Example LOW (0.1): 'Our meeting is scheduled for 3pm Tuesday per the calendar invite.' "
        "Score 0.0 = all claims appear truthful, 1.0 = pervasive falsehoods."
    ),
    "verify_by_search": (
        "Verify-by-search: Could the claims be fact-checked with a simple web search? "
        "Score high if the email makes claims that are either unverifiable or contradicted by public info. "
        "Example HIGH (0.8): 'The FDA just banned all vitamin supplements effective immediately.' "
        "Example LOW (0.1): 'Google announced Q3 earnings yesterday.' "
        "Score 0.0 = claims are easily verifiable and likely true, 1.0 = claims are "
        "unverifiable or contradicted by public information."
    ),
    "manipulation": (
        "Manipulation: Does the email use psychological pressure tactics? Look for: "
        "artificial urgency ('act now or lose access'), fear appeals, guilt-tripping, "
        "love-bombing, reciprocity exploitation, social proof fabrication, or scarcity tactics. "
        "Example HIGH (0.9): 'If you don't wire the money in 2 hours, your family will suffer consequences.' "
        "Example LOW (0.1): 'Please review the attached document when you get a chance.' "
        "Score 0.0 = no manipulation, 1.0 = heavy manipulation."
    ),
    "deceit": (
        "Deceit: Is the sender being deliberately deceptive about their identity, "
        "intentions, or the nature of the communication? Look for impersonation, "
        "hidden agendas, misleading subject lines, or bait-and-switch tactics. "
        "Example HIGH (0.9): Email claims to be from IT dept but sender is external gmail address. "
        "Example LOW (0.1): A colleague sending a genuine project update. "
        "Score 0.0 = transparent, 1.0 = deeply deceptive."
    ),
    "vulnerability_risk": (
        "Vulnerability risk: Does the email target vulnerable populations (elderly, "
        "children, non-native speakers, financially distressed)? Look for simplified "
        "language designed to confuse, exploitation of trust, lottery/prize scams, "
        "fake debt collection, or targeting specific vulnerable demographics. "
        "Example HIGH (0.8): 'Congratulations grandma! You won $1M. Just send $50 processing fee.' "
        "Example LOW (0.1): 'Team standup notes from today's meeting.' "
        "Score 0.0 = no targeting, 1.0 = clearly exploitative."
    ),
    "subtle_toxicity": (
        "Subtle toxicity: Does the email contain veiled hostility, passive aggression, "
        "microaggressions, backhanded compliments, or coded discriminatory language? "
        "Not overt hate speech, but insidious negativity. "
        "Example HIGH (0.7): 'That's a really good idea... for someone at your level.' "
        "Example LOW (0.1): 'Great work on the presentation!' "
        "Score 0.0 = no toxicity, 1.0 = pervasively toxic."
    ),
    "polarization": (
        "Polarization: Does the email push extreme viewpoints, use us-vs-them framing, "
        "demonize outgroups, or attempt to radicalize the reader? Look for divisive "
        "rhetoric, conspiracy theories, or inflammatory generalizations. "
        "Example HIGH (0.8): 'Those people are destroying our country and we must stop them at all costs.' "
        "Example LOW (0.1): 'Here is a balanced summary of the policy proposals.' "
        "Score 0.0 = balanced/neutral, 1.0 = highly polarizing."
    ),
    "classic_email_metrics": (
        "Classic email red flags: Check for spam indicators -- ALL CAPS subject/body, excessive "
        "punctuation (!!!), too-good-to-be-true offers, unsolicited attachments, "
        "mismatched reply-to addresses, generic greetings ('Dear Customer'), broken formatting, "
        "or suspicious sender names. "
        "Example HIGH (0.9): 'CONGRATULATIONS!!! YOU HAVE WON $5,000,000!!! CLICK HERE NOW!!!' "
        "Example LOW (0.1): 'Hi Sarah, attached is the quarterly report. Best, John.' "
        "Score 0.0 = professional email, 1.0 = obvious spam."
    ),
    "authority_impersonation": (
        "Authority impersonation: Does the sender falsely claim to be from a position "
        "of authority (CEO, IT admin, government official, bank representative)? "
        "Look for domain mismatches, unusual requests from 'leadership', claims "
        "of official capacity without verification, or BEC (business email compromise) patterns. "
        "Example HIGH (0.9): Email from 'ceo@gmai1.com' saying 'I need you to buy gift cards urgently.' "
        "Example LOW (0.1): Email from verified company domain with routine instructions. "
        "Score 0.0 = no impersonation, 1.0 = clear authority impersonation."
    ),
}


class EmailTrustScorer:
    """Thread-aware email trust scorer with structured explanation output."""

    def __init__(self, provider: ScoringProvider, spec: Spec) -> None:
        self.provider = provider
        self.spec = spec

    def score_chain(self, chain: EmailChain) -> ScorerOutput:
        """Score a single email chain. Returns trust vector + structured explanation."""
        prompt = self._build_prompt(chain)
        raw_response = self.provider.score(prompt)
        return self._parse_response(raw_response)

    def score_batch(self, chains: list[EmailChain]) -> list[ScorerOutput]:
        """Score multiple chains (sequential)."""
        return [self.score_chain(c) for c in chains]

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_prompt(self, chain: EmailChain) -> str:
        """Build a thread-aware scoring prompt with detailed axis definitions,
        thread signals, and explicit output format instructions."""

        # Build chain text with thread context
        thread_parts = []
        for i, email in enumerate(chain.emails):
            thread_parts.append(
                f"--- Email {i + 1} (depth={email.reply_depth}) ---\n"
                f"From: {email.from_addr}\n"
                f"To: {email.to_addr}\n"
                f"Subject: {email.subject}\n"
                f"Time: {email.timestamp}\n\n"
                f"{email.body}"
            )
        thread_text = "\n\n".join(thread_parts)

        # Thread encoder signals
        signals = self._extract_thread_signals(chain)
        signals_text = "\n".join(f"  - {s}" for s in signals) if signals else "  None detected"

        # Build detailed axis descriptions
        axis_descriptions = []
        for a in self.spec.trust_axes:
            desc = AXIS_DEFINITIONS.get(a.name, f"{a.name}: Score 0.0 (safe) to 1.0 (maximum concern).")
            axis_descriptions.append(f"  {a.name} (weight={a.weight}): {desc}")
        axes_block = "\n".join(axis_descriptions)

        # Get axis names for the JSON template
        axis_names = [a.name for a in self.spec.trust_axes]
        example_vector = ", ".join(f'"{name}": 0.0' for name in axis_names)

        prompt = f"""You are an expert email security analyst specializing in phishing detection, social engineering, and trust assessment. Your job is to analyze email threads for trust and safety risks with high precision.

TASK: Score the following email chain on each trust axis below. Each score is a float from 0.0 (completely safe / no concern) to 1.0 (maximum risk / highest concern). Be precise -- use the full range of scores, not just 0.0 and 1.0.

TRUST AXES (score each one carefully using the definitions and examples):
{axes_block}

THREAD SIGNALS (automatically detected patterns -- use these as supporting evidence):
{signals_text}

EMAIL CHAIN TO ANALYZE:
{thread_text}

SCORING INSTRUCTIONS:
1. Read the ENTIRE email chain carefully, considering how the conversation evolves across replies.
2. Cross-reference the automatically detected thread signals with your own analysis.
3. For each axis, assign a precise score between 0.0 and 1.0 based on the definitions and examples above.
4. Consider interactions between axes: e.g., phishing often co-occurs with authority_impersonation and manipulation.
5. In the explanation, you MUST list EVERY axis that scores above 0.5 with a specific reason citing evidence from the email.
6. If NO axis scores above 0.5, explain why the email appears safe.
7. The summary should be a single sentence capturing the overall risk level and primary concerns.

CRITICAL: Your response must be ONLY a valid JSON object. No markdown code fences, no extra text before or after the JSON. Just the raw JSON object.

Required output format (replace placeholder values with your scores and analysis):
{{"trust_vector": {{{example_vector}}}, "explanation": {{"reasons": ["axis_name: specific reason citing evidence from the email"], "summary": "One sentence overall risk assessment"}}}}"""
        return prompt

    # ── Thread signal extraction ─────────────────────────────────────────────

    def _extract_thread_signals(self, chain: EmailChain) -> list[str]:
        """Extract thread-aware signals from the email chain with enhanced detection."""
        signals = []

        if len(chain.emails) > 1:
            # Reply timing analysis
            for i in range(1, len(chain.emails)):
                prev_time = chain.emails[i - 1].timestamp
                curr_time = chain.emails[i].timestamp
                delta = (curr_time - prev_time).total_seconds()
                if delta < 60:
                    signals.append(
                        f"VERY rapid reply at email {i + 1} ({delta:.0f}s) -- "
                        "possible automated or pre-scripted response"
                    )
                elif delta < 300:
                    signals.append(f"Rapid reply at email {i + 1} ({delta:.0f}s)")

            # Thread depth escalation
            depths = [e.reply_depth for e in chain.emails]
            if len(set(depths)) > 1:
                signals.append(f"Thread depth escalation: {depths}")

            # Authority shifts: detect when a new, more authoritative sender appears
            senders = [e.from_addr for e in chain.emails]
            unique_senders = list(dict.fromkeys(senders))  # preserve order, dedupe
            if len(unique_senders) > 1:
                signals.append(f"Sender changes in thread: {unique_senders}")
                # Check if later senders claim higher authority
                for i in range(1, len(chain.emails)):
                    if chain.emails[i].from_addr != chain.emails[i - 1].from_addr:
                        signals.append(
                            f"Authority shift at email {i + 1}: "
                            f"{chain.emails[i - 1].from_addr} -> {chain.emails[i].from_addr}"
                        )

            # Escalation in tone: check if urgency increases over the thread
            urgency_per_email = []
            urgency_keywords = [
                "urgent", "immediately", "asap", "right now", "deadline",
                "time-sensitive", "act now", "expires", "final notice",
                "last chance", "don't delay", "do not delay"
            ]
            for email in chain.emails:
                text = (email.subject + " " + email.body).lower()
                count = sum(1 for w in urgency_keywords if w in text)
                urgency_per_email.append(count)

            if len(urgency_per_email) > 1 and urgency_per_email[-1] > urgency_per_email[0]:
                signals.append(
                    f"Urgency escalation across thread: {urgency_per_email} "
                    "(later emails more urgent)"
                )

            # Persuasion progression: requests become more specific/demanding
            request_keywords = [
                "please send", "wire transfer", "click here", "download",
                "provide your", "confirm your", "verify your", "update your",
                "send me", "share your", "give me", "transfer"
            ]
            requests_per_email = []
            for email in chain.emails:
                text = (email.subject + " " + email.body).lower()
                count = sum(1 for w in request_keywords if w in text)
                requests_per_email.append(count)
            if len(requests_per_email) > 1 and requests_per_email[-1] > requests_per_email[0]:
                signals.append(
                    f"Request escalation: later emails contain more action demands "
                    f"({requests_per_email})"
                )

            # Subject line changes (potential topic hijacking)
            subjects = [e.subject.strip().lower() for e in chain.emails]
            unique_subjects = list(dict.fromkeys(subjects))
            if len(unique_subjects) > 1:
                # Check if subject changed significantly (not just Re: prefix)
                base_subjects = []
                for s in subjects:
                    cleaned = s
                    while cleaned.startswith("re:") or cleaned.startswith("fw:") or cleaned.startswith("fwd:"):
                        cleaned = cleaned.split(":", 1)[-1].strip()
                    base_subjects.append(cleaned)
                if len(set(base_subjects)) > 1:
                    signals.append(
                        f"Subject line changed mid-thread: possible topic hijacking"
                    )

        # ── Single-email and cross-thread signals ────────────────────────────

        all_text = " ".join(e.subject + " " + e.body for e in chain.emails).lower()

        # Urgency keywords (expanded)
        urgency_words = [
            "urgent", "immediately", "asap", "right now", "deadline",
            "time-sensitive", "act now", "expires", "final notice",
            "last chance", "don't delay", "do not delay", "hours left",
            "limited time", "respond immediately", "time is running out",
            "must act", "cannot wait", "expiring soon"
        ]
        found_urgency = [w for w in urgency_words if w in all_text]
        if found_urgency:
            signals.append(f"Urgency signals: {found_urgency}")

        # Authority claims (expanded)
        authority_words = [
            "ceo", "cfo", "cto", "coo", "director", "manager",
            "admin", "it department", "help desk", "helpdesk",
            "president", "vice president", "vp", "chairman",
            "compliance", "legal department", "hr department",
            "security team", "system administrator", "board of directors",
            "executive", "chief", "superintendent", "commissioner"
        ]
        found_authority = [w for w in authority_words if w in all_text]
        if found_authority:
            signals.append(f"Authority claims: {found_authority}")

        # Financial / credential requests
        financial_words = [
            "bank account", "routing number", "wire transfer", "bitcoin",
            "cryptocurrency", "gift card", "payment", "invoice",
            "social security", "ssn", "credit card", "password",
            "login credentials", "pin number", "account number",
            "sort code", "iban", "swift code", "tax id",
            "w-2", "w2", "direct deposit"
        ]
        found_financial = [w for w in financial_words if w in all_text]
        if found_financial:
            signals.append(f"Financial/credential requests detected: {found_financial}")

        # Link/attachment signals
        link_patterns = ["click here", "click below", "click this link",
                         "open the attachment", "see attached", "download",
                         "follow this link", "visit this page", "log in here",
                         "sign in here", "update your account"]
        found_links = [w for w in link_patterns if w in all_text]
        if found_links:
            signals.append(f"Link/attachment action requests: {found_links}")

        # Emotional manipulation
        emotional_words = [
            "disappointed in you", "i trusted you", "you owe me",
            "don't let me down", "i'm counting on you", "this is your fault",
            "you'll regret", "consequences", "disciplinary action",
            "i'm very disappointed", "you should be ashamed",
            "everyone is watching", "your reputation"
        ]
        found_emotional = [w for w in emotional_words if w in all_text]
        if found_emotional:
            signals.append(f"Emotional manipulation signals: {found_emotional}")

        # Domain mismatch detection
        if len(chain.emails) > 0:
            from_domains = set()
            for email in chain.emails:
                addr = email.from_addr.lower()
                if "@" in addr:
                    from_domains.add(addr.split("@")[-1])
            if len(from_domains) > 1:
                signals.append(
                    f"Multiple sender domains detected: {sorted(from_domains)} "
                    "(possible spoofing)"
                )

            # Check for suspicious domain patterns (typosquatting indicators)
            suspicious_tlds = [".xyz", ".top", ".click", ".loan", ".win", ".gq", ".ml", ".tk"]
            for domain in from_domains:
                for tld in suspicious_tlds:
                    if domain.endswith(tld):
                        signals.append(f"Suspicious TLD in sender domain: {domain}")
                        break

        # ALL CAPS detection
        all_raw = " ".join(e.subject + " " + e.body for e in chain.emails)
        words = all_raw.split()
        caps_words = [w for w in words if len(w) > 3 and w.isupper()]
        if len(caps_words) > 3:
            signals.append(f"Excessive ALL CAPS words detected ({len(caps_words)} words)")

        # Excessive punctuation
        exclamation_count = all_raw.count("!")
        question_count = all_raw.count("?")
        if exclamation_count > 5:
            signals.append(f"Excessive exclamation marks: {exclamation_count} found")
        if question_count > 5:
            signals.append(f"Excessive question marks: {question_count} found")

        # Generic greetings
        generic_greetings = ["dear customer", "dear user", "dear account holder",
                             "dear valued customer", "dear sir/madam", "dear friend",
                             "dear member", "dear recipient"]
        found_generic = [g for g in generic_greetings if g in all_text]
        if found_generic:
            signals.append(f"Generic greeting detected: {found_generic}")

        # Too-good-to-be-true offers
        offer_words = ["congratulations", "you have won", "you've won", "prize",
                       "lottery", "million dollars", "free gift", "claim your",
                       "you are selected", "you have been selected", "inheritance"]
        found_offers = [w for w in offer_words if w in all_text]
        if found_offers:
            signals.append(f"Too-good-to-be-true offers: {found_offers}")

        # Confidentiality pressure
        secrecy_words = [
            "keep this confidential", "do not share", "between us",
            "don't tell anyone", "secret", "private matter",
            "do not discuss", "keep this between"
        ]
        found_secrecy = [w for w in secrecy_words if w in all_text]
        if found_secrecy:
            signals.append(f"Secrecy/confidentiality pressure: {found_secrecy}")

        return signals

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> ScorerOutput:
        """Parse LLM response into ScorerOutput with robust fallback handling."""
        data = None

        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: Strip code fences (```json ... ``` or ``` ... ```)
        if data is None:
            cleaned = raw.strip()
            # Remove opening code fence
            if cleaned.startswith("```"):
                # Remove first line (```json or ```)
                first_newline = cleaned.find("\n")
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            # Remove closing code fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()
                cleaned = cleaned[:-3].rstrip()

            try:
                data = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: Find the outermost JSON object with brace matching
        if data is None:
            data = self._extract_json_object(raw)

        # Strategy 4: Try stripping any text before/after the JSON object
        if data is None:
            # Find first { and last }
            first_brace = raw.find("{")
            last_brace = raw.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                candidate = raw[first_brace:last_brace + 1]
                try:
                    data = json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Strategy 5: Try to fix common JSON issues
        if data is None:
            cleaned = raw.strip()
            # Remove any leading/trailing non-JSON text
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1:
                cleaned = cleaned[first_brace:last_brace + 1]
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            # Fix single quotes to double quotes
            cleaned = cleaned.replace("'", '"')
            try:
                data = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 6: Try to find just the trust_vector portion
        if data is None:
            tv_match = re.search(r'"trust_vector"\s*:\s*\{([^}]+)\}', raw)
            if tv_match:
                try:
                    tv_str = "{" + tv_match.group(1) + "}"
                    trust_vector = json.loads(tv_str)
                    # Try to find explanation too
                    exp_data = {"reasons": [], "summary": "Partial parse - trust vector only."}
                    exp_match = re.search(r'"explanation"\s*:\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', raw)
                    if exp_match:
                        try:
                            exp_str = "{" + exp_match.group(1) + "}"
                            exp_data = json.loads(exp_str)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    data = {"trust_vector": trust_vector, "explanation": exp_data}
                except (json.JSONDecodeError, TypeError):
                    pass

        if data is None:
            logger.error("Failed to parse response as JSON: %s", raw[:500])
            return self._default_output()

        # Extract trust vector
        trust_vector = data.get("trust_vector", {})
        explanation_data = data.get("explanation", {})

        # Handle case where explanation is a string instead of dict
        if isinstance(explanation_data, str):
            explanation_data = {"reasons": [], "summary": explanation_data}

        # Handle case where explanation is a list (reasons only)
        if isinstance(explanation_data, list):
            explanation_data = {"reasons": explanation_data, "summary": ""}

        # Ensure all axes are present with defaults
        axis_names = {a.name for a in self.spec.trust_axes}
        for axis in self.spec.trust_axes:
            if axis.name not in trust_vector:
                trust_vector[axis.name] = 0.0

        # Clamp values to [0, 1] and filter to known axes
        clean_vector = {}
        for k, v in trust_vector.items():
            if k in axis_names:
                try:
                    clean_vector[k] = max(0.0, min(1.0, float(v)))
                except (ValueError, TypeError):
                    clean_vector[k] = 0.0

        # Ensure all axes present after filtering
        for axis in self.spec.trust_axes:
            if axis.name not in clean_vector:
                clean_vector[axis.name] = 0.0

        # Build explanation: ensure reasons reference flagged axes (score > 0.5)
        raw_reasons = explanation_data.get("reasons", [])
        if not isinstance(raw_reasons, list):
            raw_reasons = [str(raw_reasons)]

        flagged_axes = [name for name, score in clean_vector.items() if score > 0.5]

        # Ensure all flagged axes are mentioned in reasons
        mentioned_axes = set()
        valid_reasons = []
        for reason in raw_reasons:
            if isinstance(reason, str) and reason.strip():
                valid_reasons.append(reason.strip())
                # Track which axes are mentioned
                for axis_name in flagged_axes:
                    if axis_name in reason.lower() or axis_name.replace("_", " ") in reason.lower():
                        mentioned_axes.add(axis_name)

        # Add missing flagged axes to reasons with descriptive text
        for axis_name in flagged_axes:
            if axis_name not in mentioned_axes:
                score = clean_vector[axis_name]
                axis_def = AXIS_DEFINITIONS.get(axis_name, "")
                # Extract first sentence of definition for context
                short_desc = axis_def.split(".")[0] if axis_def else axis_name
                valid_reasons.append(
                    f"{axis_name}: flagged with score {score:.2f} - {short_desc}"
                )

        summary = explanation_data.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            if flagged_axes:
                summary = f"Email flagged for: {', '.join(flagged_axes)}. Review recommended."
            else:
                summary = "No significant trust concerns detected in this email chain."

        explanation = Explanation(
            reasons=valid_reasons,
            summary=summary,
        )

        return ScorerOutput(trust_vector=clean_vector, explanation=explanation)

    def _extract_json_object(self, text: str) -> dict | None:
        """Extract the outermost JSON object using brace-matching."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            c = text[i]

            if escape_next:
                escape_next = False
                continue

            if c == "\\":
                if in_string:
                    escape_next = True
                continue

            if c == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try fixing common issues in the candidate
                        fixed = re.sub(r",\s*}", "}", candidate)
                        fixed = re.sub(r",\s*]", "]", fixed)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            return None

        return None

    def _default_output(self) -> ScorerOutput:
        """Return default output when parsing fails."""
        return ScorerOutput(
            trust_vector={a.name: 0.0 for a in self.spec.trust_axes},
            explanation=Explanation(reasons=[], summary="Failed to parse LLM response."),
        )

    # ── Stage 2: Student model training ──────────────────────────────────────

    def fine_tune(self, data_path: str, trainer: TrainingProvider) -> str:
        """Train a dense student model on teacher-generated soft labels.

        Returns the path to the best checkpoint.
        """
        import glob
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader

        # Try to import autotrust student utilities
        try:
            from autotrust.student import DenseStudent
            from autotrust.export import export_pytorch
            from autotrust.schemas import StudentConfig, CheckpointMeta
            USE_AUTOTRUST_STUDENT = True
        except ImportError:
            USE_AUTOTRUST_STUDENT = False
            logger.info("autotrust.student not available, using built-in model")

        # Determine run ID and create output directory
        run_id = os.environ.get("RUN_ID", "run_0")
        checkpoint_dir = os.path.join("runs", run_id, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, "best.pt")

        # ── Load teacher data ────────────────────────────────────────────
        teacher_files = glob.glob(os.path.join(data_path, "*.json"))
        if not teacher_files:
            teacher_files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)

        axis_names = [a.name for a in self.spec.trust_axes]
        num_axes = len(axis_names)

        samples = []
        for fpath in teacher_files:
            try:
                with open(fpath, "r") as f:
                    record = json.load(f)

                # Extract text features from email chain
                chain_text = ""
                if "emails" in record:
                    for email in record["emails"]:
                        chain_text += email.get("subject", "") + " " + email.get("body", "") + " "
                elif "text" in record:
                    chain_text = record["text"]
                elif "chain" in record:
                    chain_text = str(record["chain"])

                # Extract teacher scores (soft labels)
                trust_vector = record.get("trust_vector", record.get("scores", {}))
                scores = [float(trust_vector.get(name, 0.0)) for name in axis_names]

                # Extract reason tags and escalate flag
                explanation = record.get("explanation", {})
                if isinstance(explanation, str):
                    explanation = {"reasons": [], "summary": explanation}
                reason_tags = explanation.get("reasons", [])

                # Build reason tag binary vector (which axes are mentioned in reasons)
                reason_tag_vector = [0.0] * num_axes
                for tag in reason_tags:
                    if isinstance(tag, str):
                        tag_lower = tag.lower()
                        for idx, axis_name in enumerate(axis_names):
                            if axis_name in tag_lower:
                                reason_tag_vector[idx] = 1.0

                # Escalate if any high-risk axis > 0.7
                escalate = any(s > 0.7 for s in scores)

                samples.append({
                    "text": chain_text.strip(),
                    "scores": scores,
                    "reason_tags": reason_tag_vector,
                    "escalate": escalate,
                })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Skipping malformed teacher file %s: %s", fpath, e)
                continue

        if not samples:
            logger.error("No training samples found in %s", data_path)
            raise RuntimeError(f"No training samples found in {data_path}")

        logger.info("Loaded %d training samples", len(samples))

        # ── Simple text tokenizer (word-level) ───────────────────────────
        vocab = {"<PAD>": 0, "<UNK>": 1}
        # Build vocabulary with frequency filtering
        word_freq = {}
        for sample in samples:
            for word in sample["text"].lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add words with freq >= 2 (or all if small dataset)
        min_freq = 2 if len(samples) > 100 else 1
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq and len(vocab) < 50000:
                vocab[word] = len(vocab)

        max_seq_len = 512
        vocab_size = len(vocab)

        def tokenize(text: str) -> list[int]:
            tokens = []
            for word in text.lower().split()[:max_seq_len]:
                tokens.append(vocab.get(word, 1))
            # Pad
            while len(tokens) < max_seq_len:
                tokens.append(0)
            return tokens[:max_seq_len]

        # ── Dataset ──────────────────────────────────────────────────────
        class TrustDataset(Dataset):
            def __init__(self, samples_list):
                self.samples = samples_list

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                input_ids = torch.tensor(tokenize(s["text"]), dtype=torch.long)
                scores = torch.tensor(s["scores"], dtype=torch.float32)
                escalate = torch.tensor(1.0 if s["escalate"] else 0.0, dtype=torch.float32)
                reason_tags = torch.tensor(s["reason_tags"], dtype=torch.float32)
                return input_ids, scores, escalate, reason_tags

        # ── Dense Student Model ──────────────────────────────────────────
        class DenseStudentModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_axes_out,
                         num_layers=4, dropout=0.1, nhead=8):
                super().__init__()
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.positional = nn.Embedding(max_seq_len, embed_dim)

                # Layer norm after embedding
                self.embed_ln = nn.LayerNorm(embed_dim)

                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers,
                    norm=nn.LayerNorm(embed_dim),
                )

                # Trust vector head (main output)
                self.trust_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_axes_out),
                    nn.Sigmoid(),
                )

                # Escalation head
                self.escalate_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )

                # Reason tag head (multi-label: one per axis)
                self.reason_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_axes_out),
                    nn.Sigmoid(),
                )

            def forward(self, input_ids):
                B, L = input_ids.shape
                positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

                x = self.embedding(input_ids) + self.positional(positions)
                x = self.embed_ln(x)

                # Create padding mask
                padding_mask = (input_ids == 0)
                x = self.encoder(x, src_key_padding_mask=padding_mask)

                # Pool: mean of non-padded tokens
                mask = (~padding_mask).unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

                trust_vector = self.trust_head(pooled)
                escalate = self.escalate_head(pooled).squeeze(-1)
                reason_tags = self.reason_head(pooled)

                return trust_vector, escalate, reason_tags

        # ── Training configuration ───────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s", device)

        embed_dim = 192
        hidden_dim = 384
        num_layers = 4
        nhead = 8  # Must divide embed_dim
        dropout = 0.1
        batch_size = 32
        learning_rate = 3e-4
        num_epochs = 80
        patience = 15
        warmup_epochs = 5

        model = DenseStudentModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_axes_out=num_axes,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Student model has %.2fM parameters", total_params / 1e6)
        if total_params > 200e6:
            raise ValueError(
                f"Model has {total_params / 1e6:.1f}M params, exceeding 200M limit"
            )

        # Loss weights per axis (from spec)
        axis_weights = torch.tensor(
            [a.weight for a in self.spec.trust_axes], dtype=torch.float32
        ).to(device)
        # Normalize weights so they sum to num_axes
        axis_weights = axis_weights / axis_weights.sum() * num_axes

        # Split into train/val (stratified-ish by escalate flag)
        escalate_samples = [s for s in samples if s["escalate"]]
        safe_samples = [s for s in samples if not s["escalate"]]

        val_esc = escalate_samples[:max(1, len(escalate_samples) // 10)]
        val_safe = safe_samples[:max(1, len(safe_samples) // 10)]
        train_esc = escalate_samples[len(val_esc):]
        train_safe = safe_samples[len(val_safe):]

        train_samples = train_esc + train_safe
        val_samples = val_esc + val_safe

        # Fallback if stratification produced empty sets
        if not train_samples or not val_samples:
            val_size = max(1, len(samples) // 10)
            val_samples = samples[:val_size]
            train_samples = samples[val_size:]

        logger.info("Train: %d samples, Val: %d samples", len(train_samples), len(val_samples))

        train_loader = DataLoader(
            TrustDataset(train_samples), batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            TrustDataset(val_samples), batch_size=batch_size, shuffle=False,
            num_workers=0, drop_last=False,
        )

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

        # Warmup + cosine annealing schedule
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        mse_loss_fn = nn.MSELoss(reduction="none")
        bce_loss_fn = nn.BCELoss()

        # Huber loss for robustness to noisy teacher labels
        huber_loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.1)

        # ── Training loop ────────────────────────────────────────────────
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for input_ids, scores, escalate, reason_tags in train_loader:
                input_ids = input_ids.to(device)
                scores = scores.to(device)
                escalate = escalate.to(device)
                reason_tags = reason_tags.to(device)

                pred_trust, pred_escalate, pred_reasons = model(input_ids)

                # Weighted Huber loss for trust vector (robust to noise)
                trust_loss = (huber_loss_fn(pred_trust, scores) * axis_weights).mean()

                # MSE component for precision
                mse_component = (mse_loss_fn(pred_trust, scores) * axis_weights).mean()

                # Combined trust loss
                combined_trust_loss = 0.7 * trust_loss + 0.3 * mse_component

                # Escalation loss
                esc_loss = bce_loss_fn(pred_escalate, escalate)

                # Reason tag auxiliary loss
                reason_loss = bce_loss_fn(pred_reasons, reason_tags)

                # Total loss with balancing
                loss = combined_trust_loss + 0.3 * esc_loss + 0.2 * reason_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * input_ids.size(0)
                train_count += input_ids.size(0)

            scheduler.step()

            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            val_trust_mae_sum = 0.0

            with torch.no_grad():
                for input_ids, scores, escalate, reason_tags in val_loader:
                    input_ids = input_ids.to(device)
                    scores = scores.to(device)
                    escalate = escalate.to(device)
                    reason_tags = reason_tags.to(device)

                    pred_trust, pred_escalate, pred_reasons = model(input_ids)

                    trust_loss = (huber_loss_fn(pred_trust, scores) * axis_weights).mean()
                    mse_component = (mse_loss_fn(pred_trust, scores) * axis_weights).mean()
                    combined_trust_loss = 0.7 * trust_loss + 0.3 * mse_component

                    esc_loss = bce_loss_fn(pred_escalate, escalate)
                    reason_loss = bce_loss_fn(pred_reasons, reason_tags)
                    loss = combined_trust_loss + 0.3 * esc_loss + 0.2 * reason_loss

                    val_loss_sum += loss.item() * input_ids.size(0)
                    val_count += input_ids.size(0)

                    # Track MAE for trust vector
                    val_trust_mae_sum += torch.abs(pred_trust - scores).mean().item() * input_ids.size(0)

            avg_train = train_loss_sum / max(train_count, 1)
            avg_val = val_loss_sum / max(val_count, 1)
            avg_mae = val_trust_mae_sum / max(val_count, 1)
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_mae=%.4f  lr=%.6f",
                epoch + 1, num_epochs, avg_train, avg_val, avg_mae, current_lr,
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                # Save best checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "axis_names": axis_names,
                    "config": {
                        "vocab_size": vocab_size,
                        "embed_dim": embed_dim,
                        "hidden_dim": hidden_dim,
                        "num_axes": num_axes,
                        "num_layers": num_layers,
                        "nhead": nhead,
                        "dropout": dropout,
                        "max_seq_len": max_seq_len,
                    },
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "total_params": total_params,
                }
                torch.save(checkpoint, best_path)
                logger.info("Saved best checkpoint at epoch %d (val_loss=%.4f)", epoch + 1, best_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        logger.info("Training complete. Best val_loss=%.4f. Checkpoint: %s", best_val_loss, best_path)
        return best_path

    def load_fine_tuned(self, checkpoint: str) -> None:
        """Load a trained student checkpoint for inference."""
        import torch
        import torch.nn as nn

        ckpt = torch.load(checkpoint, map_location="cpu")
        config = ckpt["config"]

        self._student_vocab = ckpt["vocab"]
        self._student_axis_names = ckpt["axis_names"]
        self._student_config = config

        max_seq_len = config["max_seq_len"]

        # Rebuild model
        class DenseStudentModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_axes_out,
                         num_layers=4, dropout=0.1, nhead=8, seq_len=512):
                super().__init__()
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.positional = nn.Embedding(seq_len, embed_dim)
                self.embed_ln = nn.LayerNorm(embed_dim)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers,
                    norm=nn.LayerNorm(embed_dim),
                )

                self.trust_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_axes_out),
                    nn.Sigmoid(),
                )
                self.escalate_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                self.reason_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_axes_out),
                    nn.Sigmoid(),
                )
                self.max_seq_len = seq_len

            def forward(self, input_ids):
                B, L = input_ids.shape
                positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
                x = self.embedding(input_ids) + self.positional(positions)
                x = self.embed_ln(x)
                padding_mask = (input_ids == 0)
                x = self.encoder(x, src_key_padding_mask=padding_mask)
                mask = (~padding_mask).unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                trust_vector = self.trust_head(pooled)
                escalate = self.escalate_head(pooled).squeeze(-1)
                reason_tags = self.reason_head(pooled)
                return trust_vector, escalate, reason_tags

        model = DenseStudentModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_axes_out=config["num_axes"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            nhead=config.get("nhead", 8),
            seq_len=config["max_seq_len"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._student_model = model
        logger.info("Loaded student model from %s (%.2fM params)",
                     checkpoint, ckpt.get("total_params", 0) / 1e6)

    def score_chain_student(self, chain: "EmailChain") -> dict:
        """Score using the loaded student model. Returns dict with trust_vector, reason_tags, escalate."""
        import torch

        if not hasattr(self, "_student_model"):
            raise RuntimeError("No student model loaded. Call load_fine_tuned() first.")

        model = self._student_model
        vocab = self._student_vocab
        axis_names = self._student_axis_names
        config = self._student_config
        max_seq_len = config["max_seq_len"]

        # Tokenize the chain
        chain_text = ""
        for email in chain.emails:
            chain_text += email.subject + " " + email.body + " "
        chain_text = chain_text.strip().lower()

        tokens = []
        for word in chain_text.split()[:max_seq_len]:
            tokens.append(vocab.get(word, 1))
        while len(tokens) < max_seq_len:
            tokens.append(0)
        tokens = tokens[:max_seq_len]

        input_ids = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            trust_vec, escalate_score, reason_scores = model(input_ids)

        trust_vector = {}
        for i, name in enumerate(axis_names):
            trust_vector[name] = float(trust_vec[0, i].item())

        reason_tags = []
        for i, name in enumerate(axis_names):
            if reason_scores[0, i].item() > 0.5:
                reason_tags.append(name)

        escalate = bool(escalate_score[0].item() > 0.5)

        return {
            "trust_vector": trust_vector,
            "reason_tags": reason_tags,
            "escalate": escalate,
        }
