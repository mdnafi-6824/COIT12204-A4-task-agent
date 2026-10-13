"""Guardrails. Some run before the model sees a request and some run after
it answers.

The input checks stop obviously bad input (too long, control characters,
prompt injection attempts) from ever reaching the model. The output checks
make sure a parsed AgentResult actually makes sense for the action. The JSON
schema can't do that on its own. A due date in the past is valid JSON but
it's a useless suggestion."""

import re
from datetime import timedelta

from app import config
from app.logging_config import get_logger
from app.models import AgentResult
from app.store import today

log = get_logger("guardrails")

# Phrases from prompt injection attempts. I keep these narrow on purpose:
# an early version matched "system prompt" on its own and rejected normal
# titles like "Update the system prompt docs".
_INJECTION_PATTERNS = [
    r"\bignore (all |the |any )?(previous|prior|above|earlier) instructions\b",
    r"\bdisregard (the |all |your )?(rules|instructions)\b",
    r"\b(reveal|print|show|repeat) (me )?(the |your )?system prompt\b",
    r"\byou are now (a|an|the) \w+ (assistant|model|ai)\b",
]
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputRejected(ValueError):
    """I raise this when input fails a hard check. The API turns it into a 400."""


def clean_text(value: str | None, field: str = "text") -> str:
    """Tidy up free text from the user. Hard limits raise, soft concerns just log."""
    if value is None:
        return ""
    text = _CONTROL_CHARS.sub("", str(value)).strip()
    if len(text) > config.MAX_INPUT_CHARS:
        raise InputRejected(f"{field} is too long ({len(text)} chars, limit {config.MAX_INPUT_CHARS}).")
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            log.warning("GUARDRAIL possible prompt injection in %s: %r", field, text[:80])
            raise InputRejected(f"{field} contains instructions aimed at the assistant and was rejected.")
    return text


def check_result(result: AgentResult, expected_action: str) -> tuple[AgentResult, list[str]]:
    """Check the model's structured output and fix it where that's safe to do.

    I return the (maybe corrected) result plus a list of warnings saying what
    was wrong. The warnings go back to the caller and into the log, so the
    evaluation can count how often the model needed correcting."""
    warnings: list[str] = []
    fixes: dict = {}

    if result.action != expected_action:
        warnings.append(f"action was '{result.action}', expected '{expected_action}'; corrected")
        fixes["action"] = expected_action

    if expected_action == "classify_priority" and result.priority is None:
        warnings.append("priority missing for classify_priority")

    if expected_action == "suggest_due_date":
        if result.suggested_due_date is None:
            warnings.append("suggested_due_date missing for suggest_due_date")
        else:
            if result.suggested_due_date < today():
                warnings.append(
                    f"suggested_due_date {result.suggested_due_date} is in the past; moved to today"
                )
                fixes["suggested_due_date"] = today()
            elif result.suggested_due_date > today() + timedelta(days=365):
                warnings.append("suggested_due_date more than a year away")

    if expected_action == "weekly_summary" and not (result.summary or "").strip():
        warnings.append("summary missing for weekly_summary")

    if fixes:
        result = result.model_copy(update=fixes)
    for w in warnings:
        log.warning("GUARDRAIL output: %s", w)
    return result, warnings
