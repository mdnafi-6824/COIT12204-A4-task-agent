"""The Reviewer agent.

It accepts one intent, review_result, and answers the orchestrator with
either result_approved or revision_requested.

It checks the Worker's answer in two layers:

    1. Rules. Deterministic checks against the real task data: a due date
       can't be in the past, a priority has to agree with the due date
       rubric, a summary has to mention the overdue work, and an answer
       that came from the Worker's own fallback is sent back once so the
       model gets another go.
    2. Model. If enabled, the same model is asked to judge the answer and
       reply with a small JSON verdict. If that call fails or times out the
       Reviewer carries on with the rules only, so the system never stalls
       because the second model call broke.

Rules run first and always. The model can only add rejections, never
override a rule rejection. That keeps the Reviewer predictable."""

import re
from datetime import date
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app import config
from app.agent import _extract_json_object, _text, get_llm
from app.agents.base import BaseAgent
from app.logging_config import get_logger
from app.messages import AgentMessage
from app.store import TaskStore, today

log = get_logger("reviewer")


class ReviewVerdict(BaseModel):
    approved: bool = Field(description="true if the answer is correct and useful, false otherwise")
    issues: list[str] = Field(default_factory=list, description="Each concrete problem found, one sentence each. Empty if approved.")
    feedback: Optional[str] = Field(default=None, description="One or two sentences telling the worker exactly what to fix. Null if approved.")


VERDICT_PARSER = PydanticOutputParser(pydantic_object=ReviewVerdict)

REVIEWER_SYSTEM = (
    "You are a strict reviewer inside a task manager. Another agent has produced an answer "
    "about a task. Check it against the task data and today's date ({today}). Reject it if the "
    "priority does not match the urgency, if a suggested date is in the past or unrealistic, if "
    "the summary misses overdue work or names tasks that do not exist, or if the reasoning "
    "contradicts the task data. Approve it if it is correct and clear. Do not reject for style.\n\n"
    "Reply with ONLY a JSON object, no other text.\n\n{format_instructions}"
)


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def __init__(self, store: TaskStore, use_llm: Optional[bool] = None, llm=None):
        super().__init__()
        self.store = store
        self.use_llm = config.REVIEWER_USE_LLM if use_llm is None else use_llm
        self._llm = llm  # built lazily so the rules-only mode never touches a provider

    # ------------------------------------------------------------------
    def process(self, message: AgentMessage) -> AgentMessage:
        content = message.content
        result = content.get("result") or {}
        task = content.get("task")
        action = result.get("action") or (content.get("request") or {}).get("action")

        issues = self._rule_check(action, result, task, content)
        checked_by = "rules"
        # Second improvement after the evaluation: when the worker's model is
        # down, its revised answer is still the rule fallback, and the model
        # reviewer kept rejecting it because the reasoning says "fallback".
        # Every conversation then hit max revisions. The rules have already
        # decided to accept a second fallback, so the model has nothing to add.
        accepted_fallback = bool(content.get("fallback_used") and content.get("revision"))
        if not issues and self.use_llm and not accepted_fallback:
            llm_issues, model_answered = self._llm_check(action, result, self._context_for(action, task, content))
            if model_answered:
                checked_by = "rules+model"
            if llm_issues:
                issues.extend(llm_issues)
                checked_by = "model"

        if issues:
            feedback = " ".join(issues)
            log.warning("REVIEW rejected (%s): %s", checked_by, feedback)
            return message.reply(
                sender="reviewer", receiver="orchestrator", intent="revision_requested",
                content={"issues": issues, "feedback": feedback, "checked_by": checked_by, "result": result},
                step=message.step + 1,
            )
        approved: dict[str, Any] = {"result": result, "checked_by": checked_by}
        if accepted_fallback:
            approved["warning"] = "accepted the worker's rule based fallback after one retry"
            log.warning("REVIEW approved (%s) %s", checked_by, approved["warning"])
        else:
            log.info("REVIEW approved (%s)", checked_by)
        return message.reply(
            sender="reviewer", receiver="orchestrator", intent="result_approved",
            content=approved,
            step=message.step + 1,
        )

    def fallback(self, message: AgentMessage, reason: str) -> Optional[AgentMessage]:
        """If the Reviewer itself breaks, I approve with a warning rather than
        block the whole system. The result is still the Worker's real answer."""
        result = (message.content or {}).get("result") or {}
        log.warning("REVIEW skipped (%s), approving unreviewed", reason)
        return message.reply(
            sender="reviewer", receiver="orchestrator", intent="result_approved",
            content={"result": result, "checked_by": "none", "warning": f"reviewer unavailable: {reason}"},
            step=message.step + 1,
        )

    # ------------------------------------------------------------------
    # Layer 1: rules
    # ------------------------------------------------------------------
    def _rule_check(self, action: str, result: dict, task: Optional[dict], content: dict) -> list[str]:
        issues: list[str] = []
        if content.get("fallback_used") and not content.get("revision"):
            # first attempt came from the rule fallback, ask the model once more
            issues.append("The answer came from the worker's fallback rules, not the model. Try again and use the task data.")
            return issues

        if not (result.get("reasoning") or "").strip():
            issues.append("Reasoning is empty.")
        conf = result.get("confidence")
        if conf is None or not (0 <= float(conf) <= 1):
            issues.append("Confidence must be between 0 and 1.")

        due = _parse_date(task.get("due_date")) if task else None
        if action == "classify_priority":
            priority = result.get("priority")
            if task and priority is None:
                issues.append("Priority is missing even though the task exists.")
            elif task and due is not None:
                expected = _rubric_priority(due)
                if priority != expected and not task.get("priority"):
                    issues.append(
                        f"Task is due {due.isoformat()} ({(due - today()).days} days from today), the rubric says "
                        f"'{expected}' but the answer says '{priority}'."
                    )
        elif action == "suggest_due_date":
            d = _parse_date(result.get("suggested_due_date"))
            if d is None:
                issues.append("No suggested_due_date was given.")
            elif d < today():
                issues.append(f"Suggested date {d.isoformat()} is in the past. Today is {today().isoformat()}.")
            elif due is not None and due < today() and (d - today()).days > 3:
                issues.append(f"The task is already overdue, so {d.isoformat()} is too far away. Suggest today or within 3 days.")
            elif (d - today()).days > config.REVIEWER_MAX_DAYS_AHEAD:
                issues.append(f"Suggested date {d.isoformat()} is more than {config.REVIEWER_MAX_DAYS_AHEAD} days away.")
        elif action == "weekly_summary":
            summary = (result.get("summary") or "").strip()
            if not summary:
                issues.append("Summary is empty.")
            else:
                overdue = self.store.overdue()
                if overdue and "overdue" not in summary.lower():
                    issues.append(f"There are {len(overdue)} overdue task(s) but the summary does not mention anything overdue.")
                days = int((content.get("request") or {}).get("days_ahead") or 7)
                due_soon = self.store.due_within(days)[:3]
                if due_soon and not any(_mentions(summary, t.title) for t in due_soon):
                    titles = ", ".join(t.title for t in due_soon)
                    issues.append(f"The summary should name at least one of the most urgent tasks: {titles}.")
        return issues

    def _context_for(self, action: str, task: Optional[dict], content: dict) -> dict:
        """What the model reviewer gets to check against. Improvement after
        the first evaluation: for a weekly summary there is no single task, so
        the model saw "task data: None" and rejected every summary as invented.
        Now it gets the same task list the worker's tools returned. For a free
        text task it gets the title and description from the request."""
        request = content.get("request") or {}
        if action == "weekly_summary":
            days = int(request.get("days_ahead") or 7)
            all_tasks = self.store.all()
            by_status: dict[str, int] = {}
            for t in all_tasks:
                by_status[t.status] = by_status.get(t.status, 0) + 1
            return {
                "today": today().isoformat(),
                "days_ahead": days,
                "open_tasks_due_or_overdue": [t.model_dump(mode="json") for t in self.store.due_within(days)],
                "overdue_count": len(self.store.overdue()),
                # the worker also sees these totals through summarise_tasks, so
                # the reviewer has to see them too or it calls them invented
                "total_tasks": len(all_tasks),
                "tasks_by_status": by_status,
                "open_tasks_total": len(self.store.open_tasks()),
            }
        if task is None:
            return {"unsaved_task": {"title": request.get("title"), "description": request.get("description")},
                    "note": "This task is not saved, so there is no record to look up. Judge from the title and description."}
        return {"task": task, "today": today().isoformat()}

    # ------------------------------------------------------------------
    # Layer 2: the model
    # ------------------------------------------------------------------
    def _llm_check(self, action: str, result: dict, context: dict) -> tuple[list[str], bool]:
        """Returns (issues, model_answered). model_answered is False when the
        call failed or the verdict could not be parsed, so the caller knows
        the approval came from the rules alone."""
        try:
            if self._llm is None:
                self._llm = get_llm()
            system = REVIEWER_SYSTEM.format(today=today().isoformat(), format_instructions=VERDICT_PARSER.get_format_instructions())
            human = (
                f"Action: {action}\nContext (the real data the answer must agree with): {context}\n"
                f"Answer to review: {result}\n"
                "Is this answer correct and useful given that context? Reply with the JSON verdict only."
            )
            reply = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            raw = _text(reply)
            verdict = self._parse_verdict(raw)
            if verdict is None:
                log.warning("REVIEW model verdict unparseable, using rules only | raw=%r", raw[:160])
                return [], False
            if verdict.approved:
                return [], True
            issues = verdict.issues or ([verdict.feedback] if verdict.feedback else ["The reviewer model rejected the answer."])
            return [i for i in issues if i], True
        except Exception as exc:  # noqa: BLE001  the second model call must never take the system down
            log.error("REVIEW model error (%s: %s), using rules only", type(exc).__name__, str(exc)[:120])
            return [], False

    @staticmethod
    def _parse_verdict(raw: str) -> Optional[ReviewVerdict]:
        for candidate in (raw, _extract_json_object(raw) or ""):
            if not candidate:
                continue
            try:
                return VERDICT_PARSER.parse(candidate.strip().strip("`").replace("json\n", "", 1))
            except Exception:  # noqa: BLE001
                continue
        return None


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _rubric_priority(due: date) -> str:
    days = (due - today()).days
    if days <= 3:
        return "high"
    if days <= 14:
        return "medium"
    return "low"


def _mentions(summary: str, title: str) -> bool:
    """Loose match: at least two significant words of the title appear."""
    words = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 3]
    hits = sum(1 for w in words if w in summary.lower())
    return hits >= min(2, len(words))

