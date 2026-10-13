"""The Worker agent.

This is my Assessment 3 TaskAgent wrapped so it speaks the message protocol.
It accepts two intents:

    execute_task    run an action (classify_priority, suggest_due_date, weekly_summary)
    revise_result   run the same action again with the Reviewer's feedback

Either way it replies to the Reviewer with a review_result message that
carries the structured result plus everything the Reviewer needs to check it
(the real task data, whether the fallback was used, the guardrail warnings)."""

from typing import Any, Optional

from app import config
from app.agent import TaskAgent
from app.agents.base import AgentError, BaseAgent
from app.messages import AgentMessage
from app.store import TaskStore


class WorkerAgent(BaseAgent):
    name = "worker"

    def __init__(self, agent: TaskAgent, store: TaskStore):
        super().__init__()
        self.agent = agent
        self.store = store

    def process(self, message: AgentMessage) -> AgentMessage:
        request = message.content.get("request") or {}
        feedback: Optional[str] = None
        if message.intent == "revise_result":
            feedback = message.content.get("feedback") or "The previous answer was rejected."
            # the reviewer can be wordy and the agent's input guardrail caps free
            # text, so I cut the feedback rather than let the run crash on it
            if len(feedback) > config.MAX_INPUT_CHARS:
                feedback = feedback[: config.MAX_INPUT_CHARS - 3] + "..."
        action = request.get("action")
        if action not in ("classify_priority", "suggest_due_date", "weekly_summary"):
            raise AgentError(f"worker cannot run action '{action}'")

        run = self.agent.run(
            action,
            task_id=request.get("task_id"),
            title=request.get("title"),
            description=request.get("description"),
            days_ahead=int(request.get("days_ahead") or 7),
            feedback=feedback,
        )
        task = self.store.get(request["task_id"]) if request.get("task_id") else None
        content: dict[str, Any] = {
            "request": request,
            "result": run.result.model_dump(mode="json"),
            "task": task.model_dump(mode="json") if task else None,
            "fallback_used": run.fallback_used,
            "warnings": run.warnings,
            "tools_called": run.tools_called,
            "llm_calls": run.llm_calls,
            "latency_ms": run.latency_ms,
            "revision": message.intent == "revise_result",
        }
        return message.reply(sender="worker", receiver="reviewer", intent="review_result", content=content, step=message.step + 1)

    def fallback(self, message: AgentMessage, reason: str) -> Optional[AgentMessage]:
        """If I time out or crash, the orchestrator still needs an answer. I
        reuse the TaskAgent's rule based fallback directly, no model call."""
        request = message.content.get("request") or {}
        task = self.store.get(request["task_id"]) if request.get("task_id") else None
        result = self.agent._fallback(
            request.get("action", "classify_priority"), task,
            request.get("title"), request.get("description"),
            int(request.get("days_ahead") or 7), reason,
        )
        content = {
            "request": request,
            "result": result.model_dump(mode="json"),
            "task": task.model_dump(mode="json") if task else None,
            "fallback_used": True,
            "warnings": [f"worker fallback: {reason}"],
            "tools_called": [],
            "llm_calls": 0,
            "latency_ms": 0,
            "revision": message.intent == "revise_result",
        }
        return message.reply(sender="worker", receiver="reviewer", intent="review_result", content=content, step=message.step + 1)
