"""The orchestrator. It is the only thing that moves messages between the
Worker and the Reviewer.

One conversation looks like this:

    user -> worker        execute_task
    worker -> reviewer    review_result
    reviewer -> orch      result_approved          (done)
                          revision_requested       (loop back)
    orch -> worker        revise_result            (with the reviewer's feedback)
    ...
    orch -> user          final_result

What it enforces:

    validation      every message is an AgentMessage, built and checked before it moves
    routing         receiver has to be a registered agent, otherwise it is a routing error
    max revisions   the reviewer can send work back at most MAX_REVISIONS times
    max steps       a hard cap on messages per conversation, whatever happens
    timeouts        each agent call runs in a thread with a deadline
    fallback        a timed out or crashed agent is asked for its fallback answer,
                    and if the loop runs out the last result is returned marked as such
    logging         every message, every routing decision, every failure

Conversations are kept in memory (last 50) and saved to a JSON file so the
dashboard and the report can show them."""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from app import config
from app.agents.base import AgentError, BaseAgent
from app.logging_config import get_logger
from app.messages import AgentMessage, Constraints

log = get_logger("orchestrator")


@dataclass
class Conversation:
    id: str
    request: dict
    messages: list[dict] = field(default_factory=list)
    steps: int = 0
    revisions: int = 0
    status: str = "running"        # approved | fallback | max_revisions | max_steps | error
    final_result: Optional[dict] = None
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    elapsed_ms: int = 0

    def add(self, message: AgentMessage) -> None:
        self.messages.append(message.model_dump(mode="json"))
        self.steps = max(self.steps, message.step)

    def transcript(self) -> list[str]:
        return [f"{m['step']}: {m['sender']} -> {m['receiver']} [{m['intent']}]" for m in self.messages]


class Orchestrator:
    def __init__(
        self,
        agents: dict[str, BaseAgent],
        max_revisions: int = config.MAX_REVISIONS,
        max_steps: int = config.MAX_STEPS,
        agent_timeout: int = config.AGENT_TIMEOUT_SECONDS,
        persist_path: Optional[Path] = config.CONVERSATIONS_FILE,
        keep: int = 50,
    ):
        self.agents = agents
        self.max_revisions = max_revisions
        self.max_steps = max_steps
        self.agent_timeout = agent_timeout
        self.persist_path = persist_path
        self.keep = keep
        self.conversations: list[Conversation] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")
        log.info("Orchestrator ready agents=%s max_revisions=%d max_steps=%d timeout=%ds",
                 sorted(agents), max_revisions, max_steps, agent_timeout)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, request: dict) -> Conversation:
        with self._lock:
            return self._run(request)

    def _run(self, request: dict) -> Conversation:
        conv = Conversation(id=uuid.uuid4().hex[:8], request=request, started_at=_now())
        started = time.perf_counter()
        log.info("CONVERSATION START %s request=%s", conv.id, json.dumps(request, default=str))
        constraints = Constraints(max_revisions=self.max_revisions, timeout_seconds=self.agent_timeout)

        message = self._build(conv, sender="user", receiver="worker", intent="execute_task",
                              content={"request": request}, step=0, constraints=constraints)
        last_result: Optional[dict] = None

        while message is not None:
            if conv.steps >= self.max_steps:
                conv.errors.append(f"max steps ({self.max_steps}) reached")
                log.error("CONVERSATION %s hit max steps %d", conv.id, self.max_steps)
                conv.status = "max_steps"
                break

            reply = self._route(conv, message)
            if reply is None:            # routing failed and no fallback possible
                conv.status = "error"
                break
            conv.add(reply)

            if reply.intent == "review_result":
                last_result = reply.content.get("result")
                message = reply          # goes to the reviewer next
                continue

            if reply.intent == "result_approved":
                conv.final_result = reply.content.get("result") or last_result
                conv.status = "approved"
                if reply.content.get("warning"):
                    conv.errors.append(reply.content["warning"])
                break

            if reply.intent == "revision_requested":
                if conv.revisions >= self.max_revisions:
                    conv.errors.append(f"max revisions ({self.max_revisions}) reached, returning last result")
                    log.warning("CONVERSATION %s max revisions reached, falling back to last result", conv.id)
                    conv.final_result = last_result
                    conv.status = "max_revisions"
                    break
                conv.revisions += 1
                log.info("CONVERSATION %s revision %d/%d: %s", conv.id, conv.revisions, self.max_revisions,
                         reply.content.get("feedback", "")[:120])
                message = self._build(
                    conv, sender="orchestrator", receiver="worker", intent="revise_result",
                    content={"request": request, "previous_result": last_result, "feedback": reply.content.get("feedback")},
                    step=reply.step + 1, constraints=constraints,
                )
                continue

            conv.errors.append(f"unexpected intent '{reply.intent}' from {reply.sender}")
            log.error("CONVERSATION %s unexpected intent %s", conv.id, reply.intent)
            conv.status = "error"
            break

        if conv.final_result is None and last_result is not None:
            # something went wrong after the worker answered, so the user still
            # gets that answer, marked as a fallback rather than approved
            conv.final_result = last_result
            if conv.status in ("running", "error"):
                conv.status = "fallback"
        elif conv.final_result is None and conv.status == "running":
            conv.status = "error"

        # the last message always goes to the user, so the log shows the hand back
        self._build(
            conv, sender="orchestrator", receiver="user", intent="final_result",
            content={"status": conv.status, "result": conv.final_result, "revisions": conv.revisions, "errors": conv.errors},
            step=conv.steps + 1, constraints=constraints,
        )

        conv.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("CONVERSATION END %s status=%s steps=%d revisions=%d elapsed=%dms",
                 conv.id, conv.status, conv.steps, conv.revisions, conv.elapsed_ms)
        self._remember(conv)
        return conv

    # ------------------------------------------------------------------
    # Building and routing messages
    # ------------------------------------------------------------------
    def _build(self, conv: Conversation, **fields) -> Optional[AgentMessage]:
        """Validate a message before it exists. A bad message is logged and
        counted as an error instead of raising into the API."""
        try:
            message = AgentMessage(conversation_id=conv.id, **fields)
        except ValidationError as exc:
            reason = exc.errors()[0].get("msg", str(exc))
            conv.errors.append(f"invalid message: {reason}")
            log.error("MSG INVALID %s %s", conv.id, reason)
            return None
        conv.add(message)
        return message

    def _route(self, conv: Conversation, message: AgentMessage) -> Optional[AgentMessage]:
        agent = self.agents.get(message.receiver)
        if agent is None:
            conv.errors.append(f"no agent registered for receiver '{message.receiver}'")
            log.error("ROUTE ERROR %s no agent for receiver '%s'", conv.id, message.receiver)
            return None
        log.info("ROUTE %s -> %s", message.short(), agent.name)

        future = self._pool.submit(agent.handle, message)
        try:
            return future.result(timeout=message.constraints.timeout_seconds)
        except FutureTimeout:
            reason = f"{agent.name} timed out after {message.constraints.timeout_seconds}s"
        except AgentError as exc:
            reason = f"{agent.name} error: {exc}"
        except Exception as exc:  # noqa: BLE001  anything else is still an agent failure
            reason = f"{agent.name} crashed: {type(exc).__name__}: {str(exc)[:120]}"

        conv.errors.append(reason)
        log.error("AGENT ERROR %s %s", conv.id, reason)
        self._build(conv, sender=message.receiver, receiver="orchestrator", intent="agent_error",
                    content={"reason": reason, "in_reply_to": message.message_id}, step=message.step + 1,
                    constraints=message.constraints)
        fallback = agent.fallback(message, reason)
        if fallback is None:
            log.error("AGENT ERROR %s %s has no fallback, conversation stops", conv.id, agent.name)
            return None
        log.warning("FALLBACK %s %s answered with its fallback", conv.id, agent.name)
        # the fallback was built by the agent, so I validate it the same way
        return AgentMessage.model_validate(fallback.model_dump())

    # ------------------------------------------------------------------
    # Memory of past conversations
    # ------------------------------------------------------------------
    def _remember(self, conv: Conversation) -> None:
        self.conversations.append(conv)
        self.conversations = self.conversations[-self.keep :]
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps([asdict(c) for c in self.conversations], indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not save conversations to %s: %s", self.persist_path, exc)

    def stats(self) -> dict:
        convs = self.conversations
        n = len(convs)
        return {
            "conversations": n,
            "approved": sum(1 for c in convs if c.status == "approved"),
            "with_revisions": sum(1 for c in convs if c.revisions > 0),
            "fallbacks": sum(1 for c in convs if c.status in ("fallback", "max_revisions", "max_steps", "error")),
            "mean_steps": round(sum(c.steps for c in convs) / n, 2) if n else 0,
            "mean_elapsed_ms": int(sum(c.elapsed_ms for c in convs) / n) if n else 0,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def conversation_to_dict(conv: Conversation) -> dict[str, Any]:
    return asdict(conv)
