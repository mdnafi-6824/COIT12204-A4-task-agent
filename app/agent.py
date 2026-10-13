"""My single LangChain agent.

This is what happens on one run:

    1. I build the request as a HumanMessage (task details or a summary ask).
    2. I run the LCEL chain  prompt | llm.bind_tools(TOOLS).
    3. If the model asked for tools, I run them, add the results as
       ToolMessages and go back to step 2. This is capped by MAX_TOOL_ROUNDS.
    4. I parse the final text with PydanticOutputParser into an AgentResult.
       If that fails I ask the model one more time for JSON only.
    5. I run the output guardrails, update AgentState and return an AgentRun.

If anything goes wrong (the model raises, the output can't be parsed, the
tool loop runs out of rounds) the run ends in my rule based fallback rather
than an exception. That way the API always has a structured answer to send
back and the state always records what happened."""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app import config
from app.guardrails import check_result, clean_text
from app.logging_config import get_logger
from app.models import AgentResult, Task
from app.prompts import HUMAN_TEMPLATES, PARSER, build_prompt
from app.state import AgentState
from app.store import TaskStore, today
from app.tools import TOOLS, TOOLS_BY_NAME, set_store

log = get_logger("agent")

# strips ```json ... ``` fences that the model sometimes wraps around the answer
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def get_llm() -> BaseChatModel:
    """Builds the real model. My tests monkeypatch this function so no test
    ever talks to a real provider."""
    if config.LLM_PROVIDER == "anthropic":
        # imported here so the OpenAI only setup doesn't need this package
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            temperature=0,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=1,
            api_key=config.ANTHROPIC_API_KEY or None,
        )
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=1,
        api_key=config.OPENAI_API_KEY or None,
    )


@dataclass
class AgentRun:
    """Everything I want to know about one run, not just the answer."""

    action: str
    result: AgentResult
    fallback_used: bool = False
    tools_called: list[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    latency_ms: int = 0
    prompt_version: str = config.PROMPT_VERSION


class TaskAgent:
    def __init__(
        self,
        store: TaskStore,
        state: AgentState,
        llm: Optional[BaseChatModel] = None,
        prompt_version: Optional[str] = None,
    ):
        self.store = store
        self.state = state
        set_store(store)
        self.prompt_version = prompt_version or config.PROMPT_VERSION
        self.prompt = build_prompt(self.prompt_version)
        self.llm = llm or get_llm()
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        # the LCEL chain: template, then the model with the tools attached
        self.chain = self.prompt | self.llm_with_tools
        # FastAPI runs sync endpoints in a thread pool, so two requests could
        # update the state at the same time. One lock keeps runs in order.
        self._lock = threading.Lock()
        log.info("Agent ready (prompt=%s, model=%s)", self.prompt_version, config.MODEL_NAME)

    # ------------------------------------------------------------------
    # The wrapper. This is the only thing the API calls.
    # ------------------------------------------------------------------
    def run(
        self,
        action: str,
        task_id: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        days_ahead: int = 7,
        feedback: Optional[str] = None,
    ) -> AgentRun:
        """feedback is used by the two agent system: when the Reviewer rejects
        an answer, the Orchestrator calls run() again with the reviewer's notes
        so the Worker can fix the specific problem instead of starting over."""
        if action not in HUMAN_TEMPLATES:
            raise ValueError(f"Unknown action '{action}'")
        with self._lock:
            return self._run(action, task_id, title, description, days_ahead, feedback)

    def _run(self, action, task_id, title, description, days_ahead, feedback=None) -> AgentRun:
        started = time.perf_counter()
        log.info("RUN START action=%s task_id=%s revision=%s", action, task_id, bool(feedback))

        task = self.store.get(task_id) if task_id is not None else None
        request = self._build_request(action, task, task_id, title, description, days_ahead, feedback)

        tools_called: list[str] = []
        run = AgentRun(action=action, result=None, prompt_version=self.prompt_version)  # type: ignore[arg-type]
        try:
            raw, run.llm_calls = self._execute(request, tools_called)
            run.raw_output = raw
            result = self._parse(raw)
            if result is None:
                # one more go, then I give up and use the fallback
                log.warning("PARSE FAILED, asking the model for JSON only")
                raw, extra = self._retry_json(request, raw, tools_called)
                run.llm_calls += extra
                run.raw_output = raw
                result = self._parse(raw)
            if result is None:
                raise ValueError("Model output could not be parsed as AgentResult")
            result, run.warnings = check_result(result, action)
            run.result = result
        except Exception as exc:  # noqa: BLE001  I want every failure to end in the fallback
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            log.error("RUN ERROR action=%s -> %s", action, reason)
            run.error = reason
            run.fallback_used = True
            run.result = self._fallback(action, task, title, description, days_ahead, reason)

        run.tools_called = tools_called
        run.latency_ms = int((time.perf_counter() - started) * 1000)
        self.state.record_run(
            action=action,
            result=run.result.model_dump(mode="json"),
            fallback_used=run.fallback_used,
            error=run.error,
            tools_called=tools_called,
        )
        log.info(
            "RUN END action=%s fallback=%s tools=%s llm_calls=%d latency=%dms",
            action, run.fallback_used, tools_called, run.llm_calls, run.latency_ms,
        )
        return run

    # ------------------------------------------------------------------
    # The bits behind run()
    # ------------------------------------------------------------------
    def _build_request(self, action, task, task_id, title, description, days_ahead, feedback=None) -> HumanMessage:
        if action == "weekly_summary":
            text = HUMAN_TEMPLATES[action].format(days_ahead=days_ahead)
        else:
            if task is not None:
                # I only give the id on purpose, so the model has to use lookup_task
                task_block = f"Task id: {task.id}. Use lookup_task to see its details."
            elif task_id is not None:
                task_block = f"Task id: {task_id}."
            else:
                clean_title = clean_text(title, "title")
                clean_desc = clean_text(description, "description")
                task_block = (
                    f"This task is not saved yet, so there is nothing to look up.\n"
                    f"Title: {clean_title}\nDescription: {clean_desc or '(none)'}"
                )
            text = HUMAN_TEMPLATES[action].format(task_block=task_block)
        if feedback:
            text += (
                "\n\nA reviewer rejected your previous answer for this reason: "
                f"{clean_text(feedback, 'feedback')}\n"
                "Fix that specific problem and give a corrected answer."
            )
        return HumanMessage(content=text)

    def _invoke(self, messages: list[BaseMessage]) -> AIMessage:
        response = self.chain.invoke({"messages": messages, "today": today().isoformat()})
        if not isinstance(response, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(response).__name__}")
        return response

    def _execute(self, request: HumanMessage, tools_called: list[str]) -> tuple[str, int]:
        """The tool loop. Returns the final text and how many model calls it took."""
        messages: list[BaseMessage] = [request]
        calls = 0
        for round_no in range(1, config.MAX_TOOL_ROUNDS + 1):
            ai = self._invoke(messages)
            calls += 1
            messages.append(ai)
            if not ai.tool_calls:
                return _text(ai), calls
            log.info("ROUND %d model requested %d tool call(s)", round_no, len(ai.tool_calls))
            for call in ai.tool_calls:
                messages.append(self._run_tool(call, tools_called))
        # Ran out of rounds. I keep the tools bound (Anthropic rejects a history
        # with tool results if the tools aren't declared) and just tell the
        # model it has to answer now.
        log.warning("Tool round limit (%d) reached, forcing a final answer", config.MAX_TOOL_ROUNDS)
        messages.append(HumanMessage(content="No more tool calls are allowed. Reply now with ONLY the JSON object."))
        final = self._invoke(messages)
        return _text(final), calls + 1

    def _run_tool(self, call: dict, tools_called: list[str]) -> ToolMessage:
        name = call["name"]
        args = call.get("args") or {}
        call_id = call.get("id") or name  # some providers leave the id out
        tool = TOOLS_BY_NAME.get(name)
        tools_called.append(name)
        if tool is None:
            log.warning("Model asked for unknown tool '%s'", name)
            self.state.record_tool_call(name, args, ok=False, result_preview="unknown tool")
            return ToolMessage(content=json.dumps({"error": f"Unknown tool {name}"}), tool_call_id=call_id)
        try:
            output = tool.invoke(args)
            ok = not (isinstance(output, dict) and "error" in output)
        except Exception as exc:  # noqa: BLE001
            output, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
            log.error("TOOL ERROR %s(%s): %s", name, args, exc)
        content = json.dumps(output, default=str)
        self.state.record_tool_call(name, args, ok=ok, result_preview=content)
        return ToolMessage(content=content, tool_call_id=call_id)

    def _retry_json(self, request: HumanMessage, bad_output: str, tools_called: list[str]) -> tuple[str, int]:
        messages: list[BaseMessage] = [
            request,
            AIMessage(content=bad_output or ""),
            HumanMessage(
                content="That reply was not valid JSON for the required schema. "
                "Reply again with ONLY the JSON object, no explanation, no code fences."
            ),
        ]
        ai = self._invoke(messages)
        if ai.tool_calls:  # it still wants tools, so I allow one round and then stop
            messages.append(ai)
            for call in ai.tool_calls:
                messages.append(self._run_tool(call, tools_called))
            ai = self._invoke(messages)
            return _text(ai), 2
        return _text(ai), 1

    @staticmethod
    def _parse(raw: str) -> Optional[AgentResult]:
        if not raw or not raw.strip():
            return None
        cleaned = _FENCE_RE.sub("", raw.strip()).strip()
        try:
            return PARSER.parse(cleaned)
        except Exception as exc:  # OutputParserException or a pydantic ValidationError
            first_error = str(exc).splitlines()[0][:160]
        # Improvement after evaluation: the model often talks first and then
        # gives the JSON, so I dig the JSON object out of the prose before
        # giving up and spending another model call on a retry.
        embedded = _extract_json_object(cleaned)
        if embedded:
            try:
                result = PARSER.parse(embedded)
                log.info("PARSE RECOVERED JSON object from inside a longer reply")
                return result
            except Exception as exc:  # noqa: BLE001
                first_error = str(exc).splitlines()[0][:160]
        log.warning("PARSE ERROR: %s | raw=%r", first_error, raw[:160])
        return None

    # ------------------------------------------------------------------
    # My fallback. No model, just rules, and it always returns something.
    # ------------------------------------------------------------------
    def _fallback(self, action, task: Optional[Task], title, description, days_ahead, reason) -> AgentResult:
        log.warning("FALLBACK used for %s (%s)", action, reason)
        note = f"Fallback rule used because the model step failed ({reason})."
        if action == "classify_priority":
            return AgentResult(
                action=action,
                task_id=task.id if task else None,
                priority=_rule_priority(task, title, description),
                reasoning=note,
                confidence=0.3,
            )
        if action == "suggest_due_date":
            return AgentResult(
                action=action,
                task_id=task.id if task else None,
                suggested_due_date=_rule_due_date(task),
                reasoning=note,
                confidence=0.3,
            )
        due = self.store.due_within(days_ahead)
        overdue = [t for t in due if t.due_date and t.due_date < today()]
        titles = ", ".join(t.title for t in due[:5]) or "nothing"
        summary = (
            f"{len(due)} open task(s) due in the next {days_ahead} days, {len(overdue)} overdue. "
            f"Most urgent: {titles}."
        )
        return AgentResult(action=action, summary=summary, reasoning=note, confidence=0.5)


def _extract_json_object(text: str) -> Optional[str]:
    """Finds the first balanced {...} block in some text. I walk the string and
    count braces, skipping anything inside a JSON string, so braces in the
    reasoning text don't confuse it."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):  # some providers send content as a list of blocks
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


_URGENT_WORDS = ("urgent", "asap", "blocking", "expires", "outage", "bug", "error", "deadline", "failed", "failure", "overheat", "down")


def _rule_priority(task: Optional[Task], title, description) -> str:
    # simple rules: keep an existing priority, otherwise go by the due date,
    # otherwise look for urgent sounding words
    if task is not None:
        if task.priority:
            return task.priority
        if task.due_date is not None:
            days = (task.due_date - today()).days
            return "high" if days <= 3 else "medium" if days <= 14 else "low"
        text = f"{task.title} {task.description}"
    else:
        text = f"{title or ''} {description or ''}"
    text = text.lower()
    if any(word in text for word in _URGENT_WORDS):
        return "high"
    return "medium" if len(text.split()) > 6 else "low"


def _rule_due_date(task: Optional[Task]):
    # keep a future date if there is one, otherwise pick an offset by priority
    if task is not None and task.due_date is not None and task.due_date >= today():
        return task.due_date
    # Improvement after evaluation: an overdue task fell back to today + 7,
    # which is a bad answer for something that's already late. Now it's tomorrow.
    if task is not None and task.due_date is not None and task.due_date < today():
        return today() + timedelta(days=1)
    priority = (task.priority if task else None) or "medium"
    offset = {"high": 1, "medium": 7, "low": 14}[priority]
    return today() + timedelta(days=offset)
