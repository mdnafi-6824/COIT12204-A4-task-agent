"""The agent's internal state.

AgentState keeps track of what the agent has done so far. The brief asked for
at least three tracked variables. I track these:

    run_count       how many times the agent has run
    error_count     how many runs hit an error (model failure, bad output...)
    fallback_count  how many runs ended with my rule based fallback
    last_action     the last action that was requested
    last_suggestion the last structured result the agent gave
    last_tool_call  name, args and success flag of the last tool call
    history         a capped list of every run and tool call, newest last

Every change goes through _transition(), which logs the old and new value.
So the log file shows the state moving step by step. The state also saves
itself to a JSON file and loads it back, which is the "persistent state"
bonus item and means it survives a restart of the API."""

import json
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import config
from app.logging_config import get_logger

log = get_logger("state")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentState:
    run_count: int = 0
    error_count: int = 0
    fallback_count: int = 0
    last_action: Optional[str] = None
    last_suggestion: Optional[dict] = None
    last_tool_call: Optional[dict] = None
    last_error: Optional[str] = None
    history: list[dict] = field(default_factory=list)
    updated_at: Optional[str] = None
    history_limit: int = config.HISTORY_LIMIT
    persist_path: Optional[Path] = None

    # -- the one place values change ---------------------------------------
    def _transition(self, name: str, new_value: Any) -> None:
        old_value = getattr(self, name)
        if old_value == new_value:
            return  # nothing changed, so I don't spam the log
        setattr(self, name, new_value)
        self.updated_at = _now()
        log.info("STATE TRANSITION %s: %s -> %s", name, _short(old_value), _short(new_value))

    def _push_history(self, entry: dict) -> None:
        entry = {"at": _now(), **entry}
        history = self.history + [entry]
        if len(history) > self.history_limit:
            history = history[-self.history_limit :]
        # I don't go through _transition here, the log line would be huge
        self.history = history
        self.updated_at = entry["at"]

    # -- things the agent tells me about -----------------------------------
    def record_tool_call(self, name: str, args: dict, ok: bool, result_preview: str = "") -> None:
        self._transition(
            "last_tool_call",
            {"name": name, "args": args, "ok": ok, "at": _now()},
        )
        self._push_history({"type": "tool_call", "name": name, "args": args, "ok": ok, "result": result_preview[:200]})

    def record_run(
        self,
        action: str,
        result: Optional[dict],
        fallback_used: bool = False,
        error: Optional[str] = None,
        tools_called: Optional[list[str]] = None,
    ) -> None:
        self._transition("run_count", self.run_count + 1)
        self._transition("last_action", action)
        if error:
            self._transition("error_count", self.error_count + 1)
            self._transition("last_error", error)
        if fallback_used:
            self._transition("fallback_count", self.fallback_count + 1)
        if result is not None:
            self._transition("last_suggestion", result)
        self._push_history(
            {
                "type": "run",
                "action": action,
                "fallback_used": fallback_used,
                "error": error,
                "tools_called": tools_called or [],
                "result": result,
            }
        )
        self.save()

    def reset(self) -> None:
        log.info("STATE RESET requested")
        for name, default in (
            ("run_count", 0),
            ("error_count", 0),
            ("fallback_count", 0),
            ("last_action", None),
            ("last_suggestion", None),
            ("last_tool_call", None),
            ("last_error", None),
        ):
            self._transition(name, default)
        self.history = []
        self.save()

    # -- read only views ---------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "run_count": self.run_count,
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "last_action": self.last_action,
            "last_suggestion": self.last_suggestion,
            "last_tool_call": self.last_tool_call,
            "history_length": len(self.history),
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("persist_path", None)
        data.pop("history_limit", None)
        return data

    # -- save and load -----------------------------------------------------
    def save(self) -> None:
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not persist state to %s: %s", self.persist_path, exc)

    @classmethod
    def load(cls, path: Path, history_limit: int = config.HISTORY_LIMIT) -> "AgentState":
        path = Path(path)
        state = cls(persist_path=path, history_limit=history_limit)
        if not path.exists():
            log.info("No saved state at %s, starting fresh", path)
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(cls)} - {"persist_path", "history_limit"}
            for key, value in data.items():
                if key in allowed:
                    setattr(state, key, value)
            log.info("Loaded state from %s (run_count=%s)", path, state.run_count)
        except (OSError, ValueError) as exc:
            # a broken file shouldn't stop the app, I just start again
            log.warning("Saved state at %s unreadable (%s), starting fresh", path, exc)
        return state


def _short(value: Any, limit: int = 120) -> str:
    # keeps the log lines readable when the value is a big dict
    text = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
