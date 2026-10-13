"""Tests for AgentState: the counters, the log lines and the save/load."""

import json

from app.state import AgentState


def test_initial_state_is_empty(state):
    assert state.snapshot() == {
        "run_count": 0,
        "error_count": 0,
        "fallback_count": 0,
        "last_action": None,
        "last_suggestion": None,
        "last_tool_call": None,
        "history_length": 0,
        "updated_at": None,
    }


def test_record_run_updates_tracked_variables(state):
    result = {"action": "classify_priority", "priority": "high"}
    state.record_run("classify_priority", result, tools_called=["lookup_task"])
    snap = state.snapshot()
    assert snap["run_count"] == 1
    assert snap["last_action"] == "classify_priority"
    assert snap["last_suggestion"] == result
    assert snap["error_count"] == 0
    assert snap["updated_at"] is not None
    assert state.history[-1]["type"] == "run"
    assert state.history[-1]["tools_called"] == ["lookup_task"]


def test_error_and_fallback_counters(state):
    state.record_run("suggest_due_date", {"x": 1}, fallback_used=True, error="TimeoutError: slow")
    state.record_run("suggest_due_date", {"x": 2}, fallback_used=True, error="ValueError: bad json")
    state.record_run("suggest_due_date", {"x": 3})
    assert state.run_count == 3
    assert state.error_count == 2
    assert state.fallback_count == 2
    assert state.last_error == "ValueError: bad json"
    assert state.last_suggestion == {"x": 3}


def test_record_tool_call_sets_last_tool_call(state):
    state.record_tool_call("lookup_task", {"task_id": 4}, ok=True, result_preview="{...}")
    assert state.last_tool_call["name"] == "lookup_task"
    assert state.last_tool_call["args"] == {"task_id": 4}
    assert state.last_tool_call["ok"] is True
    assert state.history[-1]["type"] == "tool_call"


def test_history_is_bounded(state):
    for i in range(12):
        state.record_run("weekly_summary", {"i": i})
    assert len(state.history) == 5  # the fixture sets history_limit to 5
    assert state.history[-1]["result"] == {"i": 11}
    assert state.run_count == 12  # the counter still counts even when history is capped


def test_transitions_are_logged(state, caplog_agent):
    state.record_run("classify_priority", {"priority": "low"})
    msgs = [r.getMessage() for r in caplog_agent.records]
    assert any("STATE TRANSITION run_count: 0 -> 1" in m for m in msgs)
    assert any("STATE TRANSITION last_action: None -> classify_priority" in m for m in msgs)


def test_unchanged_value_does_not_log(state, caplog_agent):
    state.record_run("weekly_summary", {"a": 1})
    caplog_agent.clear()
    state.record_run("weekly_summary", {"a": 1})
    msgs = [r.getMessage() for r in caplog_agent.records]
    assert not any("last_action" in m for m in msgs)  # same action again so no transition line
    assert any("run_count: 1 -> 2" in m for m in msgs)


def test_state_persists_and_reloads(tmp_path):
    path = tmp_path / "state.json"
    s = AgentState(persist_path=path)
    s.record_tool_call("summarise_tasks", {}, ok=True)
    s.record_run("weekly_summary", {"summary": "ok"})
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["run_count"] == 1

    reloaded = AgentState.load(path)
    assert reloaded.run_count == 1
    assert reloaded.last_tool_call["name"] == "summarise_tasks"
    assert len(reloaded.history) == 2


def test_load_handles_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    s = AgentState.load(path)
    assert s.run_count == 0  # it started fresh instead of crashing


def test_reset_clears_everything(state):
    state.record_run("classify_priority", {"priority": "high"}, fallback_used=True, error="x")
    state.reset()
    snap = state.snapshot()
    assert snap["run_count"] == 0 and snap["error_count"] == 0 and snap["fallback_count"] == 0
    assert snap["last_suggestion"] is None and snap["history_length"] == 0
