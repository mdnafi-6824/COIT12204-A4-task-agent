"""Tests for the tools. I call them through LangChain's .invoke so it's the
same code path the agent uses."""

from app.tools import TOOLS, list_open_tasks, lookup_task, summarise_tasks


def test_tools_are_registered_with_schemas():
    names = {t.name for t in TOOLS}
    assert names == {"lookup_task", "list_open_tasks", "summarise_tasks"}
    schema = lookup_task.args_schema.model_json_schema()
    assert "task_id" in schema["properties"]
    assert lookup_task.description  # the docstring is what the model sees as the description


def test_lookup_task_returns_task_fields(store):
    result = lookup_task.invoke({"task_id": 2})
    assert result["title"] == "Renew campus WiFi certificate"
    assert result["priority"] == "high"
    assert result["days_until_due"] == 2


def test_lookup_task_missing_returns_error_not_exception(store):
    result = lookup_task.invoke({"task_id": 999})
    assert result == {"error": "Task 999 does not exist."}


def test_list_open_tasks_filters_window_and_done(store):
    rows = list_open_tasks.invoke({"days_ahead": 7})
    ids = [r["id"] for r in rows]
    # 5 is overdue so it's in, 1/2/3 are within 7 days, 4 is day 10 so it's out,
    # 6 has no date so it's out, 8 is done so it's out
    assert ids == [5, 1, 2, 3]
    assert rows[0]["days_until_due"] == -2


def test_list_open_tasks_clamps_days(store):
    rows = list_open_tasks.invoke({"days_ahead": 999})  # should get clamped to 30
    assert 7 in [r["id"] for r in rows]  # day 25 now included
    assert 4 in [r["id"] for r in rows]


def test_summarise_tasks_counts(store):
    result = summarise_tasks.invoke({})
    assert result["total"] == 8
    assert result["by_status"] == {"in_progress": 1, "todo": 6, "done": 1}
    assert result["overdue"] == 1  # only task 5, task 8 is overdue but it's done
    assert result["no_due_date"] == 1
    assert result["today"] == "2026-09-15"


def test_tool_calls_are_logged(store, caplog_agent):
    lookup_task.invoke({"task_id": 1})
    messages = [r.getMessage() for r in caplog_agent.records]
    assert any("TOOL CALL lookup_task(task_id=1)" in m for m in messages)
    assert any("TOOL RESULT lookup_task" in m for m in messages)
