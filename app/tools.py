"""The LangChain tools my agent can call.

Each one is a normal Python function with the @tool decorator on top.
LangChain reads the type hints and the docstring to build the schema the
model sees, so I wrote the docstrings for the model rather than for me.

The tools get to the task store through a module level variable that the
agent sets up (and my tests swap out). I return error dictionaries instead of
raising, because if a tool raises the whole run dies. With a dictionary the
model can read the error and explain it."""

from typing import Optional

from langchain_core.tools import tool

from app.logging_config import get_logger
from app.store import TaskStore, today

log = get_logger("tools")

_store: Optional[TaskStore] = None


def set_store(store: TaskStore) -> None:
    global _store
    _store = store


def get_store() -> TaskStore:
    if _store is None:
        raise RuntimeError("Task store has not been configured. Call set_store() first.")
    return _store


def _task_to_dict(task) -> dict:
    data = task.model_dump(mode="json")
    if task.due_date is not None:
        # I add this so the model doesn't have to do date maths itself
        data["days_until_due"] = (task.due_date - today()).days
    return data


@tool
def lookup_task(task_id: int) -> dict:
    """Look up a single task by its numeric id. Returns the task's title,
    description, priority, due_date, status and days_until_due. If the id does
    not exist the result contains an 'error' key instead."""
    log.info("TOOL CALL lookup_task(task_id=%s)", task_id)
    task = get_store().get(int(task_id))
    if task is None:
        log.warning("TOOL RESULT lookup_task: task %s not found", task_id)
        return {"error": f"Task {task_id} does not exist."}
    result = _task_to_dict(task)
    log.info("TOOL RESULT lookup_task: found '%s'", task.title)
    return result


@tool
def list_open_tasks(days_ahead: int = 7) -> list[dict]:
    """List tasks that are not done and are either overdue or due within the
    next days_ahead days (default 7). Tasks with no due date are not included.
    Returns a list ordered by due date, earliest first."""
    # clamp it so the model can't ask for 10000 days
    days_ahead = max(1, min(int(days_ahead), 30))
    log.info("TOOL CALL list_open_tasks(days_ahead=%s)", days_ahead)
    tasks = get_store().due_within(days_ahead)
    log.info("TOOL RESULT list_open_tasks: %d tasks", len(tasks))
    return [_task_to_dict(t) for t in tasks]


@tool
def summarise_tasks() -> dict:
    """Return overall counts for the task list: total, by status, by priority,
    how many are overdue and how many have no due date. Use this for a high
    level picture before writing a summary."""
    log.info("TOOL CALL summarise_tasks()")
    store = get_store()
    tasks = store.all()
    by_status: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    for t in tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
        key = t.priority or "unset"
        by_priority[key] = by_priority.get(key, 0) + 1
    result = {
        "total": len(tasks),
        "by_status": by_status,
        "by_priority": by_priority,
        "overdue": len(store.overdue()),
        "no_due_date": sum(1 for t in store.open_tasks() if t.due_date is None),
        "today": today().isoformat(),
    }
    log.info("TOOL RESULT summarise_tasks: %s", result)
    return result


TOOLS = [lookup_task, list_open_tasks, summarise_tasks]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
