"""A small task store backed by a JSON file.

This is my stand in for the database a real task manager would have. I kept
it simple on purpose because the assessment is about the agent, but the tools
still need real data to look up."""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from app.logging_config import get_logger
from app.models import Task, TaskCreate

log = get_logger("store")


def today() -> date:
    """I wrapped this so my tests can freeze the date."""
    return date.today()


class TaskStore:
    def __init__(self, path: Path, seed_if_missing: bool = True):
        self.path = Path(path)
        self._tasks: dict[int, Task] = {}
        if self.path.exists():
            self.load()
        elif seed_if_missing:
            # first run, so I write the sample tasks
            for task in seed_tasks():
                self._tasks[task.id] = task
            self.save()

    # -- load and save -----------------------------------------------------
    def load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._tasks = {int(item["id"]): Task(**item) for item in raw}
        log.info("Loaded %d tasks from %s", len(self._tasks), self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump(mode="json") for t in self._tasks.values()]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- reading -----------------------------------------------------------
    def get(self, task_id: int) -> Optional[Task]:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.id)

    def open_tasks(self) -> list[Task]:
        return [t for t in self.all() if t.status != "done"]

    def due_within(self, days: int, include_overdue: bool = True) -> list[Task]:
        # tasks with no due date can't be "due within" anything so I skip them
        limit = today() + timedelta(days=days)
        result = []
        for t in self.open_tasks():
            if t.due_date is None:
                continue
            if t.due_date <= limit and (include_overdue or t.due_date >= today()):
                result.append(t)
        return sorted(result, key=lambda t: t.due_date)  # type: ignore[arg-type]

    def overdue(self) -> list[Task]:
        return [t for t in self.open_tasks() if t.due_date and t.due_date < today()]

    # -- writing -----------------------------------------------------------
    def add(self, payload: TaskCreate) -> Task:
        next_id = max(self._tasks, default=0) + 1
        task = Task(id=next_id, created_at=today(), **payload.model_dump())
        self._tasks[next_id] = task
        self.save()
        log.info("Added task %d '%s'", task.id, task.title)
        return task

    def update(self, task_id: int, **fields) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated = task.model_copy(update=fields)
        self._tasks[task_id] = updated
        self.save()
        log.info("Updated task %d: %s", task_id, fields)
        return updated


def seed_tasks(base: Optional[date] = None) -> list[Task]:
    """Sample tasks. The due dates are relative to today so the weekly summary
    always has something in it no matter when I run the demo."""
    base = base or today()

    def d(n):
        return base + timedelta(days=n)

    rows = [
        (1, "Submit COIT12204 Assessment 3", "Final report, code and demo video via Moodle.", "high", d(1), "in_progress"),
        (2, "Renew campus WiFi certificate", "Cert expires soon, staff will lose access if missed.", "high", d(2), "todo"),
        (3, "Weekly team stand-up notes", "Write up action items from Monday stand-up.", "low", d(3), "todo"),
        (4, "Order new iPads for assessment lab", "Get three quotes and send purchase request to finance.", None, d(10), "todo"),
        (5, "Fix login bug on student portal", "Users report 500 error when password contains an ampersand.", None, d(-2), "todo"),
        (6, "Draft newsletter for October intake", "Marketing wants a draft by mid month, no firm date yet.", "medium", None, "todo"),
        (7, "Archive 2024 enrolment records", "Move old records to cold storage. Nobody is waiting on this.", "low", d(25), "todo"),
        (8, "Update README for attendance system", "Document the new export endpoint.", "low", d(-7), "done"),
    ]
    return [
        Task(id=i, title=t, description=desc, priority=p, due_date=due, status=s, created_at=d(-14))
        for i, t, desc, p, due, s in rows
    ]
