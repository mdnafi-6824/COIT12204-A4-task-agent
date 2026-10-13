"""Rewrites data/tasks.json with the sample tasks, with due dates relative to
today. I run this before a demo so the weekly summary has current data:

    python -m scripts.seed_tasks
"""

import json

from app import config
from app.store import seed_tasks, today

if __name__ == "__main__":
    config.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TASKS_FILE.write_text(json.dumps([t.model_dump(mode="json") for t in seed_tasks()], indent=2))
    print(f"Wrote {len(seed_tasks())} tasks to {config.TASKS_FILE} relative to {today()}")
