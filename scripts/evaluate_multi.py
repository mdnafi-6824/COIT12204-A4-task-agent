"""Runs my evaluation cases through the two agent system with the REAL model.

    python -m scripts.evaluate_multi                 # 8 cases, 2 repeats, reviewer uses rules + model
    python -m scripts.evaluate_multi --repeats 3
    python -m scripts.evaluate_multi --reviewer rules   # reviewer uses rules only
    python -m scripts.evaluate_multi --break-worker     # simulate the worker's model being down

For every conversation I record the status, how many revisions it took,
how many steps, what the reviewer rejected and why, whether the final answer
was correct, and the time. The full transcripts go to
evaluation/multi/transcripts_<stamp>.json so the report can quote real
agent to agent exchanges, and a summary table is added to
evaluation/multi/summary.md."""

import argparse
import json
import logging
import statistics
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from app import config
from app.agent import TaskAgent
from app.agents import ReviewerAgent, WorkerAgent
from app.logging_config import setup_logging
from app.orchestrator import Orchestrator, conversation_to_dict
from app.state import AgentState
from app.store import TaskStore, seed_tasks, today

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "evaluation" / "multi"


def _date_ok(result, max_days):
    d = result.get("suggested_due_date")
    if not d:
        return False
    d = date.fromisoformat(d)
    return today() <= d <= today() + timedelta(days=max_days)


CASES = [
    {"id": "M1_explicit_high", "request": {"action": "classify_priority", "task_id": 2}, "check": lambda r: r.get("priority") == "high"},
    {"id": "M2_housekeeping_low", "request": {"action": "classify_priority", "task_id": 7}, "check": lambda r: r.get("priority") == "low"},
    {"id": "M3_unset_medium", "request": {"action": "classify_priority", "task_id": 4}, "check": lambda r: r.get("priority") == "medium"},
    {"id": "M4_overdue_bug_high", "request": {"action": "classify_priority", "task_id": 5}, "check": lambda r: r.get("priority") == "high"},
    {"id": "M5_due_overdue", "request": {"action": "suggest_due_date", "task_id": 5}, "check": lambda r: _date_ok(r, 3)},
    {"id": "M6_due_none", "request": {"action": "suggest_due_date", "task_id": 6}, "check": lambda r: _date_ok(r, 60)},
    {"id": "M7_weekly_summary", "request": {"action": "weekly_summary", "days_ahead": 7}, "check": lambda r: "overdue" in (r.get("summary") or "").lower()},
    {"id": "M8_text_urgent", "request": {"action": "classify_priority", "title": "Server room aircon failed", "description": "Temperature rising, servers will overheat within hours"}, "check": lambda r: r.get("priority") == "high"},
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reviewer", choices=["rules", "both"], default="both")
    parser.add_argument("--break-worker", action="store_true", help="Point the worker at a bad key so every answer is a fallback")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not config.API_KEY:
        sys.exit(f"No API key for provider '{config.LLM_PROVIDER}'. Put it in .env first.")

    logger = setup_logging()
    if not args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.TASKS_FILE.exists():
        config.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.TASKS_FILE.write_text(json.dumps([t.model_dump(mode="json") for t in seed_tasks()], indent=2))
    store = TaskStore(config.TASKS_FILE)

    state = AgentState(persist_path=Path(tempfile.mkdtemp()) / "state.json")
    real_keys = (config.ANTHROPIC_API_KEY, config.OPENAI_API_KEY)
    if args.break_worker:
        # the worker's model is built with a bad key, so every first answer is a rule fallback
        config.ANTHROPIC_API_KEY = config.OPENAI_API_KEY = "sk-broken"
    worker = WorkerAgent(TaskAgent(store, state), store)
    config.ANTHROPIC_API_KEY, config.OPENAI_API_KEY = real_keys   # the reviewer still gets the real key
    reviewer = ReviewerAgent(store, use_llm=(args.reviewer == "both"))
    orch = Orchestrator({"worker": worker, "reviewer": reviewer}, persist_path=Path(tempfile.mkdtemp()) / "c.json", keep=500)

    label = f"reviewer={args.reviewer}" + (" worker=BROKEN" if args.break_worker else "")
    print(f"\n=== Two agent evaluation ({label}, model {config.MODEL_NAME}) ===")
    rows, transcripts = [], []
    for case in CASES:
        for rep in range(1, args.repeats + 1):
            conv = orch.run(dict(case["request"]))
            result = conv.final_result or {}
            rejections = [m["content"].get("feedback", "") for m in conv.messages if m["intent"] == "revision_requested"]
            row = {
                "case": case["id"], "repeat": rep, "status": conv.status, "steps": conv.steps,
                "revisions": conv.revisions, "correct": bool(case["check"](result)) and conv.status == "approved",
                "rejections": rejections, "errors": conv.errors, "elapsed_ms": conv.elapsed_ms,
                "answer": result.get("priority") or result.get("suggested_due_date") or (result.get("summary") or "")[:60],
            }
            rows.append(row)
            transcripts.append(conversation_to_dict(conv))
            flag = "OK " if row["correct"] else "BAD"
            print(f"  [{flag}] {case['id']:<20} rep{rep} status={conv.status:<13} revisions={conv.revisions} steps={conv.steps} "
                  f"{conv.elapsed_ms:>6}ms -> {row['answer']!s:.50}")
            for r in rejections:
                print(f"        rejected: {r[:150]}")

    n = len(rows)
    summary = {
        "label": label, "model": config.MODEL_NAME, "runs": n,
        "approved": sum(1 for r in rows if r["status"] == "approved") / n,
        "accuracy": sum(1 for r in rows if r["correct"]) / n,
        "with_revisions": sum(1 for r in rows if r["revisions"] > 0),
        "total_rejections": sum(len(r["rejections"]) for r in rows),
        "fallbacks": sum(1 for r in rows if r["status"] != "approved"),
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 1),
        "mean_elapsed_ms": int(statistics.mean(r["elapsed_ms"] for r in rows)),
    }
    print(f"  approved={summary['approved']:.0%} accuracy={summary['accuracy']:.0%} with_revisions={summary['with_revisions']}/{n} "
          f"rejections={summary['total_rejections']} fallbacks={summary['fallbacks']} mean_steps={summary['mean_steps']} "
          f"latency={summary['mean_elapsed_ms']}ms")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"results_{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    (OUT_DIR / f"transcripts_{stamp}.json").write_text(json.dumps(transcripts, indent=2, default=str))
    md = [f"\n## Two agent run {stamp} ({label}, model {config.MODEL_NAME}, {args.repeats} repeats, {len(CASES)} cases)\n",
          "| Approved | Accuracy | Runs with revisions | Rejections | Fallbacks | Mean steps | Mean latency |",
          "|---|---|---|---|---|---|---|",
          f"| {summary['approved']:.0%} | {summary['accuracy']:.0%} | {summary['with_revisions']}/{n} | {summary['total_rejections']} | "
          f"{summary['fallbacks']} | {summary['mean_steps']} | {summary['mean_elapsed_ms']} ms |"]
    with (OUT_DIR / "summary.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"  saved {OUT_DIR.relative_to(ROOT)}/results_{stamp}.json and transcripts_{stamp}.json")


if __name__ == "__main__":
    main()
