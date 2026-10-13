"""Runs my evaluation cases against the REAL model.

Run it from the project root with your API key in .env:

    python -m scripts.evaluate                       # prompt v3, 2 repeats
    python -m scripts.evaluate --versions v1 v2 v3   # compare prompt versions
    python -m scripts.evaluate --repeats 3

For each prompt version I run every case N times and record:
    parsed      the model produced valid structured output (no fallback)
    correct     the answer matched the expectation for that case
    tools_ok    the expected tool was called
    warnings    output guardrail corrections that were needed
    consistent  all repeats gave the same answer
    latency_ms  wall clock time per run

Results go to evaluation/results_<version>_<timestamp>.json and a summary
table gets added to evaluation/summary.md. Those files are my evidence for
the evaluation section of the report."""

import argparse
import json
import logging
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app import config
from app.agent import TaskAgent
from app.logging_config import setup_logging
from app.state import AgentState
from app.store import TaskStore, seed_tasks, today

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "evaluation"


# ---------------------------------------------------------------------------
# My cases. Each one has the call to make, the tool I expect the model to use,
# and a check() that says if the answer was acceptable.
# ---------------------------------------------------------------------------
def _date_ok(result, max_days=None):
    d = result.suggested_due_date
    if d is None or d < today():
        return False
    return True if max_days is None else (d - today()).days <= max_days


CASES = [
    {
        "id": "C1_explicit_high",
        "call": dict(action="classify_priority", task_id=2),
        "expect_tool": "lookup_task",
        "check": lambda r: r.priority == "high",
        "note": "Task already marked high, due in 2 days",
    },
    {
        "id": "C2_housekeeping_low",
        "call": dict(action="classify_priority", task_id=7),
        "expect_tool": "lookup_task",
        "check": lambda r: r.priority == "low",
        "note": "Archive job, due in 25 days, nobody waiting",
    },
    {
        "id": "C3_unset_medium",
        "call": dict(action="classify_priority", task_id=4),
        "expect_tool": "lookup_task",
        "check": lambda r: r.priority == "medium",
        "note": "No priority set, due in 10 days (rubric says medium)",
    },
    {
        "id": "C4_overdue_bug_high",
        "call": dict(action="classify_priority", task_id=5),
        "expect_tool": "lookup_task",
        "check": lambda r: r.priority == "high",
        "note": "Overdue production bug",
    },
    {
        "id": "C5_text_urgent",
        "call": dict(action="classify_priority", title="Server room aircon failed", description="Temperature rising, servers will overheat within hours"),
        "expect_tool": None,
        "check": lambda r: r.priority == "high",
        "note": "Free text, clearly urgent",
    },
    {
        "id": "C6_text_trivial",
        "call": dict(action="classify_priority", title="Reorganise browser bookmarks", description="Whenever, no deadline"),
        "expect_tool": None,
        "check": lambda r: r.priority == "low",
        "note": "Free text, clearly trivial",
    },
    {
        "id": "C7_due_date_overdue",
        "call": dict(action="suggest_due_date", task_id=5),
        "expect_tool": "lookup_task",
        "check": lambda r: _date_ok(r, max_days=3),
        "note": "Overdue task: date must not be in the past and should be soon",
    },
    {
        "id": "C8_due_date_none",
        "call": dict(action="suggest_due_date", task_id=6),
        "expect_tool": "lookup_task",
        "check": lambda r: _date_ok(r, max_days=30),
        "note": "No due date, description says mid month",
    },
    {
        "id": "C9_weekly_summary",
        "call": dict(action="weekly_summary", days_ahead=7),
        "expect_tool": "list_open_tasks",
        "check": lambda r: bool(r.summary) and ("COIT12204" in r.summary or "WiFi" in r.summary or "login" in r.summary.lower()),
        "note": "Summary should name at least one of the urgent tasks",
    },
    {
        "id": "C10_missing_task",
        "call": dict(action="classify_priority", task_id=404),
        "expect_tool": "lookup_task",
        "check": lambda r: r.priority is None or r.confidence <= 0.2,
        "note": "Task does not exist: model must not invent a priority",
    },
]


def run_version(version: str, repeats: int, store: TaskStore) -> dict:
    state = AgentState(persist_path=Path(tempfile.mkdtemp()) / "state.json")
    agent = TaskAgent(store, state, prompt_version=version)
    rows = []
    answers = defaultdict(list)
    print(f"\n=== Prompt {version} ===")
    for case in CASES:
        for rep in range(1, repeats + 1):
            run = agent.run(**case["call"])
            r = run.result
            answer = r.priority or (r.suggested_due_date.isoformat() if r.suggested_due_date else None) or (r.summary or "")[:60]
            answers[case["id"]].append(answer)
            row = {
                "case": case["id"],
                "repeat": rep,
                "parsed": not run.fallback_used,
                "correct": bool(case["check"](r)) and not run.fallback_used,
                "expected_tool": case["expect_tool"],
                "tools_called": run.tools_called,
                "tools_ok": (case["expect_tool"] in run.tools_called) if case["expect_tool"] else True,
                "warnings": run.warnings,
                "error": run.error,
                "llm_calls": run.llm_calls,
                # a retry happened if there were more model calls than tool rounds plus the final answer
                "retry": run.llm_calls > len(run.tools_called) + 1,
                "latency_ms": run.latency_ms,
                "answer": answer,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "raw_output": (run.raw_output or "")[:400],
            }
            rows.append(row)
            flag = "OK " if row["correct"] else "BAD"
            fb = " (fallback)" if run.fallback_used else ""
            print(f"  [{flag}] {case['id']:<22} rep{rep} -> {answer!s:<40} conf={r.confidence:.2f} tools={run.tools_called}{fb}")
            if run.warnings:
                print(f"        warnings: {run.warnings}")

    n = len(rows)
    consistent = sum(1 for cid, a in answers.items() if len(set(map(str, a))) == 1)
    summary = {
        "version": version,
        "model": config.MODEL_NAME,
        "runs": n,
        "parsed_rate": sum(r["parsed"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "tool_use_rate": sum(r["tools_ok"] for r in rows) / n,
        "warning_runs": sum(1 for r in rows if r["warnings"]),
        "retry_runs": sum(1 for r in rows if r["retry"]),
        "consistency": consistent / len(CASES),
        "mean_latency_ms": int(statistics.mean(r["latency_ms"] for r in rows)),
        "state_after": state.snapshot(),
    }
    print(
        f"  parsed={summary['parsed_rate']:.0%} accuracy={summary['accuracy']:.0%} "
        f"tools={summary['tool_use_rate']:.0%} warnings={summary['warning_runs']} retries={summary['retry_runs']} "
        f"consistency={summary['consistency']:.0%} latency={summary['mean_latency_ms']}ms"
    )
    return {"summary": summary, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", default=[config.PROMPT_VERSION])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reseed", action="store_true", help="Rewrite data/tasks.json with dates relative to today first")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs on the console (they always go to logs/agent.log anyway)")
    args = parser.parse_args(argv)

    if not config.API_KEY:
        sys.exit(f"No API key for provider '{config.LLM_PROVIDER}'. Put it in .env first.")

    logger = setup_logging()
    if not args.verbose:  # keeps the console readable, the file log still has everything
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)
    OUT_DIR.mkdir(exist_ok=True)
    if args.reseed or not config.TASKS_FILE.exists():
        config.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.TASKS_FILE.write_text(json.dumps([t.model_dump(mode="json") for t in seed_tasks()], indent=2))
        print(f"Seeded {config.TASKS_FILE} with dates relative to {today()}")
    store = TaskStore(config.TASKS_FILE)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = []
    for version in args.versions:
        result = run_version(version, args.repeats, store)
        path = OUT_DIR / f"results_{version}_{stamp}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  saved {path.relative_to(ROOT)}")
        summaries.append(result["summary"])

    md = [f"\n## Evaluation run {stamp} (model {config.MODEL_NAME}, {args.repeats} repeats, {len(CASES)} cases)\n",
          "| Prompt | Parsed | Accuracy | Tool use | Runs needing retry | Runs needing correction | Consistency | Mean latency |",
          "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        md.append(
            f"| {s['version']} | {s['parsed_rate']:.0%} | {s['accuracy']:.0%} | {s['tool_use_rate']:.0%} | "
            f"{s['retry_runs']}/{s['runs']} | {s['warning_runs']}/{s['runs']} | {s['consistency']:.0%} | {s['mean_latency_ms']} ms |"
        )
    with (OUT_DIR / "summary.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
