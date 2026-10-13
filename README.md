# COIT12204 Task Agent (Assessments 3 and 4)

A single LangChain based agent embedded in a small task manager API. The agent
can classify a task's priority, suggest a due date, or write a weekly summary.
It calls tools to read real task data, returns structured JSON, tracks its own
state, and falls back to simple rules when the language model fails. A small
glass style dashboard at `/` shows the tasks, runs the agent, and streams the
state, history and log live.

Assessment 4 turns it into a two agent system. The A3 agent becomes the
**Worker**, a **Reviewer** agent checks every answer against the real task
data, and an **Orchestrator** routes validated messages between them, sends
rejected answers back for revision, and enforces max revisions, max steps,
timeouts and fallbacks. See "Two agent system" below.



## Project layout

```
app/
  main.py           FastAPI application, endpoints, error handlers, request logging
  static/index.html The dashboard (plain HTML, CSS and JS, no build step)
  messages.py       AgentMessage schema: sender, receiver, intent, content, constraints (A4)
  orchestrator.py   Routes messages, tracks steps, max revisions/steps, timeouts, fallback (A4)
  agents/base.py    BaseAgent: accepted intents, logging of received and sent messages (A4)
  agents/worker.py  WorkerAgent: the A3 TaskAgent behind the message protocol (A4)
  agents/reviewer.py ReviewerAgent: rule checks plus an optional model verdict (A4)
  agent.py          TaskAgent: prompt | llm.bind_tools chain, tool loop, parsing, fallback
  tools.py          LangChain tools: lookup_task, list_open_tasks, summarise_tasks
  state.py          AgentState: tracked variables, transition logging, JSON persistence
  prompts.py        System prompt versions v1, v2, v3 and the human message templates
  guardrails.py     Input checks (length, injection) and output checks (dates, fields)
  models.py         Pydantic schemas for tasks, agent output, requests and responses
  store.py          JSON backed task store with sample data
  config.py         Environment driven settings
  logging_config.py Console + file logging (logs/agent.log)
tests/
  conftest.py       Fixtures: frozen date, temp store/state, fake LLM via monkeypatch, TestClient
  fakes.py          FakeChatModel that replays scripted AIMessages
  test_tools.py     Tool behaviour
  test_state.py     State transitions, logging, persistence
  test_agent.py     Agent wrapper, tool loop, parsing retry, fallback, guardrails
  test_api.py       Endpoint integration, validation, error handling, logging
  test_messages.py  Message schema validation (A4)
  test_agents.py    Worker and Reviewer on their own (A4)
  test_orchestrator.py Routing, revision loop, limits, timeouts, invalid receivers (A4)
scripts/
  evaluate.py       Runs the single agent evaluation cases against the real model
  evaluate_multi.py Runs the two agent evaluation, saves transcripts (A4)
  seed_tasks.py     Rewrites data/tasks.json with dates relative to today
data/tasks.json     Sample tasks (run scripts/seed_tasks.py to refresh the dates)
data/agent_state.json  Persisted agent state (created on first run)
logs/agent.log      Log file
```

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your API key in .env (see below)
```

`.env` values:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic`. |
| `OPENAI_API_KEY` | (none) | Needed when the provider is openai. Tests never use it. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model with tool calling. |
| `ANTHROPIC_API_KEY` | (none) | Needed when the provider is anthropic. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Any Claude model with tool calling. |
| `PROMPT_VERSION` | `v3` | `v1`, `v2` or `v3`. See `app/prompts.py`. |
| `MAX_TOOL_ROUNDS` | `3` | Tool calling rounds before the agent forces an answer. |
| `MAX_INPUT_CHARS` | `500` | Free text longer than this is rejected with 400. |
| `LLM_TIMEOUT_SECONDS` | `30` | Per call timeout for the model. |
| `HISTORY_LIMIT` | `50` | Entries kept in the state history. |
| `TASKS_FILE`, `STATE_FILE`, `LOG_FILE` | `data/tasks.json`, `data/agent_state.json`, `logs/agent.log` | Where things are stored. |
| `REVIEWER_USE_LLM` | `true` | Reviewer asks the model for a verdict after the rules pass. `false` = rules only. |
| `MAX_REVISIONS` | `2` | How many times the Reviewer can send an answer back. |
| `MAX_STEPS` | `10` | Hard cap on messages in one conversation. |
| `AGENT_TIMEOUT_SECONDS` | `45` | Per agent call timeout inside the orchestrator. |

## Run the API

```bash
python -m scripts.seed_tasks        # optional: fresh sample dates
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ for the dashboard, or http://127.0.0.1:8000/docs for
the Swagger UI.

Example calls:

```bash
curl -X POST localhost:8000/api/agent/classify_priority -H "content-type: application/json" -d '{"task_id": 5}'
curl -X POST localhost:8000/api/agent/suggest_due_date  -H "content-type: application/json" -d '{"task_id": 6}'
curl -X POST localhost:8000/api/agent/weekly_summary    -H "content-type: application/json" -d '{"days_ahead": 7}'
curl localhost:8000/api/agent/state
curl localhost:8000/api/agent/history
```

| Endpoint | Method | Body | Purpose |
|---|---|---|---|
| `/` | GET | | Dashboard |
| `/api/agent/classify_priority` | POST | `{task_id}` or `{title, description}` | Priority (low/medium/high) with reasoning and confidence |
| `/api/agent/suggest_due_date` | POST | `{task_id}` | ISO due date suggestion |
| `/api/agent/weekly_summary` | POST | `{days_ahead}` (1 to 30, default 7) | Plain English summary of what is due |
| `/api/agent/state` | GET | | Current AgentState snapshot |
| `/api/agent/history` | GET | `?limit=20` | Recent runs and tool calls |
| `/api/agent/logs` | GET | `?lines=60` | Tail of `logs/agent.log` |
| `/api/agent/state/reset` | POST | | Clear the state |
| `/api/tasks`, `/api/tasks/{id}` | GET/POST | | Plain task CRUD for demos |
| `/health` | GET | | Liveness and current config |
| `/api/orchestrator/run` | POST | `{action, task_id \| title, description, days_ahead}` | Run the two agent system, returns the whole conversation |
| `/api/orchestrator/conversations` | GET | `?limit=10` | Recent conversations with every message |
| `/api/orchestrator/conversations/{id}` | GET | | One conversation |
| `/api/orchestrator/stats` | GET | | Approved, revisions, fallbacks, mean steps |

Every agent response has the same shape: `result` (the structured
`AgentResult`), `fallback_used`, `tools_called`, `warnings` (output guardrail
corrections), `prompt_version` and a `state` snapshot. Errors come back as
`{"error", "detail", "request_id"}` with 400 (guardrail rejected the input),
404 (unknown task), 422 (validation) or 500.

## Run the tests

```bash
pytest            # 116 tests, a few seconds, no network
pytest -v         # one line per test
```

The tests never call a real model. `tests/conftest.py` monkeypatches
`app.agent.get_llm` so the agent receives a `FakeChatModel` that replays
scripted `AIMessage`s (tool requests, JSON answers, prose, or an exception).
`TestClient` drives the FastAPI app with the same fake behind it.

## Run the evaluation

```bash
python -m scripts.evaluate --versions v1 v2 v3 --repeats 2
```

This runs 10 fixed cases against the real model for each prompt version and
writes `evaluation/results_<version>_<timestamp>.json` plus a summary table in
`evaluation/summary.md`. It is roughly 40 to 80 model calls per prompt
version. The results from my own runs are kept in `evaluation/` as evidence
for the report.

There is no authentication on any endpoint and `/api/agent/logs` returns the
server log, so this is meant for local use only.

## Two agent system (Assessment 4)

```
user -> worker          execute_task      {request}
worker -> reviewer      review_result     {result, task, fallback_used, warnings, tools_called}
reviewer -> orchestrator result_approved  {result, checked_by}
                     or revision_requested {issues, feedback}
orchestrator -> worker  revise_result     {request, previous_result, feedback}   (loop, max MAX_REVISIONS)
orchestrator -> user    final_result      {status, result, revisions, errors}
```

Every message is an `AgentMessage` (`app/messages.py`) with `sender`,
`receiver`, `intent`, `content`, `constraints` (`format`, `max_revisions`,
`timeout_seconds`), plus `conversation_id`, `step`, `in_reply_to`,
`message_id` and `sent_at`. Pydantic rejects unknown receivers, unknown
intents, an intent the receiver does not accept, and an agent messaging
itself. The orchestrator logs `ROUTE`, each agent logs `MSG RECEIVED` and
`MSG SENT`, the reviewer logs `REVIEW approved` or `REVIEW rejected`, and
the orchestrator logs `CONVERSATION START` and `CONVERSATION END`.

The Reviewer checks with rules first (due date not in the past, priority
matches the due date rubric unless the task owner set one, summaries must
mention overdue work and name an urgent task, a first fallback answer is
sent back once) and then, if `REVIEWER_USE_LLM` is on, asks the model for a
JSON verdict. A rule rejection can never be overridden by the model. If the
reviewer's model call fails, it carries on with the rules.

Resilience: every agent call runs under a timeout; a timed out or crashed
agent is asked for its fallback answer (the Worker uses the rule based
fallback from A3, the Reviewer approves unreviewed with a warning); the
revision loop stops at `MAX_REVISIONS` and returns the last answer marked
`max_revisions`; `MAX_STEPS` is a hard cap; a message to an unregistered
receiver is a logged routing error and the last good answer is returned as
a `fallback`.

Run it from the dashboard ("Two agents" tab) or:

```bash
curl -X POST localhost:8000/api/orchestrator/run -H "content-type: application/json" -d '{"action": "suggest_due_date", "task_id": 5}'
python -m scripts.evaluate_multi --repeats 2            # evaluation with transcripts
python -m scripts.evaluate_multi --reviewer rules       # rules only reviewer
python -m scripts.evaluate_multi --break-worker         # worker model down, shows the fallback path
```

Results and full transcripts land in `evaluation/multi/` as
`results_<stamp>.json` and `transcripts_<stamp>.json`, with a running
`summary.md`. The runs I did while debugging are kept in
`evaluation/multi/before_fix/` and `evaluation/multi/after_fix_1/`; the
files at the top level are the final run.

## How a run works

1. The endpoint validates the body (Pydantic) and checks the task exists.
2. `TaskAgent.run()` builds a `HumanMessage` for the action and invokes the
   LCEL chain `ChatPromptTemplate | llm.bind_tools(TOOLS)`, where `llm` is
   `ChatAnthropic` or `ChatOpenAI` depending on `LLM_PROVIDER`.
3. If the model returns tool calls, each tool is executed with `tool.invoke`,
   the result is appended as a `ToolMessage`, and the chain runs again (at
   most `MAX_TOOL_ROUNDS` times).
4. The final text is parsed with `PydanticOutputParser` into `AgentResult`.
   If that fails the model is asked once more for JSON only.
5. Output guardrails correct or flag problems (wrong action, past due date,
   missing fields).
6. If anything raised, a rule based fallback produces an `AgentResult` instead.
7. `AgentState.record_run()` updates the counters and history, logs every
   transition, and saves to `data/agent_state.json`.
