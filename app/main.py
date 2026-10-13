"""The FastAPI app that exposes my agent.

Endpoints
    GET  /                              the dashboard (app/static/index.html)
    POST /api/agent/classify_priority   body: {task_id} or {title, description}
    POST /api/agent/suggest_due_date    body: {task_id}
    POST /api/agent/weekly_summary      body: {days_ahead}
    GET  /api/agent/state               current AgentState snapshot
    GET  /api/agent/history             recent runs and tool calls
    GET  /api/agent/logs                last lines of logs/agent.log
    POST /api/agent/state/reset         clear the state
    GET  /api/tasks, GET /api/tasks/{id}, POST /api/tasks   plain CRUD for the demo
    GET  /health

Assessment 4, the two agent system:
    POST /api/orchestrator/run            run a request through Worker -> Reviewer -> (revise) -> final
    GET  /api/orchestrator/conversations  recent conversations with every message
    GET  /api/orchestrator/conversations/{id}
    GET  /api/orchestrator/stats

Every request gets a short id. It shows up in the log lines and in the JSON
response, so in the demo I can match a log line to the call that made it."""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.agent import AgentRun, TaskAgent
from app.agents import ReviewerAgent, WorkerAgent
from app.guardrails import InputRejected
from app.logging_config import get_logger, setup_logging
from app.models import (
    AgentResponse,
    ClassifyPriorityRequest,
    ErrorResponse,
    OrchestratorRequest,
    StateSnapshot,
    SuggestDueDateRequest,
    Task,
    TaskCreate,
    WeeklySummaryRequest,
)
from app.orchestrator import Orchestrator, conversation_to_dict
from app.state import AgentState
from app.store import TaskStore

log = get_logger("api")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# The dashboard polls these every few seconds. I don't log those GETs,
# otherwise the log is nothing but polling and the interesting lines get lost.
QUIET_PATHS = {"/", "/health", "/api/agent/state", "/api/agent/history", "/api/agent/logs",
               "/api/orchestrator/conversations", "/api/orchestrator/stats"}


def create_app(
    store: Optional[TaskStore] = None,
    state: Optional[AgentState] = None,
    agent: Optional[TaskAgent] = None,
    orchestrator: Optional[Orchestrator] = None,
) -> FastAPI:
    """App factory. My tests pass in their own store, state and agent so they
    never touch the real data files or the real model."""
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # I build the heavy stuff here rather than at import time
        app.state.store = store or TaskStore(config.TASKS_FILE)
        app.state.agent_state = state or AgentState.load(config.STATE_FILE)
        app.state.agent = agent or TaskAgent(app.state.store, app.state.agent_state)
        # the two agent system shares the same TaskAgent as its Worker
        app.state.orchestrator = orchestrator or Orchestrator(
            {
                "worker": WorkerAgent(app.state.agent, app.state.store),
                "reviewer": ReviewerAgent(app.state.store),
            }
        )
        log.info("API started with %d tasks", len(app.state.store.all()))
        yield
        app.state.agent_state.save()
        log.info("API shutting down, state saved")

    app = FastAPI(
        title="COIT12204 Task Agent",
        description="Single LangChain agent for a task manager, with state tracking and guardrails.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware and error handlers
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        quiet = request.method == "GET" and request.url.path in QUIET_PATHS
        started = time.perf_counter()
        if not quiet:
            log.info("REQUEST %s %s %s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            # unhandled errors would otherwise skip this middleware and lose the request id
            log.exception("UNHANDLED %s %s", request_id, exc)
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": type(exc).__name__, "request_id": request_id},
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        if not quiet:
            log.info("RESPONSE %s status=%s %dms", request_id, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None)
        log.warning("VALIDATION %s %s", rid, exc.errors())
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request", "detail": _flatten_errors(exc), "request_id": rid},
        )

    @app.exception_handler(InputRejected)
    async def guardrail_error(request: Request, exc: InputRejected):
        rid = getattr(request.state, "request_id", None)
        log.warning("GUARDRAIL REJECT %s %s", rid, exc)
        return JSONResponse(status_code=400, content={"error": "Input rejected", "detail": str(exc), "request_id": rid})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "detail": None, "request_id": rid})

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # I send back the exception type but never the traceback
        rid = getattr(request.state, "request_id", None)
        log.exception("UNHANDLED %s %s", rid, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": type(exc).__name__, "request_id": rid},
        )

    # ------------------------------------------------------------------
    # Small helpers so the routes stay short
    # ------------------------------------------------------------------
    def _agent(request: Request) -> TaskAgent:
        return request.app.state.agent

    def _store(request: Request) -> TaskStore:
        return request.app.state.store

    def _require_task(request: Request, task_id: int) -> Task:
        task = _store(request).get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task

    def _format(request: Request, run: AgentRun) -> AgentResponse:
        return AgentResponse(
            request_id=request.state.request_id,
            action=run.action,  # type: ignore[arg-type]
            result=run.result,
            fallback_used=run.fallback_used,
            tools_called=run.tools_called,
            prompt_version=run.prompt_version,
            warnings=run.warnings,
            state=StateSnapshot(**_agent(request).state.snapshot()),
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    # ------------------------------------------------------------------
    # Agent endpoints
    # ------------------------------------------------------------------
    ERRORS = {400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}

    @app.post("/api/agent/classify_priority", response_model=AgentResponse, responses=ERRORS, tags=["agent"])
    def classify_priority(body: ClassifyPriorityRequest, request: Request):
        if body.task_id is not None:
            _require_task(request, body.task_id)
        run = _agent(request).run(
            "classify_priority", task_id=body.task_id, title=body.title, description=body.description
        )
        return _format(request, run)

    @app.post("/api/agent/suggest_due_date", response_model=AgentResponse, responses=ERRORS, tags=["agent"])
    def suggest_due_date(body: SuggestDueDateRequest, request: Request):
        _require_task(request, body.task_id)
        run = _agent(request).run("suggest_due_date", task_id=body.task_id)
        return _format(request, run)

    @app.post("/api/agent/weekly_summary", response_model=AgentResponse, responses=ERRORS, tags=["agent"])
    def weekly_summary(request: Request, body: Optional[WeeklySummaryRequest] = None):
        body = body or WeeklySummaryRequest()
        run = _agent(request).run("weekly_summary", days_ahead=body.days_ahead)
        return _format(request, run)

    # ------------------------------------------------------------------
    # Two agent system (Assessment 4)
    # ------------------------------------------------------------------
    def _orch(request: Request) -> Orchestrator:
        return request.app.state.orchestrator

    @app.post("/api/orchestrator/run", responses=ERRORS, tags=["orchestrator"])
    def orchestrator_run(body: OrchestratorRequest, request: Request):
        if body.task_id is not None:
            _require_task(request, body.task_id)
        conv = _orch(request).run(body.model_dump(exclude_none=True))
        data = conversation_to_dict(conv)
        data["request_id"] = request.state.request_id
        return data

    @app.get("/api/orchestrator/conversations", tags=["orchestrator"])
    def orchestrator_conversations(request: Request, limit: int = 10):
        limit = max(1, min(limit, 50))
        return {"conversations": [conversation_to_dict(c) for c in _orch(request).conversations[-limit:]]}

    @app.get("/api/orchestrator/conversations/{conversation_id}", tags=["orchestrator"])
    def orchestrator_conversation(conversation_id: str, request: Request):
        for c in _orch(request).conversations:
            if c.id == conversation_id:
                return conversation_to_dict(c)
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    @app.get("/api/orchestrator/stats", tags=["orchestrator"])
    def orchestrator_stats(request: Request):
        return _orch(request).stats()

    @app.get("/api/agent/state", response_model=StateSnapshot, tags=["agent"])
    def agent_state(request: Request):
        return StateSnapshot(**_agent(request).state.snapshot())

    @app.get("/api/agent/history", tags=["agent"])
    def agent_history(request: Request, limit: int = 20):
        limit = max(1, min(limit, config.HISTORY_LIMIT))
        return {"history": _agent(request).state.history[-limit:]}

    @app.get("/api/agent/logs", tags=["agent"])
    def agent_logs(lines: int = 60):
        # the dashboard polls this so it can show the log live
        lines = max(1, min(lines, 500))
        try:
            text = config.LOG_FILE.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"lines": []}
        return {"lines": text.splitlines()[-lines:]}

    @app.post("/api/agent/state/reset", response_model=StateSnapshot, tags=["agent"])
    def reset_state(request: Request):
        _agent(request).state.reset()
        return StateSnapshot(**_agent(request).state.snapshot())

    # ------------------------------------------------------------------
    # Task endpoints (plain CRUD so the demo has data to point at)
    # ------------------------------------------------------------------
    @app.get("/api/tasks", response_model=list[Task], tags=["tasks"])
    def list_tasks(request: Request):
        return _store(request).all()

    @app.get("/api/tasks/{task_id}", response_model=Task, tags=["tasks"])
    def get_task(task_id: int, request: Request):
        return _require_task(request, task_id)

    @app.post("/api/tasks", response_model=Task, status_code=201, tags=["tasks"])
    def create_task(body: TaskCreate, request: Request):
        return _store(request).add(body)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "model": config.MODEL_NAME, "prompt_version": config.PROMPT_VERSION}

    return app


def _flatten_errors(exc: RequestValidationError) -> str:
    # turns pydantic's error list into one readable string
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts)


app = create_app()
