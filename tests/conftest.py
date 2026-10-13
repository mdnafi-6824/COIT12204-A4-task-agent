"""Fixtures shared by all my tests.

The important ones:
    fixed_today   freezes the date so the due date maths always comes out the same
    store         a TaskStore in a temp folder with the sample tasks
    state         a fresh AgentState saved to a temp file
    make_agent    builds a TaskAgent around a FakeChatModel and monkeypatches
                  app.agent.get_llm so nothing can reach OpenAI by accident
    client        a FastAPI TestClient wired up to the fake agent
"""

import logging
import os
import tempfile
from datetime import date

# The log file path is read when app.config is imported, so I point it at a
# temp file here, before anything imports the app. Otherwise the tests would
# write into the real logs/agent.log.
os.environ.setdefault("LOG_FILE", os.path.join(tempfile.mkdtemp(), "test_agent.log"))

import pytest
from fastapi.testclient import TestClient

from app import agent as agent_module
from app import guardrails, store as store_module, tools as tools_module
from app.agent import TaskAgent
from app.agents import ReviewerAgent, WorkerAgent
from app.agents import reviewer as reviewer_module
from app.main import create_app
from app.orchestrator import Orchestrator
from app.state import AgentState
from app.store import TaskStore, seed_tasks
from tests.fakes import FakeChatModel

TODAY = date(2026, 9, 15)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    """Freeze 'today' in every module that asks for it."""
    monkeypatch.setattr(store_module, "today", lambda: TODAY)
    monkeypatch.setattr(tools_module, "today", lambda: TODAY)
    monkeypatch.setattr(agent_module, "today", lambda: TODAY)
    monkeypatch.setattr(guardrails, "today", lambda: TODAY)
    monkeypatch.setattr(reviewer_module, "today", lambda: TODAY)
    return TODAY


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Safety net. If any code path calls get_llm() it gets a fake that fails loudly."""
    monkeypatch.setattr(
        agent_module, "get_llm", lambda: FakeChatModel(raise_error=AssertionError("real LLM requested in tests"))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "tasks.json"
    s = TaskStore(path, seed_if_missing=False)
    for task in seed_tasks(base=TODAY):
        s._tasks[task.id] = task
    s.save()
    tools_module.set_store(s)
    return s


@pytest.fixture
def state(tmp_path):
    # small history limit so I can test the cap without 50 runs
    return AgentState(persist_path=tmp_path / "state.json", history_limit=5)


@pytest.fixture
def make_agent(store, state, monkeypatch):
    """Factory: make_agent(responses=[...]) or make_agent(raise_error=...)"""

    def _make(responses=None, raise_error=None, prompt_version="v3"):
        fake = FakeChatModel(responses=responses, raise_error=raise_error)
        # This is the monkeypatch the brief asks for. get_llm() now returns my fake,
        # so even TaskAgent(store, state) with no llm argument uses it.
        monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
        agent = TaskAgent(store, state, llm=fake, prompt_version=prompt_version)
        agent.fake = fake  # type: ignore[attr-defined]
        return agent

    return _make


@pytest.fixture
def make_orchestrator(store, make_agent, tmp_path):
    """Factory for the two agent system around a scripted model.
    make_orchestrator(responses=[...], use_llm=False, **orchestrator_kwargs)"""

    def _make(responses=None, raise_error=None, use_llm=False, reviewer_llm=None, **kwargs):
        agent = make_agent(responses=responses, raise_error=raise_error)
        worker = WorkerAgent(agent, store)
        reviewer = ReviewerAgent(store, use_llm=use_llm, llm=reviewer_llm)
        kwargs.setdefault("persist_path", tmp_path / "conversations.json")
        orch = Orchestrator({"worker": worker, "reviewer": reviewer}, **kwargs)
        orch.fake = agent.fake  # type: ignore[attr-defined]
        orch.worker, orch.reviewer = worker, reviewer  # type: ignore[attr-defined]
        return orch

    return _make


@pytest.fixture
def client(store, state, make_agent, make_orchestrator):
    """TestClient with a fake agent behind it. Tests script the fake via client.agent.fake."""
    agent = make_agent(responses=[])
    orch = make_orchestrator()
    orch.agents["worker"].agent = agent          # both paths share one fake model
    app = create_app(store=store, state=state, agent=agent, orchestrator=orch)
    with TestClient(app) as c:
        c.agent = agent  # type: ignore[attr-defined]
        c.orch = orch  # type: ignore[attr-defined]
        yield c


@pytest.fixture
def caplog_agent(caplog):
    caplog.set_level(logging.INFO, logger="task_agent")
    return caplog
