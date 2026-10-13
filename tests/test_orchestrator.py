"""The orchestrator: routing, the revision loop, limits, timeouts, invalid
receivers and the conversation record. The model is always the fake."""

import time

from app.agents.base import AgentError, BaseAgent
from app.messages import AgentMessage
from tests.fakes import json_answer, text_answer, tool_request

GOOD_PRIORITY = json_answer(action="classify_priority", task_id=4, priority="medium", reasoning="Due in 10 days.", confidence=0.85)


def transcript(conv):
    return [(m["step"], m["sender"], m["receiver"], m["intent"]) for m in conv.messages]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_straight_through_approval(make_orchestrator):
    orch = make_orchestrator(responses=[tool_request("lookup_task", task_id=4), GOOD_PRIORITY])
    conv = orch.run({"action": "classify_priority", "task_id": 4})
    assert conv.status == "approved"
    assert conv.revisions == 0
    assert conv.final_result["priority"] == "medium"
    assert transcript(conv) == [
        (0, "user", "worker", "execute_task"),
        (1, "worker", "reviewer", "review_result"),
        (2, "reviewer", "orchestrator", "result_approved"),
        (3, "orchestrator", "user", "final_result"),
    ]
    assert conv.steps == 3 and conv.errors == []
    assert conv.elapsed_ms >= 0 and conv.started_at


def test_rejection_then_revision_then_approval(make_orchestrator):
    orch = make_orchestrator(responses=[
        tool_request("lookup_task", task_id=5),
        json_answer(action="suggest_due_date", task_id=5, suggested_due_date="2026-09-30", reasoning="Two weeks.", confidence=0.8),
        tool_request("lookup_task", task_id=5),
        json_answer(action="suggest_due_date", task_id=5, suggested_due_date="2026-09-16", reasoning="Overdue, so tomorrow.", confidence=0.9),
    ])
    conv = orch.run({"action": "suggest_due_date", "task_id": 5})
    assert conv.status == "approved"
    assert conv.revisions == 1
    assert conv.final_result["suggested_due_date"] == "2026-09-16"
    intents = [m["intent"] for m in conv.messages]
    assert intents == ["execute_task", "review_result", "revision_requested", "revise_result", "review_result", "result_approved", "final_result"]
    # the revise message carried the reviewer's feedback to the worker
    revise = conv.messages[3]
    assert revise["sender"] == "orchestrator" and "already overdue" in revise["content"]["feedback"]
    assert revise["content"]["previous_result"]["suggested_due_date"] == "2026-09-30"
    # and the worker's second model call saw it
    assert "already overdue" in orch.fake.calls[2][1].content


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def test_max_revisions_returns_last_result(make_orchestrator):
    # task 5 is overdue, so a date two weeks out is rejected by the reviewer every time
    bad = lambda: json_answer(action="suggest_due_date", task_id=5, suggested_due_date="2026-09-30", reasoning="r", confidence=0.8)  # noqa: E731
    orch = make_orchestrator(responses=[bad(), bad(), bad(), bad()], max_revisions=2)
    conv = orch.run({"action": "suggest_due_date", "task_id": 5})
    assert conv.status == "max_revisions"
    assert conv.revisions == 2
    assert conv.final_result["suggested_due_date"] == "2026-09-30"   # last answer, returned as is
    assert any("max revisions" in e for e in conv.errors)
    assert conv.messages[-1]["intent"] == "final_result"
    assert conv.messages[-1]["content"]["status"] == "max_revisions"
    assert len(orch.fake.calls) == 3                # 1 attempt + 2 revisions, no 4th call


def test_max_steps_is_a_hard_cap(make_orchestrator):
    bad = lambda: json_answer(action="suggest_due_date", task_id=5, suggested_due_date="2026-09-30", reasoning="r", confidence=0.8)  # noqa: E731
    orch = make_orchestrator(responses=[bad()] * 6, max_revisions=5, max_steps=4)
    conv = orch.run({"action": "suggest_due_date", "task_id": 5})
    assert conv.status == "max_steps"
    assert conv.steps <= 5
    assert any("max steps" in e for e in conv.errors)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_worker_model_failure_goes_through_fallback_and_reviewer(make_orchestrator):
    orch = make_orchestrator(raise_error=ConnectionError("down"))
    conv = orch.run({"action": "classify_priority", "task_id": 5})
    # worker used its own rule fallback twice (first rejected by the reviewer, then accepted)
    assert conv.status == "approved"
    assert conv.revisions == 1
    assert conv.final_result["priority"] == "high"
    assert conv.messages[1]["content"]["fallback_used"] is True


def test_worker_timeout_uses_worker_fallback(make_orchestrator, monkeypatch):
    orch = make_orchestrator(responses=[GOOD_PRIORITY], agent_timeout=1)

    def slow(message):
        time.sleep(3)
        return None

    monkeypatch.setattr(orch.agents["worker"], "handle", slow)
    conv = orch.run({"action": "classify_priority", "task_id": 4})
    assert any("timed out after 1s" in e for e in conv.errors)
    intents = [m["intent"] for m in conv.messages]
    assert "agent_error" in intents
    assert conv.final_result is not None
    assert conv.status in ("approved", "max_revisions")


def test_reviewer_crash_approves_unreviewed(make_orchestrator, monkeypatch):
    orch = make_orchestrator(responses=[tool_request("lookup_task", task_id=4), GOOD_PRIORITY])

    def boom(message):
        raise AgentError("reviewer exploded")

    monkeypatch.setattr(orch.agents["reviewer"], "handle", boom)
    conv = orch.run({"action": "classify_priority", "task_id": 4})
    assert conv.status == "approved"
    assert conv.final_result["priority"] == "medium"
    assert any("reviewer unavailable" in e for e in conv.errors)
    assert [m["intent"] for m in conv.messages] == ["execute_task", "review_result", "agent_error", "result_approved", "final_result"]


def test_invalid_receiver_is_a_routing_error(make_orchestrator):
    orch = make_orchestrator(responses=[GOOD_PRIORITY])
    del orch.agents["reviewer"]                     # nobody to review
    conv = orch.run({"action": "classify_priority", "title": "Order toner"})
    assert conv.status == "fallback"                # the worker's answer is returned unreviewed
    assert any("no agent registered for receiver 'reviewer'" in e for e in conv.errors)
    assert conv.final_result["priority"] == "medium"


def test_agent_without_fallback_stops_the_conversation(make_orchestrator):
    class Brittle(BaseAgent):
        name = "reviewer"

        def process(self, message):
            raise AgentError("no")

    orch = make_orchestrator(responses=[GOOD_PRIORITY])
    orch.agents["reviewer"] = Brittle()
    conv = orch.run({"action": "classify_priority", "title": "Order toner"})
    assert conv.status == "fallback"
    assert conv.final_result["priority"] == "medium"
    assert any("reviewer error: no" in e for e in conv.errors)


# ---------------------------------------------------------------------------
# Logging and memory
# ---------------------------------------------------------------------------

def test_every_hop_is_logged(make_orchestrator, caplog_agent):
    orch = make_orchestrator(responses=[tool_request("lookup_task", task_id=4), GOOD_PRIORITY])
    conv = orch.run({"action": "classify_priority", "task_id": 4})
    msgs = [r.getMessage() for r in caplog_agent.records]
    cid = conv.id
    assert any(f"CONVERSATION START {cid}" in m for m in msgs)
    assert any(f"ROUTE [{cid} step 0] user -> worker execute_task -> worker" in m for m in msgs)
    assert any(f"MSG RECEIVED by worker [{cid} step 0]" in m for m in msgs)
    assert any(f"MSG SENT by worker [{cid} step 1] worker -> reviewer review_result" in m for m in msgs)
    assert any(f"ROUTE [{cid} step 1] worker -> reviewer review_result -> reviewer" in m for m in msgs)
    assert any("REVIEW approved" in m for m in msgs)
    assert any(f"CONVERSATION END {cid} status=approved steps=3 revisions=0" in m for m in msgs)


def test_conversations_are_kept_and_saved(make_orchestrator, tmp_path):
    orch = make_orchestrator(responses=[GOOD_PRIORITY, GOOD_PRIORITY], keep=1)
    orch.run({"action": "classify_priority", "title": "A"})
    second = orch.run({"action": "classify_priority", "title": "B"})
    assert [c.id for c in orch.conversations] == [second.id]
    saved = (tmp_path / "conversations.json").read_text()
    assert second.id in saved
    stats = orch.stats()
    assert stats["conversations"] == 1 and stats["approved"] == 1 and stats["mean_steps"] == 3.0


def test_bad_initial_request_is_an_error_not_a_crash(make_orchestrator):
    orch = make_orchestrator(responses=[text_answer("x")])
    orch.max_steps = 0
    conv = orch.run({"action": "classify_priority", "title": "A"})
    assert conv.status == "max_steps"
    assert conv.final_result is None
    assert conv.messages[-1]["intent"] == "final_result"


def test_agent_message_roundtrip_in_saved_conversation(make_orchestrator):
    orch = make_orchestrator(responses=[GOOD_PRIORITY])
    conv = orch.run({"action": "classify_priority", "title": "A"})
    for raw in conv.messages:
        AgentMessage.model_validate(raw)          # every stored message is still a valid message
