"""The Worker and the Reviewer on their own, without the orchestrator."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage

from app.agents import AgentError, ReviewerAgent, WorkerAgent
from app.messages import AgentMessage
from tests.fakes import FakeChatModel, json_answer, text_answer, tool_request


def execute(request, step=0, intent="execute_task", feedback=None):
    content = {"request": request}
    if feedback:
        content["feedback"] = feedback
    sender = "user" if intent == "execute_task" else "orchestrator"
    return AgentMessage(conversation_id="t", step=step, sender=sender, receiver="worker", intent=intent, content=content)


def review(result, task=None, request=None, fallback_used=False, revision=False, step=1):
    return AgentMessage(
        conversation_id="t", step=step, sender="worker", receiver="reviewer", intent="review_result",
        content={"result": result, "task": task, "request": request or {"action": result.get("action")},
                 "fallback_used": fallback_used, "revision": revision, "warnings": [], "tools_called": []},
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def test_worker_runs_the_agent_and_replies_to_reviewer(make_agent, store):
    agent = make_agent(responses=[
        tool_request("lookup_task", task_id=2),
        json_answer(action="classify_priority", task_id=2, priority="high", reasoning="Due in 2 days.", confidence=0.9),
    ])
    worker = WorkerAgent(agent, store)
    reply = worker.handle(execute({"action": "classify_priority", "task_id": 2}))
    assert reply.sender == "worker" and reply.receiver == "reviewer" and reply.intent == "review_result"
    assert reply.step == 1
    assert reply.content["result"]["priority"] == "high"
    assert reply.content["task"]["title"] == "Renew campus WiFi certificate"
    assert reply.content["tools_called"] == ["lookup_task"]
    assert reply.content["fallback_used"] is False
    assert reply.content["revision"] is False
    assert worker.received == 1 and worker.sent == 1


def test_worker_revision_passes_feedback_to_the_model(make_agent, store):
    agent = make_agent(responses=[json_answer(action="suggest_due_date", task_id=5, suggested_due_date="2026-09-16", reasoning="Tomorrow.", confidence=0.9)])
    worker = WorkerAgent(agent, store)
    reply = worker.handle(execute({"action": "suggest_due_date", "task_id": 5}, step=3, intent="revise_result", feedback="The date was in the past."))
    assert reply.content["revision"] is True
    human = agent.fake.calls[0][1].content
    assert "A reviewer rejected your previous answer" in human
    assert "The date was in the past." in human


def test_worker_truncates_long_feedback(make_agent, store):
    agent = make_agent(responses=[json_answer(action="weekly_summary", summary="One overdue: Fix login bug on student portal.", reasoning="r", confidence=0.9)])
    worker = WorkerAgent(agent, store)
    long_feedback = "The summary is wrong because " + "x" * 700
    reply = worker.handle(execute({"action": "weekly_summary"}, step=3, intent="revise_result", feedback=long_feedback))
    assert reply.intent == "review_result"          # no InputRejected crash
    human = agent.fake.calls[0][1].content
    assert "..." in human and len(human) < 1200


def test_worker_rejects_messages_not_for_it(make_agent, store):
    worker = WorkerAgent(make_agent(responses=[]), store)
    wrong = AgentMessage(conversation_id="t", step=0, sender="worker", receiver="reviewer", intent="review_result", content={})
    with pytest.raises(AgentError, match="addressed to reviewer"):
        worker.handle(wrong)


def test_worker_rejects_unknown_action(make_agent, store):
    worker = WorkerAgent(make_agent(responses=[]), store)
    with pytest.raises(AgentError, match="cannot run action"):
        worker.handle(execute({"action": "delete_all"}))


def test_worker_fallback_needs_no_model(make_agent, store):
    worker = WorkerAgent(make_agent(responses=[]), store)
    reply = worker.fallback(execute({"action": "classify_priority", "task_id": 5}), "timed out")
    assert reply.content["fallback_used"] is True
    assert reply.content["result"]["priority"] == "high"   # overdue, so the rule says high
    assert reply.content["warnings"] == ["worker fallback: timed out"]


# ---------------------------------------------------------------------------
# Reviewer rules
# ---------------------------------------------------------------------------

def test_reviewer_approves_a_correct_priority(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(4).model_dump(mode="json")   # due in 10 days, no priority set
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 4, "priority": "medium", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "result_approved" and reply.receiver == "orchestrator"
    assert reply.content["checked_by"] == "rules"


def test_reviewer_rejects_priority_that_breaks_the_rubric(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(4).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 4, "priority": "low", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "revision_requested"
    assert "rubric says 'medium'" in reply.content["feedback"]


def test_reviewer_keeps_an_explicit_priority_even_if_rubric_disagrees(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(7).model_dump(mode="json")   # low, due in 25 days, explicit priority
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 7, "priority": "medium", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "result_approved"      # explicit priorities are the owner's call


def test_reviewer_rejects_past_due_date(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(6).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "suggest_due_date", "task_id": 6, "suggested_due_date": "2026-09-01", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "revision_requested"
    assert "in the past" in reply.content["feedback"]


def test_reviewer_rejects_far_date_for_overdue_task(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(5).model_dump(mode="json")   # overdue
    reply = reviewer.handle(review({"action": "suggest_due_date", "task_id": 5, "suggested_due_date": "2026-09-30", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "revision_requested"
    assert "already overdue" in reply.content["feedback"]


def test_reviewer_rejects_summary_that_ignores_overdue_work(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    reply = reviewer.handle(review({"action": "weekly_summary", "summary": "Quiet week, nothing much on.", "reasoning": "r", "confidence": 0.9}))
    assert reply.intent == "revision_requested"
    assert "overdue" in reply.content["feedback"]
    assert "Fix login bug" in reply.content["feedback"]   # names the urgent tasks it should mention


def test_reviewer_approves_a_good_summary(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    summary = "One task is overdue (Fix login bug on student portal). Also due soon: Submit COIT12204 Assessment 3 and the WiFi certificate renewal."
    reply = reviewer.handle(review({"action": "weekly_summary", "summary": summary, "reasoning": "r", "confidence": 0.9}))
    assert reply.intent == "result_approved"


def test_reviewer_sends_a_first_fallback_answer_back_once(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(5).model_dump(mode="json")
    good = {"action": "classify_priority", "task_id": 5, "priority": "high", "reasoning": "fallback", "confidence": 0.3}
    first = reviewer.handle(review(good, task, fallback_used=True, revision=False))
    assert first.intent == "revision_requested"
    assert "fallback" in first.content["feedback"]
    again = reviewer.handle(review(good, task, fallback_used=True, revision=True, step=4))
    assert again.intent == "result_approved"    # second time it accepts, the model is clearly down


def test_reviewer_model_is_skipped_for_an_accepted_fallback(store):
    # the model would reject a fallback forever, so the rules decide alone on a revision
    verdict = FakeChatModel(responses=[AIMessage(content='{"approved": false, "issues": ["reasoning is a fallback message"], "feedback": "x"}')])
    reviewer = ReviewerAgent(store, use_llm=True, llm=verdict)
    task = store.get(5).model_dump(mode="json")
    good = {"action": "classify_priority", "task_id": 5, "priority": "high", "reasoning": "Fallback rule used", "confidence": 0.3}
    reply = reviewer.handle(review(good, task, fallback_used=True, revision=True, step=4))
    assert reply.intent == "result_approved"
    assert "fallback" in reply.content["warning"]
    assert verdict.calls == []        # the model was never asked


def test_reviewer_rejects_empty_reasoning_and_bad_confidence(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    task = store.get(2).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 2, "priority": "high", "reasoning": "  ", "confidence": 1.7}, task))
    assert reply.intent == "revision_requested"
    assert len(reply.content["issues"]) == 2


# ---------------------------------------------------------------------------
# Reviewer with the model layer
# ---------------------------------------------------------------------------

def test_reviewer_model_can_reject_after_rules_pass(store):
    verdict = FakeChatModel(responses=[AIMessage(content='{"approved": false, "issues": ["Reasoning contradicts the due date."], "feedback": "Fix the reasoning."}')])
    reviewer = ReviewerAgent(store, use_llm=True, llm=verdict)
    task = store.get(4).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 4, "priority": "medium", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "revision_requested"
    assert reply.content["checked_by"] == "model"
    assert reply.content["issues"] == ["Reasoning contradicts the due date."]


def test_reviewer_model_approval_is_parsed_from_chatty_reply(store):
    verdict = FakeChatModel(responses=[text_answer('Looks fine to me.\n\n```json\n{"approved": true, "issues": [], "feedback": null}\n```')])
    reviewer = ReviewerAgent(store, use_llm=True, llm=verdict)
    task = store.get(4).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 4, "priority": "medium", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "result_approved"
    assert reply.content["checked_by"] == "rules+model"


def test_reviewer_model_gets_the_task_list_for_a_summary(store):
    verdict = FakeChatModel(responses=[AIMessage(content='{"approved": true, "issues": [], "feedback": null}')])
    reviewer = ReviewerAgent(store, use_llm=True, llm=verdict)
    summary = "One task is overdue (Fix login bug on student portal). Also due soon: Submit COIT12204 Assessment 3."
    reply = reviewer.handle(review({"action": "weekly_summary", "summary": summary, "reasoning": "r", "confidence": 0.9},
                                   request={"action": "weekly_summary", "days_ahead": 7}))
    assert reply.intent == "result_approved"
    human = verdict.calls[0][1].content
    assert "open_tasks_due_or_overdue" in human and "Fix login bug on student portal" in human
    assert "overdue_count" in human and "total_tasks" in human and "tasks_by_status" in human


def test_reviewer_model_gets_title_for_unsaved_task(store):
    verdict = FakeChatModel(responses=[AIMessage(content='{"approved": true, "issues": [], "feedback": null}')])
    reviewer = ReviewerAgent(store, use_llm=True, llm=verdict)
    reply = reviewer.handle(review({"action": "classify_priority", "priority": "high", "reasoning": "r", "confidence": 0.9},
                                   request={"action": "classify_priority", "title": "Server room aircon failed", "description": "overheating"}))
    assert reply.intent == "result_approved"
    assert "Server room aircon failed" in verdict.calls[0][1].content


def test_reviewer_falls_back_to_rules_when_model_fails(store, caplog_agent):
    broken = FakeChatModel(raise_error=TimeoutError("slow"))
    reviewer = ReviewerAgent(store, use_llm=True, llm=broken)
    task = store.get(4).model_dump(mode="json")
    reply = reviewer.handle(review({"action": "classify_priority", "task_id": 4, "priority": "medium", "reasoning": "r", "confidence": 0.8}, task))
    assert reply.intent == "result_approved"
    assert reply.content["checked_by"] == "rules"     # honest about who actually checked it
    assert any("REVIEW model error" in r.getMessage() for r in caplog_agent.records)


def test_reviewer_fallback_approves_unreviewed(store):
    reviewer = ReviewerAgent(store, use_llm=False)
    m = review({"action": "classify_priority", "priority": "low", "reasoning": "r", "confidence": 0.5})
    reply = reviewer.fallback(m, "timed out")
    assert reply.intent == "result_approved"
    assert reply.content["checked_by"] == "none"
    assert "reviewer unavailable" in reply.content["warning"]


def test_messages_are_logged_on_both_sides(make_agent, store, caplog_agent):
    agent = make_agent(responses=[json_answer(action="weekly_summary", summary="One overdue: Fix login bug on student portal.", reasoning="r", confidence=0.9)])
    worker = WorkerAgent(agent, store)
    reply = worker.handle(execute({"action": "weekly_summary"}))
    ReviewerAgent(store, use_llm=False).handle(reply)
    msgs = [r.getMessage() for r in caplog_agent.records]
    assert any("MSG RECEIVED by worker [t step 0] user -> worker execute_task" in m for m in msgs)
    assert any("MSG SENT by worker [t step 1] worker -> reviewer review_result" in m for m in msgs)
    assert any("MSG RECEIVED by reviewer [t step 1]" in m for m in msgs)
    assert any("MSG SENT by reviewer [t step 2] reviewer -> orchestrator result_approved" in m for m in msgs)


def test_fixed_today_is_applied_to_reviewer():
    from app.agents import reviewer as rv
    assert rv.today() == date(2026, 9, 15)
