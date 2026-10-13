"""The message schema. Every message between agents has to pass these checks
before the orchestrator will move it."""

import pytest
from pydantic import ValidationError

from app.messages import ALL_INTENTS, AgentMessage, Constraints


def msg(**over):
    base = dict(conversation_id="c1", step=0, sender="user", receiver="worker", intent="execute_task",
                content={"request": {"action": "weekly_summary"}})
    base.update(over)
    return AgentMessage(**base)


def test_valid_message_gets_an_id_and_timestamp():
    m = msg()
    assert len(m.message_id) == 10
    assert m.sent_at.endswith("+00:00")
    assert m.constraints == Constraints()
    assert m.short() == "[c1 step 0] user -> worker execute_task"


def test_unknown_intent_is_rejected():
    with pytest.raises(ValidationError, match="unknown intent"):
        msg(intent="delete_everything")


def test_receiver_must_accept_the_intent():
    # review_result is for the reviewer, the worker does not understand it
    with pytest.raises(ValidationError, match="does not accept intent"):
        msg(receiver="worker", intent="review_result")


def test_unknown_receiver_is_rejected():
    with pytest.raises(ValidationError):
        msg(receiver="planner")


def test_agent_cannot_message_itself():
    with pytest.raises(ValidationError, match="cannot send a message to itself"):
        msg(sender="worker", receiver="worker", intent="execute_task")


def test_constraints_are_bounded():
    with pytest.raises(ValidationError):
        Constraints(max_revisions=99)
    with pytest.raises(ValidationError):
        Constraints(format="xml")


def test_reply_links_back_to_the_original():
    original = msg()
    reply = original.reply(sender="worker", receiver="reviewer", intent="review_result", content={"result": {}}, step=1)
    assert reply.in_reply_to == original.message_id
    assert reply.conversation_id == "c1"
    assert reply.step == 1
    assert reply.constraints == original.constraints


def test_every_intent_has_a_receiver():
    assert set(ALL_INTENTS) == {
        "execute_task", "revise_result", "review_result", "result_approved",
        "revision_requested", "task_result", "agent_error", "final_result",
    }
