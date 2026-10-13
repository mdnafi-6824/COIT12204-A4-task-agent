"""The message schema the two agents use to talk to each other.

Every message between agents has the same shape, which is the one from the
brief with a few extra fields so I can trace a conversation:

    {
      "sender": "worker",
      "receiver": "reviewer",
      "intent": "review_result",
      "content": {...},
      "constraints": {"format": "json", "max_revisions": 2},
      "conversation_id": "...", "step": 2, "in_reply_to": "...", "message_id": "...", "sent_at": "..."
    }

Pydantic validates it. Unknown senders, receivers or intents are rejected
before the message goes anywhere, and the orchestrator logs the rejection
as a routing failure rather than crashing."""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

AgentName = Literal["user", "worker", "reviewer", "orchestrator"]

# Which intents each receiver understands. The orchestrator checks this
# before routing so a message can't land on an agent that has no idea what
# to do with it.
INTENTS_BY_RECEIVER: dict[str, set[str]] = {
    "worker": {"execute_task", "revise_result"},
    "reviewer": {"review_result"},
    "orchestrator": {"result_approved", "revision_requested", "task_result", "agent_error"},
    "user": {"final_result"},
}
ALL_INTENTS = sorted({i for s in INTENTS_BY_RECEIVER.values() for i in s})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Constraints(BaseModel):
    """What the receiver has to respect when it answers."""

    format: Literal["json"] = "json"
    max_revisions: int = Field(default=2, ge=0, le=5)
    timeout_seconds: int = Field(default=45, ge=1, le=300)


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    conversation_id: str
    step: int = Field(ge=0)
    sender: AgentName
    receiver: AgentName
    intent: str
    content: dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)
    in_reply_to: Optional[str] = None
    sent_at: str = Field(default_factory=_now)

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, value: str) -> str:
        if value not in ALL_INTENTS:
            raise ValueError(f"unknown intent '{value}', expected one of {ALL_INTENTS}")
        return value

    @model_validator(mode="after")
    def _receiver_accepts_intent(self):
        allowed = INTENTS_BY_RECEIVER.get(self.receiver, set())
        if self.intent not in allowed:
            raise ValueError(
                f"receiver '{self.receiver}' does not accept intent '{self.intent}' (accepts {sorted(allowed)})"
            )
        if self.sender == self.receiver:
            raise ValueError("an agent cannot send a message to itself")
        return self

    def reply(self, sender: AgentName, receiver: AgentName, intent: str, content: dict, step: int) -> "AgentMessage":
        """Build the answer to this message with the links filled in."""
        return AgentMessage(
            conversation_id=self.conversation_id,
            step=step,
            sender=sender,
            receiver=receiver,
            intent=intent,
            content=content,
            constraints=self.constraints,
            in_reply_to=self.message_id,
        )

    def short(self) -> str:
        """One line for the log."""
        return f"[{self.conversation_id} step {self.step}] {self.sender} -> {self.receiver} {self.intent}"


class MessageValidationError(ValueError):
    """Raised by the orchestrator when a message fails schema validation."""
