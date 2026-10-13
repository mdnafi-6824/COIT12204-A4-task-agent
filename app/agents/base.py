"""The bit both agents share: a name, the intents they accept, and logging
of every message they receive and send. The orchestrator only ever calls
handle(), so the agents never talk to each other directly."""

from typing import Optional

from app.logging_config import get_logger
from app.messages import INTENTS_BY_RECEIVER, AgentMessage

log = get_logger("agents")


class AgentError(RuntimeError):
    """An agent could not handle a message. The orchestrator turns this into
    an agent_error message and decides what to do next."""


class BaseAgent:
    name: str = "base"

    def __init__(self):
        self.received = 0
        self.sent = 0

    @property
    def accepts(self) -> set[str]:
        return INTENTS_BY_RECEIVER.get(self.name, set())

    def handle(self, message: AgentMessage) -> AgentMessage:
        """Validate that the message is for me, log it, do the work, log the reply."""
        if message.receiver != self.name:
            raise AgentError(f"{self.name} received a message addressed to {message.receiver}")
        if message.intent not in self.accepts:
            raise AgentError(f"{self.name} does not accept intent '{message.intent}'")
        self.received += 1
        log.info("MSG RECEIVED by %s %s", self.name, message.short())
        reply = self.process(message)
        self.sent += 1
        log.info("MSG SENT by %s %s", self.name, reply.short())
        return reply

    def process(self, message: AgentMessage) -> AgentMessage:  # pragma: no cover - overridden
        raise NotImplementedError

    def fallback(self, message: AgentMessage, reason: str) -> Optional[AgentMessage]:
        """What the orchestrator asks for when I time out or raise. None means
        the orchestrator has to handle it on its own."""
        return None
