"""My stand in for ChatOpenAI.

FakeChatModel hands back a fixed list of AIMessages in order. So in a test I
can say exactly what "the model" does: ask for a tool, answer with JSON,
answer with rubbish, or blow up. Because it subclasses BaseChatModel it plugs
into the same chain (prompt | llm.bind_tools(...)) as the real model, which
means the agent code I'm testing is the real agent code."""

import json
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    responses: list[AIMessage]
    raise_error: Optional[Exception] = None
    calls: list[list[BaseMessage]] = []

    def __init__(self, responses=None, raise_error=None, **kwargs):
        super().__init__(responses=list(responses or []), raise_error=raise_error, **kwargs)
        self.calls = []  # I record every call so tests can check what the model was sent

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools, **kwargs):
        # the agent calls this at start up, I just ignore the tools
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.raise_error is not None:
            raise self.raise_error
        if not self.responses:
            raise AssertionError("FakeChatModel ran out of scripted responses")
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------------------------------------------------------------------------
# Little helpers so the tests read nicely
# ---------------------------------------------------------------------------
def json_answer(**fields) -> AIMessage:
    """An AIMessage whose content is a JSON AgentResult."""
    return AIMessage(content=json.dumps(fields))


def tool_request(name: str, call_id: str = "call_1", **args) -> AIMessage:
    """An AIMessage that asks the agent to run a tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def text_answer(text: str) -> AIMessage:
    return AIMessage(content=text)
