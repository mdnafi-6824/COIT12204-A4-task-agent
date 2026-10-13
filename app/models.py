"""All my Pydantic models in one place. The task itself, the JSON the agent
has to return, and the request and response bodies for the API. Keeping them
together means everything gets validated the same way."""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Priority = Literal["low", "medium", "high"]
TaskStatus = Literal["todo", "in_progress", "done"]
AgentAction = Literal["classify_priority", "suggest_due_date", "weekly_summary"]


# ---------------------------------------------------------------------------
# The task itself
# ---------------------------------------------------------------------------
class Task(BaseModel):
    id: int
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: Optional[Priority] = None
    due_date: Optional[date] = None
    status: TaskStatus = "todo"
    created_at: date


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: Optional[Priority] = None
    due_date: Optional[date] = None


# ---------------------------------------------------------------------------
# What the agent has to give back
# ---------------------------------------------------------------------------
class AgentResult(BaseModel):
    """This is the JSON object I ask the model for. PydanticOutputParser turns
    this class into the format instructions in the prompt and then checks the
    model's answer against it. The descriptions are read by the model too."""

    action: AgentAction = Field(description="The action that was performed.")
    task_id: Optional[int] = Field(
        default=None, description="The task the answer is about, if any."
    )
    priority: Optional[Priority] = Field(
        default=None, description="Priority for classify_priority. Otherwise null."
    )
    suggested_due_date: Optional[date] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) for suggest_due_date. Otherwise null.",
    )
    summary: Optional[str] = Field(
        default=None, description="Plain English summary for weekly_summary. Otherwise null."
    )
    reasoning: str = Field(description="One or two sentences explaining the decision.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident the agent is, from 0.0 to 1.0."
    )

    @field_validator("reasoning")
    @classmethod
    def _reasoning_not_blank(cls, value: str) -> str:
        # an empty reasoning string is useless so I reject it
        if not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()


# ---------------------------------------------------------------------------
# Request bodies for the API
# ---------------------------------------------------------------------------
class ClassifyPriorityRequest(BaseModel):
    task_id: Optional[int] = Field(default=None, ge=1)
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _need_id_or_title(self):
        # you can classify a saved task by id, or an unsaved one by title
        if self.task_id is None and not self.title:
            raise ValueError("Provide either task_id or a title/description.")
        return self


class SuggestDueDateRequest(BaseModel):
    task_id: int = Field(ge=1)


class WeeklySummaryRequest(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=30)


class OrchestratorRequest(BaseModel):
    """One request for the two agent system. Same fields as the single agent
    endpoints, with the action chosen explicitly."""

    action: AgentAction
    task_id: Optional[int] = Field(default=None, ge=1)
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    days_ahead: int = Field(default=7, ge=1, le=30)

    @model_validator(mode="after")
    def _fields_match_action(self):
        if self.action == "suggest_due_date" and self.task_id is None:
            raise ValueError("suggest_due_date needs a task_id.")
        if self.action == "classify_priority" and self.task_id is None and not self.title:
            raise ValueError("classify_priority needs a task_id or a title.")
        return self


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------
class StateSnapshot(BaseModel):
    run_count: int
    error_count: int
    fallback_count: int
    last_action: Optional[str]
    last_suggestion: Optional[dict]
    last_tool_call: Optional[dict]
    history_length: int
    updated_at: Optional[datetime]


class AgentResponse(BaseModel):
    request_id: str
    action: AgentAction
    result: AgentResult
    fallback_used: bool
    tools_called: list[str]
    prompt_version: str
    warnings: list[str] = []
    state: StateSnapshot


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
