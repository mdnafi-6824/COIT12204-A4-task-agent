"""My prompt templates.

I kept three versions of the system prompt on purpose. v1 is my first naive
attempt, v2 and v3 are the refinements I made after evaluating the agent.
The evaluation script can run all three on the same cases, so the prompt
refinements in my report are backed by real runs and not just my opinion.

One thing that caught me out: the JSON format instructions have curly braces
in them, and ChatPromptTemplate treats braces as placeholders. So I pass the
instructions in as a variable instead of pasting them into the template."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.models import AgentResult

PARSER = PydanticOutputParser(pydantic_object=AgentResult)

SYSTEM_PROMPTS: dict[str, str] = {
    # v1: bare minimum. No rules about tools, dates or the output shape apart
    # from the schema. This is my baseline in the evaluation.
    "v1": (
        "You are a helpful assistant for a task manager. "
        "Answer the user's request about their tasks.\n\n"
        "{format_instructions}"
    ),
    # v2: after the first round of evaluation. I added clear tool rules and
    # insisted on JSON only, because v1 kept answering in prose or guessing
    # task details it never looked up.
    "v2": (
        "You are the reasoning engine inside a task manager application. "
        "You can call tools to look up real task data. Never guess task details: "
        "if the request mentions a task id, call lookup_task first. For a weekly "
        "summary call list_open_tasks and summarise_tasks before answering.\n\n"
        "Today's date is {today}.\n\n"
        "When you have enough information, reply with ONLY a JSON object and no "
        "other text, no markdown fences.\n\n"
        "{format_instructions}"
    ),
    # v3: after the second round. I added a priority rubric, date rules and
    # confidence guidance, because v2 gave different priorities for the same
    # task on repeat runs and sometimes suggested dates in the past.
    "v3": (
        "You are the reasoning engine inside a task manager application. "
        "You can call tools to look up real task data. Never guess task details: "
        "if the request mentions a task id, call lookup_task first. For a weekly "
        "summary call list_open_tasks and summarise_tasks before answering. "
        "If a tool reports an error, do not invent data; explain the problem in "
        "the reasoning field, set confidence to 0.0 and leave the other optional "
        "fields null.\n\n"
        "Today's date is {today}. Use ISO format (YYYY-MM-DD) for every date.\n\n"
        "Priority rubric (apply it consistently):\n"
        "- high: overdue, due within 3 days, or blocking other people or systems\n"
        "- medium: due within 4 to 14 days, or clearly important but not urgent\n"
        "- low: no deadline pressure, housekeeping, or nobody is waiting on it\n\n"
        "Due date rules: a suggested due date must never be earlier than today. "
        "If a task is already overdue, suggest the earliest realistic date from "
        "today onward (usually today or tomorrow) and say it was overdue. If the "
        "task already has a sensible due date, you may keep it.\n\n"
        "Confidence: 0.9 or above only when the tool data clearly supports the "
        "answer; 0.5 to 0.8 when you had to interpret; below 0.5 when guessing.\n\n"
        "Keep reasoning to one or two sentences. Reply with ONLY a JSON object "
        "and no other text, no markdown fences.\n\n"
        "{format_instructions}"
    ),
}

DEFAULT_VERSION = "v3"


def build_prompt(version: str = DEFAULT_VERSION) -> ChatPromptTemplate:
    if version not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt version '{version}'. Choose from {list(SYSTEM_PROMPTS)}.")
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPTS[version]),
            MessagesPlaceholder("messages"),
        ]
    ).partial(format_instructions=PARSER.get_format_instructions())


# The human message for each action. I keep these short. The system prompt
# has the rules, the human message just has the request.
HUMAN_TEMPLATES: dict[str, str] = {
    "classify_priority": (
        "Classify the priority of this task.\n{task_block}\n"
        "Set action to classify_priority and fill in priority, reasoning and confidence."
    ),
    "suggest_due_date": (
        "Suggest a realistic due date for this task.\n{task_block}\n"
        "Set action to suggest_due_date and fill in suggested_due_date, reasoning and confidence."
    ),
    "weekly_summary": (
        "Write a short summary of what is due in the next {days_ahead} days, including "
        "anything overdue. Mention the most urgent items by title. "
        "Set action to weekly_summary and fill in summary, reasoning and confidence."
    ),
}
