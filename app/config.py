"""My settings file. I read everything from environment variables (or a .env
file) so I never have to put the API key in the code."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Which model provider to use. I started with OpenAI but I also support
# Anthropic, because the two SDKs plug into LangChain the same way.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
MODEL_NAME = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
API_KEY = ANTHROPIC_API_KEY if LLM_PROVIDER == "anthropic" else OPENAI_API_KEY

# Which system prompt the agent uses. I kept v1 and v2 around so I can compare
# them in the evaluation. v3 is the one I ended up with.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

TASKS_FILE = Path(os.getenv("TASKS_FILE", BASE_DIR / "data" / "tasks.json"))
STATE_FILE = Path(os.getenv("STATE_FILE", BASE_DIR / "data" / "agent_state.json"))
LOG_FILE = Path(os.getenv("LOG_FILE", BASE_DIR / "logs" / "agent.log"))

# Two agent system (Assessment 4)
REVIEWER_USE_LLM = os.getenv("REVIEWER_USE_LLM", "true").lower() in ("1", "true", "yes")
REVIEWER_MAX_DAYS_AHEAD = int(os.getenv("REVIEWER_MAX_DAYS_AHEAD", "60"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "45"))
CONVERSATIONS_FILE = Path(os.getenv("CONVERSATIONS_FILE", BASE_DIR / "data" / "conversations.json"))

# Limits I use as guardrails
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "500"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "50"))
