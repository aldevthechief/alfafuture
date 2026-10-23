"""Runtime settings from environment variables. One place to see every knob the pipeline reads.
Secrets (LLM_API_KEY, LANGFUSE_*) live in common/llm/.env (gitignored); real env vars win over the file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / 'llm' / '.env')


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None
    base_url: str
    agent_model: str  # complex multi-step reasoning: company research, insight judge, insight writer
    cheap_model: str  # bulk per-news calls: consequences, HyDE, relevance scoring
    vision_model: str  # OCR of the scanned bank analytics PDFs

    @classmethod
    def from_env(cls) -> 'Settings':
        return cls(
            llm_api_key=os.environ.get('LLM_API_KEY'),
            # organizers' LiteLLM proxy (OpenAI-compatible); model ids must match its /v1/models exactly
            base_url=os.environ.get('LLM_BASE_URL', 'https://changellenge.up.railway.app/v1'),
            agent_model=os.environ.get('AGENT_MODEL', 'z-ai/glm-5.3-flash'),
            cheap_model=os.environ.get('CHEAP_MODEL', 'google/gemma-4-26b-a4b-it'),
            vision_model=os.environ.get('VISION_MODEL', 'qwen/qwen3-vl-30b-a3b-instruct'),
        )


SETTINGS = Settings.from_env()
