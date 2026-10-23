"""Live smoke test of the LLM layer through the proxy: plain call, structured output, ToolLoop with a tool.
Costs ~0.02 ₽; traces go to Langfuse as one session `smoke-<time>`. Answers are cached, so a rerun is free.
python -X utf8 common/llm/smoke.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from langchain_core.tools import tool
from pydantic import BaseModel

from common.llm import ChatModels, ToolLoop, structured
from common.llm.models import RUN_ID


class Answer(BaseModel):
    city: str
    country: str


@tool
def key_rate() -> str:
    """Current Bank of Russia key rate."""
    return '14%'


models = ChatModels()
print('agent:', models.agent().invoke('Ответь одним словом: столица Франции?').content)
a = structured(models.cheap(), Answer).invoke('Где находится Эйфелева башня? Город и страна.')
assert a.city and a.country, a
print('structured:', a)
loop = ToolLoop(models.agent(), [key_rate], Answer, 'Используй инструменты, затем вызови финальный инструмент.')
print('tool loop:', loop.run('Узнай ключевую ставку через инструмент, затем верни город и страну ЦБ РФ.'))
print('langfuse session:', RUN_ID)
