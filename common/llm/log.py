"""Local log of every LLM call and every agent tool call: data/logs/calls.jsonl, one line per call, gitignored (prompts
contain case data). Complements Langfuse: works offline, greppable, and is the audit trail of tool calls the case asks
for (security section). LLM line: run, model, cached, seconds, tokens, cost (₽, as billed by the proxy), full prompt
and answer (images replaced by a placeholder). Tool line: run, tool, args, output.
python -X utf8 common/llm/log.py [--day 2026-10-20]   # per-run summary: calls, cache hits, tokens, ₽, errors
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

RUN_ID = f'{Path(sys.argv[0]).stem or "python"}-{datetime.now():%Y%m%d-%H%M%S}'
LOG = DATA / 'logs' / 'calls.jsonl'


def write(record: dict, path: Path = LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'run': RUN_ID, 'ts': datetime.now().isoformat(timespec='seconds'), **record},
                           ensure_ascii=False, default=str) + '\n')


def _content(c: Any) -> Any:
    if isinstance(c, list):  # multimodal parts: keep text, drop base64 images
        return [p if not (isinstance(p, dict) and p.get('type') == 'image_url') else '<image>' for p in c]
    return c


def _message(m: BaseMessage) -> dict:
    d = {'role': m.type, 'content': _content(m.content)}
    if getattr(m, 'tool_calls', None):
        d['tool_calls'] = m.tool_calls
    if getattr(m, 'tool_call_id', None):
        d['tool_call_id'] = m.tool_call_id
    return d


class CallLog(BaseCallbackHandler):
    """Attached to every ChatModels model. A cache hit (FileLLMCache) is logged too, with cached=True and no cost."""

    def __init__(self, path: Path = LOG):
        self.path, self.started = path, {}

    def on_chat_model_start(self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, **kw) -> None:
        self.started[run_id] = (time.time(), kw.get('invocation_params', {}).get('model') or kw.get('invocation_params', {}).get('model_name'),
                                [_message(m) for m in messages[0]])

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw) -> None:
        t0, model, prompt = self.started.pop(run_id, (time.time(), None, None))
        gen = response.generations[0][0]
        msg = getattr(gen, 'message', None)
        usage = (response.llm_output or {}).get('token_usage') or {}
        write({'kind': 'llm', 'model': model, 'cached': response.llm_output is None, 'sec': round(time.time() - t0, 2),
               'in': usage.get('prompt_tokens'), 'out': usage.get('completion_tokens'), 'cost': usage.get('cost'),
               'prompt': prompt, 'answer': _message(msg) if msg else gen.text}, self.path)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
        t0, model, prompt = self.started.pop(run_id, (time.time(), None, None))
        write({'kind': 'llm', 'model': model, 'sec': round(time.time() - t0, 2), 'error': f'{type(error).__name__}: {error}',
               'prompt': prompt}, self.path)


def log_tool(name: str, args: dict, output: str) -> None:
    write({'kind': 'tool', 'tool': name, 'args': args, 'output': output})


def summary(day: str | None = None, path: Path = LOG):
    import pandas as pd
    df = pd.read_json(path, lines=True) if path.exists() else pd.DataFrame()
    if df.empty:
        return df
    if day:
        df = df[df.ts.astype(str).str.startswith(day)]
    for c in ['cached', 'in', 'out', 'cost', 'error', 'tool']:
        if c not in df:
            df[c] = None
    llm = df[df.kind == 'llm']
    return pd.DataFrame({
        'llm_calls': llm.groupby('run').size(), 'cached': llm.groupby('run').cached.sum(),
        'errors': llm.groupby('run').error.count(), 'tokens_in': llm.groupby('run')['in'].sum(),
        'tokens_out': llm.groupby('run').out.sum(), 'cost_rub': llm.groupby('run').cost.sum().round(4),
        'tool_calls': df[df.kind == 'tool'].groupby('run').size()}).fillna(0)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', help='YYYY-MM-DD (local time)')
    s = summary(ap.parse_args().day)
    print(s.to_string() if len(s) else f'no calls logged in {LOG}')
    if len(s):
        print(f"total: {int(s.llm_calls.sum())} LLM calls, {s.cost_rub.sum():.4f} ₽")
