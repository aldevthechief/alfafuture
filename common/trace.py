"""Agent step journal: one JSONL file per run in data/traces/<run_id>.jsonl, one line per pipeline step.

    {"ts": "...", "run_id": "...", "step": "bank_match", "data": {...}}

Steps of the code are logged as well as model calls, so one file answers three questions: how the agent came to the
conclusion (timeline in the UI), how much the run cost against the 500 ₽/day limit (sum of cost_rub) and what to debug.
Append only, file mode 0600, API keys are masked before writing. Idea after FinRobot (finrobot/audit/transcript.py).
"""
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.data import DATA

SECRET = re.compile(r'(?i)(api[_-]?key|token|secret|authorization|bearer|password)(["\']?\s*[:=]\s*["\']?|\s+)([A-Za-z0-9_\-.]{8,})')
LONG_KEY = re.compile(r'\b(sk|pk|lf)-[A-Za-z0-9_\-]{12,}\b')


def redact(text: str) -> str:
    return LONG_KEY.sub(r'\1-***', SECRET.sub(lambda m: f'{m.group(1)}{m.group(2)}***', text))


class AgentTrace:
    def __init__(self, run_id: str | None = None, folder: Path | None = DATA / 'traces'):
        self.run_id = run_id or f'{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
        self.events: list[dict] = []
        self.path = Path(folder) / f'{self.run_id}.jsonl' if folder else None  # folder=None: memory only (tests)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, **data: Any) -> dict:
        event = {'ts': datetime.now(timezone.utc).isoformat(timespec='milliseconds'), 'run_id': self.run_id, 'step': step,
                 'data': data}
        self.events.append(event)
        if self.path:
            line = redact(json.dumps(event, ensure_ascii=False, default=str))
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        return event

    def timed(self, step: str):
        """with trace.timed('assess') as rec: ...; rec['out'] = 3   -> one event with ms."""
        return _Timed(self, step)

    def steps(self, step: str | None = None) -> list[dict]:
        return [e for e in self.events if step is None or e['step'] == step]

    def cost_rub(self) -> float:
        return round(sum(float(e['data'].get('cost_rub') or 0) for e in self.events), 4)

    @staticmethod
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


class _Timed:
    def __init__(self, trace: AgentTrace, step: str):
        self.trace, self.step, self.data = trace, step, {}

    def __enter__(self) -> dict:
        self.t0 = time.perf_counter()
        return self.data

    def __exit__(self, exc_type, exc, tb):
        self.data['ms'] = round((time.perf_counter() - self.t0) * 1000)
        if exc is not None:
            self.data['error'] = f'{exc_type.__name__}: {exc}'
        self.trace.log(self.step, **self.data)
        return False
