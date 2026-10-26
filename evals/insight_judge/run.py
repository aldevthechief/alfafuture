"""Run the judge on the labelling task: every pair in both orders (+ every distinct card pointwise) -> a JSON run file that
calibrate.py compares with the human sheet. Cost: 2 calls per pair + 1 per card (GLM low, ~0.1 RUB each).
python -X utf8 -m evals.insight_judge.run --pairs <dir> [--limit N] [--pointwise] [--model z-ai/glm-5.3-flash] [--out run.json]"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import SETTINGS
from common.data import DATA
from common.llm import ChatModels
from evals.insight_judge.judge import InsightJudge
from evals.insight_judge.pairs import load_pairs

OUT = DATA / 'evals' / 'judge_runs'


def make_model(name: str | None):
    models = ChatModels()
    model = models._make(name, 0.0, 64000) if name else models.agent(temperature=0.0)
    return model.model_copy(update={'reasoning_effort': 'low'})  # a verdict of a few fields: no long reasoning needed


async def run(pairs, judge: InsightJudge, pointwise: bool, workers: int) -> dict:
    sem = asyncio.Semaphore(workers)

    async def limited(coro):
        async with sem:
            return await coro
    verdicts = await asyncio.gather(*(limited(judge.ajudge_pair(p.exposure, p.a, p.b)) for p in pairs))
    out = dict(pairs={p.id: v for p, v in zip(pairs, verdicts)})
    if pointwise:
        cards = sorted({(p.exposure, c) for p in pairs for c in (p.a, p.b)})
        scored = await asyncio.gather(*(limited(judge.ascore_card(e, c)) for e, c in cards))
        out['cards'] = {c: s.model_dump() for (e, c), s in zip(cards, scored)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True, type=Path)
    ap.add_argument('--limit', type=int, default=None, help='first N pairs (smoke test)')
    ap.add_argument('--pointwise', action='store_true')
    ap.add_argument('--model', default=None, help=f'default {SETTINGS.agent_model}; another model family checks self-preference')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    pairs = load_pairs(a.pairs)[:a.limit]
    result = asyncio.run(run(pairs, InsightJudge(make_model(a.model)), a.pointwise, a.workers))
    result['model'] = a.model or SETTINGS.agent_model
    out = a.out or OUT / f'{result["model"].replace("/", "_")}_{len(pairs)}pairs.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{out}: {len(pairs)} pairs' + (f', {len(result["cards"])} cards' if a.pointwise else ''))


if __name__ == '__main__':
    main()
