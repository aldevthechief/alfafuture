"""Agentic factory of candidate generators, v2 (generic prompt): GLM reads the company profile (deterministic brief
from stage-2 facts), writes a business-model card (`note_business_model`, own knowledge marked as hypotheses), walks
24 general channels of influence (news-channels checklist A1–H1 + company-side I1–I3), probes its own queries on the
news pool (`probe`: numbered, top events + how many of its top-13 are new vs earlier probes) and finishes by
selecting probes by number with a coverage report per channel — queries come from the probe log, so nothing unprobed
gets in. Ranking = hybrid of experiments/retrieval (e5-large-instruct + qwen3-emb-0.6b + BM25 lemmas, RRF over
methods × queries). Evaluated on judge-v2 labels of U_106783 (the agent never sees them). v1 prompt: prompts/factory_v1.md.
python -X utf8 experiments/generator_factory/factory.py --tag A [--fewshots] [--company U_106783] [--max-steps 40]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / 'experiments/retrieval', ROOT / 'experiments/judge_v2', ROOT / 'experiments/briefs'):
    sys.path.insert(0, str(p))
import evaluate as ev
from common.llm import ChatModels, PromptLibrary, ToolLoop
from new_generators import MODELS, bm25_scorer, rank
from run import deterministic, facts_for  # experiments/briefs/run.py

JV2 = ROOT / 'experiments/judge_v2/out'
PROMPTS = PromptLibrary(HERE / 'prompts')
QUOTA = 13  # events per generator at budget ~200 with ~16 generators


class GeneratorSpec(BaseModel):
    reason: str = Field(description='короткое имя повода латиницей, например buyer_capex')
    fact: str = Field(description='факт профиля, на котором держится повод')
    instruction: str = Field(description='инструкция для энкодера: «Найди новости о …»')
    queries: list[str] = Field(min_length=1, max_length=3, description='1–3 запроса в стиле новостей о событиях')


class Pick(BaseModel):
    probe_id: int = Field(description='номер пробы из ответа probe')
    channel: str = Field(description='id канала, например A2')
    reason: str = Field(description='короткое имя повода латиницей')
    fact: str = Field(description='факт профиля, через который канал касается компании')


class Coverage(BaseModel):
    channel: str
    probe_ids: list[int] = []
    why_none: str = Field('', description='почему генератора нет: нет факта / в пуле новостей нет')


class Finish(BaseModel):
    """Selected probes and coverage of every channel."""
    model_config = ConfigDict(title='finish')
    selected: list[Pick]
    coverage: list[Coverage]


class Hybrid:
    """Same ranking as experiments/retrieval/new_generators.py, encoders kept in memory for interactive probes."""

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        _, _, self.event = ev.corpus()
        evs = pd.read_parquet(ev.DATA / 'news' / 'events.parquet')
        self.real, self.meta = set(evs.event), evs.set_index('event')
        self.D = {s: np.load(ev.HERE / 'out' / f'{s}.npz')['doc'].astype(np.float32) for s in MODELS}
        self.enc = {s: SentenceTransformer(name, device='cuda', model_kwargs={'torch_dtype': torch.float16},
                                           tokenizer_kwargs={'padding_side': pad} if pad else None)
                    for s, (name, _, pad) in MODELS.items()}
        for m in self.enc.values():
            m.max_seq_length = 512
        self.bm = bm25_scorer()
        self.probes: list[dict] = []
        self.card: dict = {}

    def events(self, instruction: str, queries: list[str], k: int = 300) -> list[int]:
        score = 0
        for q in queries:
            for s, (_, sep, _) in MODELS.items():
                v = self.enc[s].encode([f'Instruct: {instruction}\n{sep}{q}'], normalize_embeddings=True)[0].astype(np.float32)
                score = score + 1 / (60 + rank(self.D[s] @ v))
            score = score + 1 / (60 + rank(self.bm(q)))
        s = pd.Series(score).groupby(self.event).max().sort_values(ascending=False)
        return [int(e) for e in s.index if e in self.real][:k]

    def probe(self, instruction: str, queries: list[str]) -> str:
        """Check a generator: its number, top-10 events (date, title) and how many of its top-13 earlier probes did not find."""
        queries = queries[:3]
        top = self.events(instruction, queries, QUOTA)
        seen = set().union(*(p['top'] for p in self.probes)) if self.probes else set()
        pid = len(self.probes)
        self.probes.append({'probe_id': pid, 'instruction': instruction, 'queries': queries, 'top': top})
        lines = [f"{str(self.meta.first_date[e])[:10]} | {self.meta.title[e][:110]}" for e in top[:10]]
        return f'probe_id={pid}\n' + '\n'.join(lines) + f'\nновых событий в топ-{QUOTA} относительно прежних проб: {len(set(top) - seen)}'

    def note_business_model(self, products: str, buyers: str, costs_and_suppliers: str, regulation: str,
                            ownership_and_group: str, hypotheses: str) -> str:
        """Write down the company's business model: facts from the profile + industry knowledge (the latter listed in hypotheses)."""
        self.card = dict(products=products, buyers=buyers, costs_and_suppliers=costs_and_suppliers, regulation=regulation,
                         ownership_and_group=ownership_and_group, hypotheses=hypotheses)
        return 'записано'


def quotas(tops: dict[str, list[int]], n: int) -> set[int]:
    """Equal-weight quota union at the smallest budget giving >= n events."""
    for b in range(n, 20 * n, 5):
        h = max(1, round(b / len(tops)))
        chosen = set().union(*(t[:h] for t in tops.values()))
        if len(chosen) >= n:
            return chosen


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', default='U_106783')
    ap.add_argument('--tag', required=True, help='run name: outputs out/factory_<company>_<tag>*.json')
    ap.add_argument('--fewshots', action='store_true', help='add prompts/factory_fewshots.md (other industries)')
    ap.add_argument('--max-steps', type=int, default=40)
    a = ap.parse_args()
    u = a.company
    out = HERE / 'out'; out.mkdir(exist_ok=True)
    path = out / f'factory_{u}_{a.tag}.json'
    hy = Hybrid()
    if path.exists():  # rerun of the evaluation only
        run = json.loads(path.read_text(encoding='utf-8'))
    else:
        fs = PROMPTS.text('factory_fewshots') if a.fewshots else ''
        system, task = (m.content for m in PROMPTS.chat('factory').format_messages(brief=deterministic(facts_for(u)), fewshots=fs))
        tools = [StructuredTool.from_function(hy.probe), StructuredTool.from_function(hy.note_business_model)]
        res = ToolLoop(ChatModels().agent(temperature=0.2), tools, Finish, system, a.max_steps,
                       nudge='Проверь генераторы через probe или вызови finish.').run(task)
        if res is None:
            sys.exit(f'no finish within {a.max_steps} steps; probes made: {len(hy.probes)}')
        bad = [p.probe_id for p in res.selected if not 0 <= p.probe_id < len(hy.probes)]
        run = {'company': u, 'tag': a.tag, 'fewshots': a.fewshots, 'card': hy.card, 'probes': hy.probes, 'bad_probe_ids': bad,
               'selected': [p.model_dump() for p in res.selected if p.probe_id not in bad],
               'coverage': [c.model_dump() for c in res.coverage]}
        path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')
    probes, sel = run['probes'], run['selected']
    sel_ids = list(dict.fromkeys(p['probe_id'] for p in sel))
    print(f"{len(probes)} probes, {len(sel_ids)} selected, invalid probe ids {run['bad_probe_ids']}, card {'yes' if run['card'] else 'NO'}")
    for c in run['coverage']:
        print(f"  {c['channel']:3} probes {c['probe_ids']} {c['why_none'][:90]}")
    for p in sel:
        pr = probes[p['probe_id']]
        print(f"  #{p['probe_id']:2} {p['channel']:3} {p['reason'][:22]:22} | {' ‖ '.join(pr['queries'])[:140]}")

    tops = {f"sel:{i}": hy.events(probes[i]['instruction'], probes[i]['queries']) for i in sel_ids}
    drop = {f"drop:{p['probe_id']}": p['top'] for p in probes if p['probe_id'] not in sel_ids}
    lf = JV2 / f'labels_v2_{u}.jsonl'
    L = pd.DataFrame(json.loads(l)['value'] for l in open(lf, encoding='utf-8')) if lf.exists() else         pd.DataFrame(columns=['event', 'input', 'refusal', 'insight_potential', 'significance', 'mechanism_strength'])
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    P = set(L.index[(L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)])
    R = set(L.index[L.mechanism_strength >= 2])
    fin = set().union(*(t[:QUOTA] for t in tops.values()))
    dropped = set().union(*drop.values()) if drop else set()
    for name, s in ((f'selected ({len(sel_ids)} × {QUOTA})', fin), ('selected @200', quotas(tops, 200)),
                    ('selected + dropped probes', fin | dropped)):
        lab = s & set(L.index)
        print(f'{name:28} events {len(s):3} labelled {len(lab):3} pos {len(s & P):2}/{len(P)} rel {len(s & R):2}/{len(R)} '
              f'pos per labelled {len(s & P) / max(1, len(lab)):.2f}')
    todo = sorted((fin | dropped) - set(L.index))
    (out / f'factory_{u}_{a.tag}_todo.json').write_text(json.dumps(todo))
    (out / f'factory_{u}_{a.tag}_tops.json').write_text(json.dumps({k: v[:QUOTA] for k, v in (tops | drop).items()}))
    print(f'{len(todo)} unlabelled events -> out/factory_{u}_{a.tag}_todo.json')
