"""Synthetic golden set for news retrieval: per (company, role) 10 news the reader must see + 10 he must not, written WITHOUT
looking at the retrieval code or its results (that is the point: no label is shaped by what a generator can find).
Inputs: depth-1 company facts (activity, scale, geo, profile, ownership, bank products) + the role from roles.json. Not used:
the profile's sensitivities / macro channels (the taxonomy of the candidate generators).
Steps: 1) three generation calls (positives; hard negatives — near in topic, immaterial for this company; easy negatives —
other worlds), over-generated; 2) a blind audit by ANOTHER model family on shuffled batches (no label, no reason shown);
3) keep the items the auditor agrees with: 10 positives, 5 hard + 5 easy negatives; 4) write golden/<company>_<role>.json.
python -X utf8 -m evals.retrieval_golden.build --pairs U_14366:cfo,U_3263:ceo [--gen-model M] [--audit-model M]"""
import argparse
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, PromptLibrary, structured
from common.roles import role_context

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
GOLDEN = Path(__file__).with_name('golden')
TODAY = date(2026, 10, 20)  # the last day of the news corpus
FACT_CATS = ('activity', 'scale', 'geo', 'profile', 'ownership', 'bank')
SKIP_FACTS = ('Крупный бизнес по критерию кейса', 'Есть признак ключевого собственника')
KINDS = {  # kind -> (label, task, why hint, generated, kept)
    'pos': (1, 'Напиши {n} новостей, о которых этому читателю СТОИТ узнать: каждая, будь она настоящей, существенно меняет '
               'деньги, риски или решения именно этой компании в ближайший квартал. Охвати РАЗНЫЕ механизмы влияния: стоимость '
               'денег и фондирование, валюта и внешняя торговля, цены на ресурсы и издержки, регулирование и налоги, спрос и '
               'клиенты, конкуренты, поставщики и логистика, рынок труда, санкции, технологии. Не все новости про банки и ставки.',
            'как именно новость меняет деньги или решения компании', 27, 10),
    'hard_neg': (0, 'Напиши {n} новостей, которые только ПОХОЖИ на нужные: говорят об отрасли, продукте, регионе или теме '
                    'компании, но для этой компании ничего не меняют — не тот сегмент, не та страна или регион, слишком малый '
                    'масштаб, давно известный факт, мнение без последствий, событие у соседней отрасли. Не делай их скучными.',
                 'почему именно для этой компании ничего не меняется, хотя тема близка', 8, 5),
    'easy_neg': (0, 'Напиши {n} новостей из совсем других областей — другие отрасли, регионы, общество, спорт, культура, наука, — '
                    'которые обычный деловой читатель сочтёт нерелевантными для этой компании. Это настоящие деловые или '
                    'общественные новости, не абсурд.', 'почему новость не касается компании', 8, 5),
}


class GenItem(BaseModel):
    title: str
    text: str = Field(min_length=300)
    days_ago: int = Field(ge=0, le=30)
    names_industry: bool
    why: str


class GenBatch(BaseModel):
    items: list[GenItem]


class Judgement(BaseModel):
    n: int
    must_read: bool
    reason: str


class AuditBatch(BaseModel):
    items: list[Judgement]


def brief(company: str) -> str:
    from stage2_company_facts.deterministic_facts.facts import build
    facts = build([company], use_profiles=True)[company]
    return '\n'.join('- ' + f['text'] for f in facts if f['depth'] == 1 and f['cat'] in FACT_CATS
                     and not f['text'].startswith(SKIP_FACTS))


def model(name: str | None, temperature: float):
    models = ChatModels()
    m = models._make(name, temperature, 64000) if name else models.agent(temperature=temperature)
    return m.model_copy(update={'reasoning_effort': 'low'})


def generate(gen, company: str, role: str, kind: str, chunk: int = 9) -> list[dict]:
    """`chunk` news per call (a long structured answer of 18+ items did not fit); later calls see the titles already written."""
    label, task, why, total, _ = KINDS[kind]
    ctx, out = role_context(role), []
    while len(out) < total:
        n = min(chunk, total - len(out))
        done = '\n'.join('- ' + i['title'] for i in out)
        prompt = PROMPTS.render('generate', brief=brief(company), task=task.format(n=n), why_hint=why, n=n, direct=round(n * 0.4),
                                today=TODAY.isoformat(), **ctx)
        if done:
            prompt += f'\n\nУже написаны (не повторяй сюжеты):\n{done}\n'
        for attempt in range(3):  # a malformed structured answer is random: one more draw usually fixes it
            try:
                batch = structured(gen, GenBatch).invoke(prompt + (f' (вариант {attempt})' if attempt else '')).items  # a changed prompt skips the cached bad answer
                break
            except Exception:
                if attempt == 2:
                    raise
        out += [dict(kind=kind, label=label, title=i.title.strip(), text=i.text.strip(), why=i.why.strip(),
                     names_industry=i.names_industry, date=(TODAY - timedelta(days=i.days_ago)).isoformat()) for i in batch]
    return out


def audit(aud, company: str, role: str, items: list[dict], seed: str, batch: int = 10) -> None:
    """Blind: shuffled batches, only title + text. Sets item['audit'] = auditor's must_read and reason."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    ctx = role_context(role)
    for start in range(0, len(order), batch):
        part = order[start:start + batch]
        text = '\n\n'.join(f'[{k}] {items[i]["title"]}\n{items[i]["text"]}' for k, i in enumerate(part))
        prompt = PROMPTS.render('audit', brief=brief(company), items=text, **ctx)
        for j in structured(aud, AuditBatch).invoke(prompt).items:
            if 0 <= j.n < len(part):
                items[part[j.n]]['audit'] = dict(must_read=j.must_read, reason=j.reason)


def select(items: list[dict], seed: str) -> list[dict]:
    """The items the auditor agrees with, up to the target per kind, balanced between news that name the company's industry
    and news that reach it through general mechanisms (a retriever must not win on the industry word alone), the lengths close to the
    median of the agreed pool; a kind that has fewer keeps what it has."""
    rng, chosen = random.Random(seed), []
    agreed = [i for i in items if i.get('audit', {}).get('must_read') == bool(i['label'])]
    target = float(np.median([len(i['text']) for i in agreed])) if agreed else 0  # equal lengths: length must not tell the label
    for kind, (label, _, _, _, keep) in KINDS.items():
        ok = sorted((i for i in agreed if i['kind'] == kind), key=lambda i: abs(len(i['text']) - target))
        direct, indirect = [i for i in ok if i['names_industry']], [i for i in ok if not i['names_industry']]
        picked = direct[:keep // 2] + indirect[:keep - keep // 2]
        picked += [i for i in ok if i not in picked][:keep - len(picked)]
        chosen += picked
    rng.shuffle(chosen)
    for n, i in enumerate(chosen, 1):
        i['gid'] = f'G{n:02d}'
    return chosen


def build_pair(company: str, role: str, gen, aud) -> dict:
    items = [i for kind in KINDS for i in generate(gen, company, role, kind)]
    audit(aud, company, role, items, f'{company}:{role}')
    kept = select(items, f'{company}:{role}:select')
    return dict(company=company, role=role, brief=brief(company), generated=len(items),
                agreed={k: sum(1 for i in items if i['kind'] == k and i.get('audit', {}).get('must_read') == bool(KINDS[k][0]))
                        for k in KINDS},
                items=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True, help='U_14366:cfo,U_3263:ceo')
    ap.add_argument('--gen-model', default=None, help='default: the agent model (GLM)')
    ap.add_argument('--audit-model', default='deepseek/deepseek-v4.1-flash', help='must be another family than the generator')
    a = ap.parse_args()
    gen, aud = model(a.gen_model, 0.8), model(a.audit_model, 0.0)
    GOLDEN.mkdir(exist_ok=True)
    for pair in a.pairs.split(','):
        company, role = pair.split(':')
        try:
            out = build_pair(company, role, gen, aud)
        except Exception as error:  # one failed pair must not lose the others: rerun it alone
            print(f'{company}_{role}: FAILED {type(error).__name__}: {str(error)[:160]}')
            continue
        path = GOLDEN / f'{company}_{role}.json'
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        kinds = {k: sum(i['kind'] == k for i in out['items']) for k in KINDS}
        print(f'{path.name}: generated {out["generated"]}, auditor agrees {out["agreed"]}, kept {kinds}')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GOLDEN.glob('U_*.json'))}
    (GOLDEN / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
