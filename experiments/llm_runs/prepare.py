"""Freeze the inputs of the LLM runs that were written but never run, and estimate what each run costs, before
spending the 500 ₽/day quota. No LLM calls here. Writes data/llm_runs/ (gitignored: case data):
  companies.json   run set: 9 golden (stage 2) + 10 large (>=15 bn) + 11 companies of the real (company -> news) pairs
  facts.jsonl      deterministic stage-2 facts per company
  shortlists.jsonl facet_retrieval top-K news per company (lexical; the dense generators need FRIDA embeddings, absent)
  queue_*.jsonl    one line per LLM call to make: OCR pages, news for consequences, events to summarize, ranker packs ...
  manifest.json    per job: model, calls, estimated tokens and ₽ (routerai list price; real billing was 1.6-5x list)
python -X utf8 experiments/llm_runs/prepare.py [--top 100]
"""
import argparse, json, math, sys
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.config import SETTINGS
from common.data import DATA
from common.news_corpus import load_pairs
from common.search import corpus_fingerprint, news_pool
from stage4_news_relevance.plausibility.events import ensure_events
from common.llm.quota import prices
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.facet_retrieval.candidates import candidates, facets
from stage4_news_relevance.batch_ranker.ranker import lead

OUT = DATA / 'llm_runs'
GOLD = ROOT / 'stage2_company_facts' / 'golden_set'
PAIRS = ROOT / 'stage3_news_search' / 'golden_set' / 'real_pairs.json'
TOK = 3.0  # shortcut: chars per token for Russian, rough; replace by measured per-job averages from data/logs/calls.jsonl
SOURCE_FINGERPRINT = None


def prompt_chars(*rel: str) -> int:
    return sum(len((ROOT / r).read_text(encoding='utf-8')) for r in rel)


def ids_of(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding='utf-8')))  # {company id: pick info}


def dump(name: str, rows: list) -> None:
    if SOURCE_FINGERPRINT is None:
        raise ValueError('Freeze the source news corpus before preparing queues')
    with open(OUT / name, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(dict(r, corpus_fingerprint=SOURCE_FINGERPRINT), ensure_ascii=False, default=str) + '\n' for r in rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=100, help='shortlist size per company')
    a = ap.parse_args()
    source_news = pd.read_parquet(DATA / 'news' / 'news.parquet')
    SOURCE_FINGERPRINT = corpus_fingerprint(source_news)
    OUT.mkdir(parents=True, exist_ok=True)

    sets = {'golden9': ids_of(GOLD / 'companies.json'), 'large10': ids_of(GOLD / 'companies_large.json'),
            'real_pairs': list(json.loads(PAIRS.read_text(encoding='utf-8'))['pairs'])}
    companies = {}
    for name, ids in sets.items():
        for u in ids:
            companies.setdefault(u, []).append(name)
    assert all(u and u.startswith('U_') for u in companies), companies
    (OUT / 'companies.json').write_text(json.dumps(companies, ensure_ascii=False, indent=1), encoding='utf-8')
    print({k: len(v) for k, v in sets.items()}, 'unique', len(companies))

    facts = build(list(companies))
    dump('facts.jsonl', [{'id': u, 'facts': facts[u]} for u in companies])

    pairs = load_pairs(PAIRS, source_news)
    shortlists, news_ids, recall = [], set(), []
    for u in companies:
        res, fs = candidates(u, top=a.top)
        ids = res['id'].astype(int).tolist()
        news_ids.update(ids)
        shortlists.append({'id': u, 'facets': [list(f) for f in fs], 'news': ids, 'facet_of': res['facet'].tolist()})
        if u in pairs:
            recall.append((u, len(set(pairs[u]) & set(ids)), len(pairs[u])))
    dump('shortlists.jsonl', shortlists)
    hit, tot = sum(h for _, h, _ in recall), sum(t for *_, t in recall)
    print(f'shortlists: {len(shortlists)} x top {a.top}, {len(news_ids)} unique news; real pairs in shortlist {hit}/{tot}')

    news = pd.read_parquet(DATA / 'news' / 'news.parquet')
    news = news.assign(id=news['id'].astype(int)).set_index('id', drop=False)
    events = ensure_events(news_pool())
    event_of = {int(i): int(e) for e, ids in zip(events.event, events.news_ids) for i in ids}

    # --- queues: one row per LLM call ---
    pdfs = sorted((DATA / 'analytics').glob('*.pdf'))
    q_ocr = [{'pdf': p.name, 'page': n} for p in pdfs for n in range(1, len(pymupdf.open(p)) + 1)]
    q_cons = [{'news_id': i, 'chars': len(news.at[i, 'title']) + min(len(news.at[i, 'text']), 3000)} for i in sorted(news_ids)]
    ev_todo = sorted({event_of[i] for i in news_ids if i in event_of} & set(events.event[events.n_reports >= 5]))
    q_sum = [{'event': e, 'chars': 80 * int(events.set_index('event').at[e, 'n_reports'])} for e in ev_todo]
    q_brief = [{'id': u, 'chars': sum(len(f['text']) for f in facts[u] if f['depth'] == 1)} for u in companies]
    q_rank = [{'id': s['id'], 'pack': k, 'news': s['news'][k * 25:(k + 1) * 25],
               'chars': sum(len(news.at[i, 'title']) + len(lead(news.loc[i].to_dict())) for i in s['news'][k * 25:(k + 1) * 25])}
              for s in shortlists for k in range(math.ceil(len(s['news']) / 25))]
    q_hyde = [{'id': u, 'facet': k, 'text': t} for u in companies for k, t, _ in facets(u)]
    for name, q in [('ocr', q_ocr), ('consequences', q_cons), ('event_summary', q_sum), ('company_brief', q_brief),
                    ('batch_ranker', q_rank), ('hyde', q_hyde)]:
        dump(f'queue_{name}.jsonl', q)

    # --- estimates: (job, model, calls, tokens in, tokens out) ---
    S, n_co = SETTINGS, len(companies)
    jobs = [
        ('ocr', S.vision_model, len(q_ocr), len(q_ocr) * (1600 + prompt_chars('stage5_insight_detection/bank_analytics/prompts/ocr_page.md') / TOK), len(q_ocr) * 1200),
        ('consequences', S.cheap_model, len(q_cons), sum(r['chars'] for r in q_cons) / TOK + len(q_cons) * (prompt_chars('stage4_news_relevance/consequence_extraction/prompts/consequences.md') / TOK + 400), len(q_cons) * 300),
        ('event_summary', S.cheap_model, len(q_sum), sum(r['chars'] for r in q_sum) / TOK + len(q_sum) * 500, len(q_sum) * 150),
        ('company_brief', S.cheap_model, len(q_brief), sum(r['chars'] for r in q_brief) / TOK + n_co * 300, n_co * 400),
        ('batch_ranker', S.cheap_model, len(q_rank), sum(r['chars'] for r in q_rank) / TOK + len(q_rank) * (1600 / TOK + 400 + 500), len(q_rank) * 25 * 30),
        ('hyde', S.cheap_model, len(q_hyde), len(q_hyde) * 300, len(q_hyde) * 250),
        ('relevance (pipeline, top 30)', S.cheap_model, n_co * 30, n_co * 30 * 1500, n_co * 30 * 150),
        ('research_agent (~15 steps)', S.agent_model, n_co * 15, n_co * 15 * 6000, n_co * 15 * 400),
        ('insight_judge (5 x ~8 steps)', S.agent_model, n_co * 40, n_co * 40 * 5000, n_co * 40 * 300),
        ('insight_writer (~2 x write+check)', S.agent_model, n_co * 4, n_co * 4 * 6000, n_co * 4 * 1200),
    ]
    price = {m: (p_in, p_out) for m, _, _, p_in, p_out in prices(sorted({j[1] for j in jobs}))}
    manifest = []
    for job, model, calls, tin, tout in jobs:
        p_in, p_out = price.get(model) or (None, None)
        rub = (tin * p_in + tout * p_out) / 1e6 if p_in is not None else None
        manifest.append({'job': job, 'model': model, 'calls': calls, 'tokens_in': int(tin), 'tokens_out': int(tout),
                         'rub_list': round(rub, 2) if rub is not None else None})
    if SOURCE_FINGERPRINT != corpus_fingerprint(pd.read_parquet(DATA / 'news' / 'news.parquet')):
        raise ValueError('News changed during queue preparation; regenerate queues before use')
    (OUT / 'manifest.json').write_text(json.dumps({'corpus_fingerprint': SOURCE_FINGERPRINT,
                                                   'companies': n_co, 'top': a.top, 'real_pairs_in_shortlist': [hit, tot],
                                                   'jobs': manifest}, ensure_ascii=False, indent=1), encoding='utf-8')
    df = pd.DataFrame(manifest)
    df['rub_x1.6..5'] = df.rub_list.map(lambda r: f'{1.6 * r:.1f}–{5 * r:.1f}' if r is not None else '-')
    print(df.to_string(index=False))
    print(f'total list {df.rub_list.sum():.1f} ₽, billed ~{1.6 * df.rub_list.sum():.0f}–{5 * df.rub_list.sum():.0f} ₽')
