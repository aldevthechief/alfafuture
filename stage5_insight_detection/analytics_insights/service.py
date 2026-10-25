"""Insights from the bank's analytics as a part of the pipeline: the cache, the run, the selection next to the news insights.

- One cached `AnalysisRun` (contract v2, stage6 insight_v2.py) per (company, role) in `cache/` (committed: the demo companies are served
  from it without a model call). Its `key` is a hash of the role, the company facts and the PDFs of the analytics folder: new facts or a
  new report make the cache stale and the agent runs again; `cache_only` serves a stale cache as it is (marked `stale_cache`) and never
  calls a model.
- `analytics_run` = cache, else the agent (`AnalyticsInsightAgent`, temperature 0), else None; an agent run that ended with an error is
  returned but not cached.
- `result_rows` turns the run into the rows of the pipeline's result list (same keys as the news results: `news_id` None, `status`,
  `insight`, `review`, `audit`; the run itself is one more row with status `analysis_run`).
- `select` keeps the analytics insights next to the news ones within a quota: at least 1 and at most `share` of all ready insights.
python -X utf8 stage5_insight_detection/analytics_insights/service.py --import-runs data/analytics_agent/runs.json   # put the insights of a run into the cache
"""
import argparse, hashlib, json, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.analytics_insights.agent import AnalyticsCandidate, AnalyticsInsightAgent
from stage5_insight_detection.analytics_insights.to_insight import candidate_to_insight, run_to_analysis
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
from stage6_insight_generation.insight_writer import insight_v2 as v2

CACHE = Path(__file__).with_name('cache')
LEVEL = {'low': 0, 'mid': 1, 'high': 2}
CONF = {'low': 0, 'medium': 1, 'high': 2}


def cache_key(company_id: str, role: str, facts: list[dict], pdf_dir: Path = PDF_DIR) -> str:
    h = hashlib.sha1(f'v2|{company_id}|{role}'.encode())
    for f in facts:
        h.update(f'|{f["cat"]}:{f["text"]}'.encode())
    for pdf in sorted(Path(pdf_dir).glob('*.pdf')):
        h.update(f'|{pdf.name}:{pdf.stat().st_size}'.encode())
    return h.hexdigest()[:16]


def cache_file(company_id: str, role: str, folder: Path = CACHE) -> Path:
    return Path(folder) / f'{company_id}_{role}.json'


def load_cache(company_id: str, role: str, key: str, folder: Path = CACHE, strict: bool = True) -> v2.AnalysisRun | None:
    """The cached run; with strict=False also a stale one (its `stale_cache` is set)."""
    path = cache_file(company_id, role, folder)
    if not path.exists():
        return None
    run = v2.AnalysisRun.model_validate_json(path.read_text(encoding='utf-8'))
    run.cached, run.stale_cache = True, run.key != key
    return None if strict and run.stale_cache else run


def save_cache(run: v2.AnalysisRun, folder: Path = CACHE) -> Path:
    path = cache_file(run.company_id, run.role, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    from common.news_corpus import _atomic
    _atomic(path, lambda target: target.write_text(run.model_copy(update={'cached': False, 'stale_cache': False}).model_dump_json(indent=1), encoding='utf-8'))
    return path


def analytics_run(company_id: str, role: str, facts: list[dict], *, models=None, bank=None, cache_only: bool = False, refresh: bool = False,
                  cache_dir: Path = CACHE, today: str | None = None, pdf_dir: Path = PDF_DIR, agent_factory=None,
                  strict: bool = False) -> v2.AnalysisRun | None:
    # MVP default: the cache wins even when its key is stale (new facts or reports), so the demo pairs never spend the
    # shared quota; the run is marked stale_cache. strict=True (pipeline --analytics-strict-cache) reruns a stale cache.
    key = cache_key(company_id, role, facts, pdf_dir)
    if not refresh:
        # the API keeps a cache per bank-report snapshot; a new snapshot falls back to the committed cache (same stale-wins rule)
        for folder in dict.fromkeys((Path(cache_dir), CACHE)):
            run = load_cache(company_id, role, key, folder, strict=strict)
            if run is not None:
                return run
    if cache_only:
        return None
    today = today or date.today().isoformat()
    agent = agent_factory() if agent_factory else AnalyticsInsightAgent(
        model=models.agent(temperature=0.0), vision=models.vision(max_tokens=8000), bank=bank)
    found = agent.run(company_id, facts, role, today)
    insights = [candidate_to_insight(c, company_id, role, agent.bank.rows, facts) for c in found]
    run = run_to_analysis(insights, company_id, role, today, agent.finish, model=getattr(agent.model, 'model_name', ''))
    run.key, run.error = key, agent.error or ''
    if insights and not agent.error:
        save_cache(run, cache_dir)  # only a complete run is cached
    return run


def result_rows(run: v2.AnalysisRun, insights: list[v2.Insight] | None = None) -> list[dict]:
    """Rows of the pipeline's result list: one per insight (same keys as a news result) + one for the run itself."""
    rows = [{'news_id': None, 'title': i.title, 'source': 'bank_analytics', 'status': 'ready', 'insight': i.model_dump(),
             'review': {'ok': True, 'issues': i.review.warnings}, 'audit': []} for i in (run.insights if insights is None else insights)]
    rows.append({'news_id': None, 'title': 'Аналитика Банка: итог прогона', 'source': 'bank_analytics', 'status': 'analysis_run',
                 'analysis': run.model_dump(exclude={'insights'}), 'selected_insights': len(rows)})
    return rows


def select(insights: list[v2.Insight], n_news_ready: int, share: float = 0.4) -> list[v2.Insight]:
    """The analytics insights shown next to `n_news_ready` news insights: at least 1, at most `share` of all (a ≤ share · (n + a)); best first
    by role fit, impact and confidence. Without news insights all of them."""
    ranked = sorted(insights, key=lambda i: (-(i.role_fit or 0), -LEVEL.get(i.impact.level if i.impact else 'low', 0), -CONF.get(i.confidence, 0)))
    cap = len(ranked) if n_news_ready == 0 else max(1, int(share / (1 - share) * n_news_ready))
    return ranked[:cap]


def import_runs(path: Path, cache_dir: Path = CACHE) -> list[Path]:
    """Put the candidates of a finished run (data/analytics_agent/runs.json, written by run.py) into the cache, mapped to contract v2 by the
    current code; the run is not repeated."""
    from stage2_company_facts.deterministic_facts.facts import build
    from stage5_insight_detection.bank_analytics.search import BankAnalytics
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    bank = BankAnalytics()
    facts = build([r['company'] for r in data['results']])
    out = []
    for r in data['results']:
        f = facts[r['company']]
        insights = [candidate_to_insight(AnalyticsCandidate.model_validate(c), r['company'], r['role'], bank.rows, f) for c in r['candidates']]
        from stage5_insight_detection.analytics_insights.agent import Finish
        run = run_to_analysis(insights, r['company'], r['role'], date.today().isoformat(), Finish.model_validate(r['finish']) if r.get('finish') else None,
                              model=data.get('model') or 'z-ai/glm-5.3-flash', trace_session=data.get('run', ''))
        run.key = cache_key(r['company'], r['role'], f)
        out.append(save_cache(run, cache_dir))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--import-runs', required=True, help='runs.json of run.py')
    ap.add_argument('--cache-dir', default=str(CACHE))
    a = ap.parse_args()
    for p in import_runs(Path(a.import_runs), Path(a.cache_dir)):
        print('cached', p)
