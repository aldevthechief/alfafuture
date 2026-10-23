"""Opt-in bank-only run, with incremental saving and a reserve before each next thesis.

Before each insight the rest of the budget must cover the costliest insight so far (at least MIN_RESERVE ₽), otherwise
the run stops there. Every insight is saved as soon as it is made (JSON and card), and the summary (spent in all and per
company, every insight, the stop reason) is rewritten after each one, so a stop by the budget never loses a result.
No news run: the theses are not merged with news insights here (pipeline.py --bank-insights does that). The cost is the
proxy's usage.cost of each answer (digest.CostMeter); cached answers cost nothing.
python -X utf8 stage5_insight_detection/analyst_loop/bank_run.py U_3263 U_9250 U_1595 --budget 5 [--role cfo] [--out DIR]
"""
import argparse, json, math, re, sys
from collections.abc import Callable
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.analyst_loop.loop import AnalystLoop, bank_insights
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.theses import ShownLog
from stage6_insight_generation.insight_card.render import render
from common.news_corpus import bind_result, write_json
from stage2_company_facts.profile.integration import load_company
from stage5_insight_detection.analyst_loop.pool import EmptyNewsPool

MIN_RESERVE = 1.0  # ₽ kept for the next insight before the first one shows what an insight costs


def make_loop(company_id, library, models, *, trace=None):
    """All standalone bank runs use the same Kirill profile snapshot as pipeline.py."""
    from stage6_insight_generation.insight_writer.writer import InsightWriter
    stage2 = load_company(company_id)
    agent = models.agent()
    return (AnalystLoop(library, EmptyNewsPool(), agent, InsightWriter(agent), trace=trace,
                        company_facts=stage2.facts), CompanyProfile(**stage2.card))


class Stop(Exception):
    pass


def run(companies: list[str], role: str, make_loop: Callable[[str], tuple[AnalystLoop, CompanyProfile]],
        spent: Callable[[], float], budget: float, out: Path, shown: ShownLog, *, news=None, **theses) -> dict:
    """make_loop(company) -> the loop and the profile; spent() -> ₽ spent so far; theses: as_of / folder of
    loop.bank_insights. Returns the summary (also in out/)."""
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('budget must be finite and non-negative')
    out.mkdir(parents=True, exist_ok=True)
    news = pd.DataFrame(columns=['id', 'title', 'text', 'date', 'source', 'url']) if news is None else news
    summary = {'budget_rub': budget, 'spent_rub': 0.0, 'companies': {}, 'insights': [], 'stopped': ''}
    worst = MIN_RESERVE

    def save() -> None:
        summary['spent_rub'] = round(spent(), 3)
        write_json(out / 'summary.json', summary)

    for cid in companies:
        if spent() + worst > budget:
            summary['stopped'] = f'недостаточно резерва {worst:.2f} ₽ перед компанией {cid}'
            save()
            break
        loop, profile = make_loop(cid)
        start, run_thesis = spent(), loop.run_thesis

        def guarded(pick, prof, role, run_thesis=run_thesis, loop=loop, cid=cid):
            nonlocal worst
            if spent() + worst > budget:
                raise Stop(f'потрачено {spent():.2f} ₽ из {budget}, следующий инсайт может стоить до {worst:.2f} ₽')
            before, steps = spent(), len(loop.trace.events)
            res = run_thesis(pick, prof, role)
            worst = max(worst, spent() - before)
            res['cost_rub'] = round(spent() - before, 3)
            res = bind_result(res, news, allow_unresolved=True)
            name = re.sub(r'[<>:"/\\|?*]', '_', f'{cid}_{role}_bank_' + res['thesis']['id'].split(':', 1)[1])
            plain = json.loads(json.dumps(res, default=str))
            write_json(out / f'{name}.json', plain)
            (out / f'{name}.html').write_text(render(plain, loop.trace.events[steps:]), encoding='utf-8')
            summary['insights'].append({'company': cid, 'thesis': res['thesis']['id'], 'topic': res['thesis']['topic'],
                                        'status': res.get('status') or res['verdict']['status'], 'cost_rub': res['cost_rub'],
                                        'file': f'{name}.json'})
            save()
            return res

        loop.run_thesis = guarded
        try:
            bank_insights(loop, profile, role, [], shown, **theses)
        except Stop as e:
            summary['stopped'] = f'{cid}: {e}'
        finally:
            loop.run_thesis = run_thesis
        summary['companies'][cid] = round(spent() - start, 3)
        save()
        if summary['stopped']:
            break
    return summary


if __name__ == '__main__':
    from common.llm import ChatModels
    from common.trace import AgentTrace
    from stage5_insight_detection.bank_analytics.digest import CostMeter
    from stage5_insight_detection.bank_analytics.library import BankLibrary
    from stage6_insight_generation.insight_writer.writer import InsightWriter
    ap = argparse.ArgumentParser(description='bank insights for companies within a ₽ budget (needs LLM_API_KEY)')
    ap.add_argument('company_id', nargs='+')
    ap.add_argument('--role', default='cfo')
    ap.add_argument('--budget', type=float, required=True, help='reserve-based budget between insights, ₽')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'insights' / 'bank_run')
    a = ap.parse_args()
    if not math.isfinite(a.budget) or a.budget < 0:
        ap.error('--budget must be finite and non-negative')
    meter = CostMeter()
    models = ChatModels(callbacks=[meter])
    library = BankLibrary.load()

    def factory(cid: str):
        return make_loop(cid, library, models, trace=AgentTrace(f'bank-run-{cid}', folder=a.out))

    s = run(a.company_id, a.role, factory, lambda: meter.cost, a.budget, a.out, ShownLog(a.out / 'shown.json'))
    print(json.dumps(s, ensure_ascii=False, indent=1))
