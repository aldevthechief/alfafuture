"""Live run of the analytics insight agent for a few (company, role) pairs, in parallel, with a hard ₽ budget: every tool
call checks the cost this process has billed so far (data/logs/calls.jsonl lines of this run) and stops the loop above the
budget. Needs the user's approval of the exact run (CLAUDE.md). Results: data/analytics_agent/runs.json.
python -X utf8 stage5_insight_detection/analytics_insights/run.py U_14366:cfo U_57706:treasury U_177581:ceo [--budget 28]
"""
import argparse, json, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.config import SETTINGS
from common.llm import ChatModels
from common.llm.log import LOG, RUN_ID
from stage2_company_facts.deterministic_facts.facts import build
from stage5_insight_detection.analytics_insights.agent import AnalyticsInsightAgent
from stage5_insight_detection.analytics_insights.to_insight import candidate_to_insight
from stage5_insight_detection.bank_analytics.search import BankAnalytics


class Budget:
    def __init__(self, limit: float):
        self.limit, self.offset, self.lock = limit, LOG.stat().st_size if LOG.exists() else 0, threading.Lock()
        self.spent = 0.0

    def check(self) -> float:
        with self.lock:  # reads only what was appended since the last check
            with open(LOG, 'rb') as f:  # bytes: text mode counts CRLF lines short and the offset drifts into the middle of a letter
                f.seek(self.offset)
                for line in f:
                    if not line.endswith(b'\n'):
                        break  # half-written line: read it next time
                    self.offset += len(line)
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a line torn by another process writing the same log
                    if rec.get('run') == RUN_ID and rec.get('cost'):
                        self.spent += rec['cost']
            if self.spent > self.limit:
                raise RuntimeError(f'budget {self.limit} ₽ exceeded ({self.spent:.1f} ₽)')
            return self.spent


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='company:role')
    ap.add_argument('--budget', type=float, default=28.0)
    ap.add_argument('--max-steps', type=int, default=30)
    ap.add_argument('--reasoning', choices=['low', 'medium', 'high'], default=None, help="reasoning_effort of the agent model (default: the model's own)")
    ap.add_argument('--model', default=None, help='agent model id of the proxy (default: AGENT_MODEL, GLM-5.3-Flash), e.g. deepseek/deepseek-v4.1-flash')
    ap.add_argument('--temperature', type=float, default=None, help='temperature of the agent model (default: ChatModels.agent, 0.2)')
    ap.add_argument('--thinking-off', action='store_true', help='extra_body thinking disabled (DeepSeek: no reasoning tokens)')
    a = ap.parse_args()
    pairs = [p.split(':') for p in a.pairs]
    facts = build([c for c, _ in pairs], use_profiles=True)
    bank, budget = BankAnalytics(), Budget(a.budget)

    def one(pair):
        company, role = pair
        tools = Counter()
        def observe(name, args, text):
            tools[name] += 1
            budget.check()
            return text
        model = None
        if a.model or a.reasoning or a.thinking_off or a.temperature is not None:
            model = ChatModels()._make(a.model or SETTINGS.agent_model, 0.2 if a.temperature is None else a.temperature, 64000)
            model = model.model_copy(update={**({'reasoning_effort': a.reasoning} if a.reasoning else {}),
                                             **({'extra_body': {'thinking': {'type': 'disabled'}}} if a.thinking_off else {})})
        agent = AnalyticsInsightAgent(model=model, bank=bank, max_steps=a.max_steps, observe=observe)
        found = agent.run(company, facts[company], role, str(date.today()))
        insights = [candidate_to_insight(c, company, role, bank.rows, facts[company]) for c in found]  # contract v2, by code
        return {'company': company, 'role': role, 'tool_calls': dict(tools), 'error': agent.error,
                'finish': agent.finish.model_dump() if agent.finish else None,
                'candidates': [c.model_dump() for c in found], 'insights': [i.model_dump() for i in insights]}

    with ThreadPoolExecutor(len(pairs)) as pool:
        out = list(pool.map(one, pairs))
    budget.limit = float('inf')
    budget.check()  # bill of the calls still in flight when the loops ended
    path = DATA / 'analytics_agent' / 'runs.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'run': RUN_ID, 'model': a.model, 'temperature': a.temperature, 'reasoning': a.reasoning, 'thinking_off': a.thinking_off, 'cost_rub': round(budget.spent, 2), 'results': out}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'cost {budget.spent:.2f} ₽ -> {path}')
    for r in out:
        print(r['company'], r['role'], 'candidates:', len(r['candidates']), 'tools:', r['tool_calls'], 'error:', r['error'])
