"""What the insight agent (stage 5 InsightJudge ToolLoop -> stage 6 InsightWriter) does with judge v2 positives and
negatives of one company. Blind: stage 4 input is empty, the agent does not see the judge v2 verdict. Bank analytics:
whatever is OCR'd in data/analytics/ocr (pipeline.optional_bank). Results: out/agent_<company>.jsonl, one line per event (decision, insight, review).
python -X utf8 experiments/judge_v2/agent_run.py --events 3290 1431 [--role cfo] [--workers 6]"""
import argparse, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge import OUT, event_news, facts_for, parallel  # also sets sys.path
from pipeline import optional_bank
from common.llm import ChatModels, JsonlCache
from stage5_insight_detection.insight_judge.judge import InsightJudge
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', default='U_106783')
    ap.add_argument('--events', type=int, nargs='+', required=True)
    ap.add_argument('--role', default='cfo')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    u, facts = a.company, facts_for(a.company)
    key_facts = [f'[{f["cat"]}] {f["text"]}' for f in facts if f['depth'] == 1]
    texts = event_news(a.events)
    models = ChatModels()
    bank = optional_bank()
    judge, writer = InsightJudge(models.agent(temperature=0.0), bank=bank), InsightWriter(models.agent(temperature=0.3))
    OUTS, lock = JsonlCache(f'agent_{u}', folder=OUT), threading.Lock()

    def run(e):
        if OUTS.get(f'{a.role}|{e}'):
            return
        nid = int(texts[e]['id'])
        d = judge.decide(nid, u, facts, a.role, {})
        rec = {'event': e, 'news_id': nid, 'title': texts[e]['title'], 'decision': d.model_dump() if d else None}
        if d and d.is_insight:
            news = judge.related.get(nid)
            related = [judge.related.get(i) for i in d.related_news_ids[:3] if i in judge.related.row_of]
            hits = bank.search(d.mechanism) if bank else []
            insight, review = writer.write(u, a.role, d.model_dump_json(), Evidence(news=[news, *related], facts=key_facts, bank=hits))
            rec |= {'insight': insight.model_dump(), 'review': review.model_dump() if review else None}  # checker may answer without the tool call
        with lock:
            OUTS.put(f'{a.role}|{e}', rec)
        print(f"  {e}: decision {'none' if d is None else (d.is_insight, d.score)}{' + insight' if 'insight' in rec else ''} | {texts[e]['title'][:70]}", flush=True)

    parallel(run, a.events, a.workers)
