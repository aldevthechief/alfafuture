"""Pool of showable insight cards per (company, role) in three levels of detail, read from the running backend (no LLM).
python experiments/insight_ranking/build_pool.py  ->  data/experiments/ranking/pool_<company>_<role>.json"""
import json
import sys
import urllib.request
from pathlib import Path

API = 'http://localhost:18010'
OUT = Path(__file__).resolve().parents[2] / 'data' / 'experiments' / 'ranking'
PAIRS = [('U_14366', 'cfo'), ('U_42222', 'owner')]


def get(path):
    with urllib.request.urlopen(API + path, timeout=120) as r:
        return json.load(r)


def texts(items, n=None):
    return [x['text'] if isinstance(x, dict) else str(x) for x in (items or [])][:n]


def card(c, row):
    i = row['insight']
    base = dict(id=c['id'], title=c['title'], origin=i.get('origin'), channel=c['channel'], confidence=i.get('confidence'),
                impact=(i.get('impact') or {}).get('level'), published=(c.get('published') or '')[:10],
                n_risks=len(i.get('risks') or []), n_opps=len(i.get('opportunities') or []), n_actions=len(i.get('actions') or []),
                has_bank=bool(i.get('bank_view')))
    s4 = (row.get('stage4') or {}).get('relevance') or {}
    gens = (row.get('retrieval') or {}).get('generators') or []
    base.update(rel=s4.get('score'), role_fit=s4.get('role_fit'), judge=(row.get('decision') or {}).get('score'),
                n_gen=len(gens), event_id=(row.get('retrieval') or {}).get('event_id'))
    lead = texts(i.get('why_it_matters'), 1)
    base['L1'] = f"{c['title']} [{c['channel']}]"
    base['L2'] = f"{c['title']}\nПочему важно: {lead[0] if lead else ''}\nДействия: {'; '.join(a['text'] if isinstance(a, dict) else str(a) for a in (i.get('actions') or [])[:2])}"
    base['L3'] = '\n'.join([f"{c['title']}", f"Канал: {c['channel']}; уверенность: {i.get('confidence')}; влияние: {(i.get('impact') or {}).get('level')}; дата: {base['published']}",
        'Что произошло: ' + ' '.join(texts(i.get('what_happened'), 3)), 'Почему важно: ' + ' '.join(texts(i.get('why_it_matters'), 4)),
        'Риски: ' + ' | '.join(texts(i.get('risks'), 3)), 'Возможности: ' + ' | '.join(texts(i.get('opportunities'), 3)),
        'Действия: ' + ' | '.join(a['text'] if isinstance(a, dict) else str(a) for a in (i.get('actions') or [])[:3]),
        'Позиция Банка: ' + ' '.join(texts(i.get('bank_view'), 2))])
    return base


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for co, role in PAIRS:
        cards = []
        for c in get(f'/v1/cabinet/feed?company_id={co}&role={role}')['insights']:
            row = get(f"/v1/jobs/{c['job_id']}/insights/{c['result_index']}")['result']['results'][0]
            cards.append(card(c, row))
        (OUT / f'pool_{co}_{role}.json').write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding='utf-8')
        print(co, role, len(cards), 'chars L1/L2/L3:', sum(len(x['L1']) for x in cards), sum(len(x['L2']) for x in cards), sum(len(x['L3']) for x in cards))
