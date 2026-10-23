"""Audit of the parser's metadata (library.py of the `mvp` worktree): topic dictionary, figures, own_view, stance,
freshness, on the real fragments (data/analytics/fragments.jsonl). Prints samples to read by eye; no LLM.
PARSER = path of library.py (default: the worktree D:/Projects/hacks/alfa/wt-mvp).
python -X utf8 stage5_insight_detection/bank_analytics/check_metadata.py"""
import ast, collections, json, os, random, re, sys
from pathlib import Path

A = Path(r'D:\Projects\hacks\alfa\alfafuture-hack\data\analytics')
LIB = Path(r'D:\Projects\hacks\alfa\wt-mvp\stage5_insight_detection\bank_analytics\library.py').read_text(encoding='utf8')
TOPICS = next(ast.literal_eval(n.value) for n in ast.parse(LIB).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'TOPICS')
F = [json.loads(l) for l in open(A / 'fragments.jsonl', encoding='utf8')]
T = [f for f in F if f['kind'] == 'text']
random.seed(3)
low = lambda t: ' ' + t.lower().replace('ё', 'е') + ' '

print('== TOPICS (16): fragments per topic (text kind) and the trigger words that fire')
for name, keys in TOPICS.items():
    rxs = {k: re.compile(r'(?<![а-яa-z])' + re.escape(k)) for k in keys}
    hit = collections.Counter(); n = 0
    for f in T:
        t = low(f['text']); fired = [k for k, r in rxs.items() if r.search(t)]
        if name in f['topics']:
            n += 1
            for k in fired: hit[k] += 1
    print(f'  {name:18} {n:3} fragments | ' + ', '.join(f'{k}:{v}' for k, v in hit.most_common(6)))
print('  text fragments with no topic:', sum(not f['topics'] for f in T), 'of', len(T), '| with ≥3 topics:', sum(len(f['topics']) >= 3 for f in T))

print('\n== topic precision by eye: 14 random (topic, trigger word in context)')
rows = [(t, f) for f in T for t in f['topics'] if len(f['text']) > 150]
for t, f in random.sample(rows, 14):
    tx = low(f['text']); k = next((k for k in TOPICS[t] if re.search(r'(?<![а-яa-z])' + re.escape(k), tx)), '?')
    i = tx.find(k); print(f'  [{t}] «{k}» … {tx[max(0, i - 55): i + 70].strip()!r}')

print('\n== figures: stats on text fragments')
fig = [(f, g) for f in T for g in f['figures']]
print('  figures', len(fig), '| with period', sum(bool(g['period']) for f, g in fig), '| indicator =', collections.Counter(g['indicator'] or '—' for f, g in fig).most_common(5),
      '| kind', collections.Counter(g['kind'] for f, g in fig), '| measure', collections.Counter(g.get('measure') for f, g in fig))
print('  12 random figures (value unit | indicator | kind | period | measure | sentence start):')
for f, g in random.sample(fig, 12):
    print(f"   {g['value']:g}{g['unit']:5} | {g['indicator'] or '—':14} | {g['kind']:8} | {g['period'] or '—':8} | {g.get('measure')} | {g['sentence'][:95]!r}")
print('  THE key rate fragments (11Sep p1.6 and 1H26 p3.6):')
for i in ('bank:ALFA-Macro_Insights-Policy_Rate_11Sep:p1.6', 'bank:ALFA-macro_stat1H26:p3.6'):
    f = next((x for x in F if x['id'] == i), None)
    if not f: continue
    print('   ', i, '| own_view', f['own_view'], '| stance', f['stance'], '| topics', f['topics'])
    for g in f['figures']: print(f"      {g['value']:g}{g['unit']} ind={g['indicator'] or '—'} kind={g['kind']} period={g['period'] or '—'} measure={g.get('measure')} | {g['sentence'][:110]!r}")

print('\n== own_view (the bank speaks) ==')
own = [f for f in T if f['own_view']]
print('  own_view fragments', len(own), 'of', len(T), '| by report:', dict(collections.Counter(f['report'][:22] for f in own)))
first = re.compile(r'\b(мы|нам|наш\w*)\b', re.I)
miss = [f for f in T if not f['own_view'] and first.search(f['text'])]
print('  NOT own_view but contain мы/наш:', len(miss))
for f in random.sample(miss, min(7, len(miss))):
    m = first.search(f['text']); print('    -', f['id'].split(':')[-1], repr(f['text'][max(0, m.start() - 40): m.end() + 100]))
print('  8 random own_view=True (first matching sentence):')
OWN = re.compile(r'\b(мы|нам|наш\w*)\b[^.]{0,80}?(ожида|жд[её]м|прогноз|считаем|полагаем|рекоменд|взгляд|мнени|оценк|оценива|видим|сохраня)|(ожида|прогноз|рекоменд)\w*[^.]{0,40}\b(мы|наш\w*)\b', re.I)
for f in random.sample(own, 8):
    m = OWN.search(f['text']); print('    +', f['id'].split(':')[-1], repr(f['text'][max(0, m.start() - 30): m.end() + 80]) if m else '(no regex hit?)')

print('\n== stance ==')
st = [f for f in own if f['stance']]
print('  own_view with a stance:', len(st), '| values:', collections.Counter(v for f in own for v in f['stance'].values()))
for f in random.sample(st, min(9, len(st))):
    print('   ', f['id'].split(':')[-1].ljust(10), f['stance'], '|', repr(re.sub(r'\s+', ' ', f['text'])[:150]))

print('\n== stance across dates (conflict signal): own_view stance by topic and report date ==')
rows = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for f in own:
    for t, v in f['stance'].items(): rows[t][f['date']][v] += 1
for t in ('ставка', 'курс', 'потребление', 'инфляция', 'облигации'):
    print('  ', t, {d: dict(c) for d, c in sorted(rows[t].items())})

print('\n== freshness ==')
print(collections.Counter((f['report'][:24], f['date'], f['freshness'], f['age_days']) for f in F).most_common(12))
