"""Cleanliness checks of the parsed bank analytics (data/analytics/fragments.jsonl, no LLM): characters, structure,
charts, tables, numbers vs independent OCR readings (ocr_old/ = the parser's first Qwen run, ocr/ = our local
readings), the macro table printed in several reports (historical years must agree).
python -X utf8 stage5_insight_detection/bank_analytics/check_parse.py"""
import collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

A = DATA / 'analytics'
F = [json.loads(l) for l in open(A / 'fragments.jsonl', encoding='utf8')]
print('fragments', len(F), dict(collections.Counter(f['kind'] for f in F)), 'reports', len({f['report'] for f in F}))

# 1 characters
mixed = collections.Counter(w for f in F for w in re.findall(r'\w[\w\-]*', f['text'])
                            if re.search('[а-яё]', w, re.I) and re.search('[a-z]', w, re.I))
cyr_digit = [f['id'] for f in F if re.search(r'(?<![\wа-яё])[ЗО](?=[,.]\d|[,.]?%|\d)|(?<![\wа-яё])[ЗО]К\d\d', f['text'])]
print('\n[1] mixed-script words:', sum(mixed.values()), '| top:', [w for w, _ in mixed.most_common(12)],
      '\n    (БО-002P-06 / 001Р-04R are bond issue codes, legit; шkaле, точka, RФ, СBonds are OCR errors)')
print('[1] fragments with Cyrillic З/О instead of 3/0 in a number («З,%», «ЗК26»):', len(cyr_digit))

# 2 structure
letters = lambda t: sum(c.isalpha() for c in t) / max(1, len(t))
norm = lambda t: re.sub(r'\W+', ' ', t.lower()).strip()
dup = collections.Counter(norm(f['text']) for f in F)
print('\n[2] < 80 chars:', sum(len(f['text']) < 80 for f in F), '| parser fallback notes «Резервное распознавание»:',
      sum('Резервное распознавание' in f['text'] for f in F), '| digit columns without labels (letters < 35%):',
      sum(f['kind'] == 'text' and len(f['text']) > 120 and letters(f['text']) < 0.35 for f in F),
      '| fragments with an exact duplicate:', sum(c for c in dup.values() if c > 1))
tb = [f for f in F if f['kind'] == 'table']
def headless_table(f):
    """No header row: the first non-empty row has only plain numbers in its value columns (a year or a word = header)."""
    rows = [r for r in f['text'].split('\n')[1:] if r.strip().startswith('|') and not re.match(r'^\|[-:| ]+$', r.strip())]
    rows = [[x.strip() for x in r.strip().strip('|').split('|')] for r in rows]
    first = next((r for r in rows if any(r)), None)
    vals = [x for x in (first or [])[1:] if x]
    return bool(vals) and not any(re.search('[A-Za-zА-Яа-я]', x) or re.fullmatch(r'(?:19|20)\d\d', x) for x in vals)
headless = [f['id'] for f in tb if headless_table(f)]
print('[2] tables without a header row (columns unlabelled):', len(headless), 'of', len(tb), headless)

# 3 charts
def defects(t):
    d = []
    if '⚠' in t: d.append('parser suspects OCR error')
    if re.search(r'диапазон от - до -|точка: значение|тренд или разворот одной фразой|Ряды \(легенда\): —', t): d.append('empty template / leaked instruction')
    if re.search(r'\d+\.\d{6,}', t): d.append('float garbage')
    yrs = [int(y) for y in re.findall(r"(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)[а-я]*[-.' ]*(\d{2})\b", ' '.join(re.findall(r'Ось X[^\n]*', t)))]
    if any(y >= 90 or y < 14 for y in yrs): d.append('implausible axis years')
    if 'не проверено' in t: d.append('values marked «по шкале, не проверено»')
    n = re.findall(r'-?\d+[.,]\d+', ' '.join(re.findall(r'Значения[^\n]*', t)))
    if len(n) >= 6 and collections.Counter(n).most_common(1)[0][1] / len(n) > 0.6: d.append('one value repeated')
    return d
ch = [f for f in F if f['kind'] == 'chart']
dc = collections.Counter(x for f in ch for x in defects(f['text']))
print(f'\n[3] charts {len(ch)}: with a defect {sum(bool(defects(f["text"])) for f in ch)}, defect-free {sum(not defects(f["text"]) for f in ch)},',
      'with ≥1 verified figure', sum(any(g.get('verified') for g in f['figures']) for f in ch), '|', dict(dc))

# 4 tables
q = c = 0
for f in tb:
    for r in f['text'].split('\n')[2:]:
        cells = r.strip().strip('|').split('|'); c += len(cells); q += sum('(?)' in x for x in cells)
print(f'\n[4] tables {len(tb)}: cells with «(?)» (three readings disagreed) {q}/{c} = {q / max(1, c):.0%}')

# 5 numbers vs independent readings
num = lambda t: {x.replace(',', '.').replace(' ', '') for x in re.findall(r'(?<![\w.,])\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|(?<![\w.,])\d+(?:[.,]\d+)?', t)}
page = lambda d, stem, p: next((json.loads(l)['value'] for l in open(d / f'{stem}.jsonl', encoding='utf8')
                                if json.loads(l)['key'] == str(p)), None) if (d / f'{stem}.jsonl').exists() else None
tot = sus = 0
for f in F:
    if f['kind'] != 'text': continue
    alts = [num(t) for t in (page(A / d, Path(f['file']).stem, f['page']) for d in ('ocr_old', 'ocr')) if t]
    if len(alts) < 2: continue
    for n in (x for x in num(f['text']) if len(x.replace('.', '')) >= 2):
        tot += 1; sus += not any(n in a for a in alts)
print(f'\n[5] numbers of text fragments found in neither independent reading of the page: {sus}/{tot} = {sus / max(1, tot):.1%}')

# 6 the macro table in several reports: historical years must agree
def parse(f):
    rows = [r for r in f['text'].split('\n') if r.strip().startswith('|') and not re.match(r'^\|[-:| ]+$', r.strip())]
    cells = lambda r: [x.strip() for x in r.strip().strip('|').split('|')]
    hdr, out = cells(rows[0]), {}
    for r in rows[1:]:
        x = cells(r); lab = re.sub(r'\s+', ' ', re.sub(r'\(\?\)|среднегодовая', '', x[0], flags=re.I)).strip().lower()
        out[lab] = {hdr[i]: re.sub(r'\(\?\)|%', '', x[i]).strip() for i in range(1, min(len(x), len(hdr)))}
    return out
T = {f['report'][:24]: parse(f) for f in tb if 'Основные макроэкономические показатели' in f['text'] and re.search(r'\|\s*2019\s*\|', f['text'])}
bad = n = 0
for lab in set().union(*T.values()):
    for y in map(str, range(2019, 2026)):
        v = {k: d[lab][y] for k, d in T.items() if lab in d and y in d[lab]}
        n += len(v) >= 2
        if len(set(v.values())) > 1: bad += 1; print('   disagree:', lab, y, v)
print(f'\n[6] historical cells of the macro table that differ between reports: {bad} of {n} ({", ".join(T)})')
