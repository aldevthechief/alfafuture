"""Check that a quote the agent gives as evidence stands on the report page it names (no LLM, difflib only).

The parsed pages carry OCR defects (dropped or reworded phrases, Latin letters for Cyrillic, «З» / «О» for digits 3 / 0), and the
agent corrects some of them while quoting, so the match is tolerant:
- both texts go through one normal form: lower case, «ё» = «е», dashes and quotes unified, «(?)» and table bars dropped, punctuation
  dropped except a decimal comma / dot inside a number and «%», Latin lookalikes -> Cyrillic, capital «З» -> 3, «О» -> 0;
- `exact`: the normal form of the quote is inside the page;
- `fuzzy`: the place is found by the words only (every token with a digit or % is one placeholder: OCR drops, garbles and splits numbers),
  a window of the page with ≥ FUZZY of the quote's words in the same order; the page text at that place is kept as `source_excerpt`,
  differing words are listed, and the numbers of the quote are compared with the numbers of that text: a difference in a number is what
  matters (the reading was corrected or taken from the picture: the agent marks it kind=picture);
- `table_row`: a table quote is a row in words and numbers: the cell of that row and period must hold the number;
- not on the named page: the other pages of the report (`relocated`, the page is corrected), then the other reports, else `not_found`
  (`too_short` for a quote under MIN_WORDS words that is nowhere).
Offsets `start` / `end` are positions in the text of the page fragments joined with a blank line (`page_text`).
"""
import re
from difflib import SequenceMatcher

FUZZY = 0.85
MIN_WORDS = 4  # a shorter quote proves nothing: exact match only
LOOKALIKE = str.maketrans('ABCEHKMOPTXYaceopxy', 'АВСЕНКМОРТХУасеорху')
MAP = str.maketrans({'ё': 'е', 'Ё': 'Е', '«': '"', '»': '"', '“': '"', '”': '"', '„': '"', '–': '-', '—': '-', '−': '-', ' ': ' ',
                     'З': '3', 'О': '0'})  # capitals only (the defect is «ЗО%», «ЗОЗЗ»); lower «з», «о» are letters, too common in words
NUM_TOKEN = re.compile(r'(?<![А-Яа-яЁё])[ЗО\d][ЗО\d,.]*%?(?![А-Яа-яЁё\d])')  # no backtracking into «202» of «2026П»
DIGIT_LETTERS = re.compile(r'(?<![А-Яа-яЁё])(?:[ЗО]{2,}|[ЗО]?[,.][ЗО]*%|[ЗО]%)(?![А-Яа-яЁё])')  # «ЗО%», «З,%», «ЗОЗЗ»: digits read as letters
GENERIC = {'конец', 'года', 'руб'}  # words of table labels that say nothing about the row


def normalize(text: str) -> tuple[str, list[int]]:
    """(normal form, original index of every character of it)."""
    text = text.translate(LOOKALIKE).translate(MAP)
    skip = {k for m in re.finditer(r'\(\?\)', text) for k in range(m.start(), m.end())}
    out, idx = [], []
    for i, c in enumerate(text):
        keep = c.isalnum() or c == '%' or (c in ',.' and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit())
        if i in skip or not keep:
            if out and out[-1] != ' ':
                out.append(' ')
                idx.append(i)
        else:
            out.append(c.lower())
            idx.append(i)
    while out and out[-1] == ' ':
        out.pop()
        idx.pop()
    return ''.join(out), idx


def words(norm: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\S+', norm)]


def shape(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Every token with a digit or % becomes one placeholder «N», runs of them merge."""
    out = []
    for w, a, b in spans:
        if re.search(r'[\d%]', w):
            if out and out[-1][0] == 'N':
                out[-1] = ('N', out[-1][1], b)
            else:
                out.append(('N', a, b))
        else:
            out.append((w, a, b))
    return out


def number_canon(token: str) -> str:
    t = token.translate(MAP).rstrip('%').replace(',', '.').strip('.')
    return re.sub(r'\.?0+$', '', t) if '.' in t else t


def number_differences(quote: str, excerpt: str) -> list[tuple[str, str]]:
    """Numbers of the quote against the numbers of the page text it matched: [(quote spelling, page spelling)] that differ."""
    q, p = NUM_TOKEN.findall(quote), NUM_TOKEN.findall(excerpt)
    qc, pc = [number_canon(x) for x in q], [number_canon(x) for x in p]
    return [(' '.join(q[a1:a2]), ' '.join(p[b1:b2])) for tag, a1, a2, b1, b2 in SequenceMatcher(None, qc, pc, autojunk=False).get_opcodes() if tag != 'equal']


def locate(page: str, quote: str) -> dict | None:
    """Best place of the quote in the page: {score, start, end, differences: words, number_differences: numbers}."""
    pn, pidx = normalize(page)
    qn, _ = normalize(quote)
    if not qn or not pn:
        return None
    at = pn.find(qn)
    if at >= 0:
        return {'score': 1.0, 'exact': True, 'start': pidx[at], 'end': pidx[at + len(qn) - 1] + 1, 'differences': [], 'number_differences': []}
    qs, ps = shape(words(qn)), shape(words(pn))
    qw = [w for w, _, _ in qs]
    if len(qw) < MIN_WORDS or not ps:
        return None
    best, qset = None, set(qw)
    for n in {len(qw) - 2, len(qw), len(qw) + 2}:
        if n < 1:
            continue
        for i in range(0, max(1, len(ps) - n + 1)):
            window = [w for w, _, _ in ps[i:i + n]]
            if len(qset & set(window)) < 0.5 * len(qset):
                continue
            ratio = SequenceMatcher(None, qw, window, autojunk=False).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, i, min(i + n, len(ps)), window)
    if best is None or best[0] < FUZZY:
        return None
    ratio, i, j, window = best
    ops = SequenceMatcher(None, qw, window, autojunk=False).get_opcodes()
    if ops[0][0] == 'insert':  # page words before the quote's first word do not belong to the place
        i += ops[0][4] - ops[0][3]
    if ops[-1][0] == 'insert':
        j -= ops[-1][4] - ops[-1][3]
    start, end = pidx[ps[i][1]], pidx[ps[j - 1][2] - 1] + 1
    diffs = [(' '.join(qw[a1:a2]), ' '.join(window[b1:b2])) for tag, a1, a2, b1, b2 in ops
             if tag != 'equal' and not (set(qw[a1:a2]) | set(window[b1:b2])) <= {'N'}]
    return {'score': round(ratio, 3), 'exact': False, 'start': start, 'end': end, 'differences': diffs, 'number_differences': number_differences(quote, page[start:end])}


def page_text(rows: list[dict], report: str, page: int) -> str:
    return '\n\n'.join(f['text'] for f in rows if f['report'] == report and f['page'] == page and f['kind'] in ('text', 'table'))


def table_row(rows: list[dict], report: str, page: int, quote: str) -> tuple[str, str] | None:
    """A table quote is a row in words and numbers («Курс, конец года, руб./$ 2027П 93,0»). ('ok', cell) when the cell of that row and period holds
    the number, ('differs', cell) when the row and period are found with another number, None when the row is not on the page."""
    from stage5_insight_detection.bank_analytics.number_check import norm_period, numbers, table_cells
    qn = normalize(quote)[0]
    year = r'(?:19|20)\d\d\s?[пp]?\b'
    periods = {norm_period(m.group(0)) for m in re.finditer(year, qn)}
    qnums = set(numbers(re.sub(year, ' ', quote)))
    qwords = set(re.findall(r'[а-я]{3,}', qn)) - GENERIC
    found = None
    for _, _, _, label, period, value, cell in table_cells([f for f in rows if f['report'] == report and f['page'] == page]):
        lw = set(re.findall(r'[а-я]{3,}', label)) - GENERIC
        if qwords and lw and len(qwords & lw) >= 0.6 * min(len(qwords), len(lw)) and (not periods or period in periods):
            if value in qnums:
                return 'ok', f'{label}, {period}: {cell}'
            found = found or ('differs', f'{label}, {period}: {cell}')
    return found


def verify(rows: list[dict], reports: list[dict], ev: dict) -> dict:
    """ev: {report, date, page, quote, kind}. Returns the system fields: status, cited_page, page (corrected), source_excerpt, start, end,
    differences, advice (Russian sentences for the agent; empty when the quote stands as given)."""
    out = {'status': '', 'cited_page': ev['page'], 'page': ev['page'], 'source_excerpt': '', 'start': None, 'end': None, 'differences': [], 'advice': []}
    name = ev.get('report', '').lower()
    r = next((x for x in reports if x['date'] == ev.get('date')), None) or next(
        (x for x in reports if name and (x['title'].lower() in name or name in x['title'].lower())), None)
    if r is None:
        return out | {'status': 'unknown_report', 'advice': [f'отчёт «{ev.get("report")}» ({ev.get("date")}) не найден: название и дату лучше указывать как в результатах поиска']}
    title = r['title']
    if ev.get('kind') == 'picture':
        # a reading «of the picture» that quotes text the page has loses the check: say so
        own = page_text(rows, title, ev['page'])
        for seg in re.findall(r'«([^»]+)»', ev['quote']):
            if len(words(normalize(seg)[0])) >= MIN_WORDS and locate(own, seg):
                return out | {'status': 'picture', 'advice': [f'фраза «{seg[:80]}» есть в тексте страницы: её можно оформить доказательством kind=text с дословной цитатой '
                                                              '(kind=picture нужен только для значений, которых нет в тексте: графики, повреждённые цифры)']}
        return out | {'status': 'picture'}
    if ev.get('kind') == 'table':
        row = table_row(rows, title, ev['page'], ev['quote'])
        if row and row[0] == 'ok':
            return out | {'status': 'table_row', 'source_excerpt': row[1]}
        if row:
            return out | {'status': 'table_value_differs', 'source_excerpt': row[1],
                          'advice': [f'в таблице на стр. {ev["page"]} для этой строки и периода стоит другое значение: «{row[1]}»; в цитате число другое — возможно, стоит поправить']}
    own = page_text(rows, title, ev['page'])
    hit = locate(own, ev['quote'])
    if hit:
        out |= {'status': 'exact' if hit['exact'] else 'fuzzy', 'source_excerpt': own[hit['start']:hit['end']], 'start': hit['start'],
                'end': hit['end'], 'differences': hit['differences']}
        if hit['number_differences']:
            out['advice'].append('в цитате и на странице разные числа: ' + '; '.join(f'в цитате «{a}», на странице «{b}»' for a, b in hit['number_differences'])
                                 + ' — на странице это может быть дефект распознавания; если значение исправлено или прочитано по картинке, его можно '
                                 'оформить доказательством kind=picture (приблизительное значение)')
        garbled = DIGIT_LETTERS.findall(out['source_excerpt'])
        if garbled and not hit['number_differences']:
            out['advice'].append(f'в тексте страницы на этом месте цифры распознаны как буквы ({", ".join(garbled[:3])}): значения можно взять с картинки (look_at_page) '
                                 'и оформить доказательством kind=picture')
        return out
    for p in sorted({f['page'] for f in rows if f['report'] == title and f['page'] != ev['page']}):
        text = page_text(rows, title, p)
        hit = locate(text, ev['quote'])
        if hit:
            return out | {'status': 'relocated', 'page': p, 'source_excerpt': text[hit['start']:hit['end']], 'start': hit['start'], 'end': hit['end'],
                          'differences': hit['differences'],
                          'advice': [f'цитата найдена на стр. {p} отчёта «{title}», а не на стр. {ev["page"]}: страница в доказательстве исправлена']}
    for other in sorted({f['report'] for f in rows if f['report'] != title}):
        for p in sorted({f['page'] for f in rows if f['report'] == other}):
            hit = locate(page_text(rows, other, p), ev['quote'])
            if hit and hit['exact']:
                return out | {'status': 'other_report', 'advice': [f'цитата стоит в отчёте «{other}» стр. {p}, а не в «{title}»: возможно, стоит поправить отчёт, дату и страницу']}
    if len(words(normalize(ev['quote'])[0])) < MIN_WORDS:
        return out | {'status': 'too_short', 'advice': ['цитата короче 4 слов: её нельзя проверить, приведи полную фразу со страницы']}
    return out | {'status': 'not_found', 'advice': [f'цитаты нет на стр. {ev["page"]} отчёта «{title}» и на других его страницах: можно взять дословный фрагмент '
                                                    'из read_page или результата поиска (править распознавание можно, смысл и числа менять нельзя)']}
