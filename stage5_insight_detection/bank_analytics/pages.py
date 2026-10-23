"""One page of OCR text made fit to read and to parse; shared by export_text.py (Markdown per report) and library.py
(fragments for the agent), so both see the same page.

- consecutive repeats of paragraphs, lines and sentences (model loops) are dropped and counted;
- a [График] block the model left unclosed is closed after its last field (Тип, Ось, Ряды, Значения, Что видно, Источник);
- a [График] block whose values look made up (one value 4+ times, 5+ decimals, a year on the axes or in the values before
  2020 or later than the report year + 3) gets a ⚠ line; its title, axes, series and
  source stay, its values must not be used as facts;
- a phrase of 1-6 words (with letters) repeated 5+ times in a row («(прогноз) [неразборчиво] …», a model loop) is cut to one and the
  page counts as broken; a Markdown table whose cells are all «—» becomes «значения не распознаны»;
- when an older OCR run is given, a page whose main OCR looks broken (under 75% of the fallback's length, digits read as
  I/O/Z, a short chunk looping) gets the fallback text appended under FALLBACK_NOTE, so nothing is lost; only its
  paragraphs and lines with no close match in the main text (word Jaccard < 0.6, under 80% of the words) are kept.
Pure code, no LLM.
"""
import re
from dataclasses import dataclass

CHART_WORDS = re.compile(r'\[График\]|график|диаграмм|\bрис\.|\bось\b|\bоси\b', re.I)
SENTENCE = re.compile(r'(?<=[.!?])\s+')
CHART_BLOCK = re.compile(r'\[График\].*?\[/График\]', re.S)
VALUE = re.compile(r'[-+−]?\d+(?:[.,]\d+)?')
GARBLED = re.compile(r'(?<!\w)(?=[\dIOZ,.%]*\d)(?=[\dIOZ,.%]*[IOZ])[\dIOZ,.%]{2,}(?!\w)|(?:(\w{2,4}?)\1{7,})')
FALLBACK_NOTE = '### Резервное распознавание (старый OCR): основное на этой странице неполное или искажено ({why})'
SUSPECT_NOTE = '⚠ Проверка кодом: значения графика похожи на ошибку распознавания ({why}); сверяйте с PDF.'


def _drop_repeats(units: list[str], max_block: int, key=lambda u: u.strip()) -> tuple[list[str], int]:
    """Drop a block of 1..max_block units equal to the block right before it (A A, AB AB); returns kept units, dropped."""
    out, removed, i = [], 0, 0
    while i < len(units):
        for k in range(1, max_block + 1):
            block = [key(u) for u in units[i:i + k]]
            if len(block) == k and len(out) >= k and any(block) and block == [key(u) for u in out[-k:]]:
                removed += k
                i += k
                break
        else:
            out.append(units[i])
            i += 1
    return out, removed


def dedup(text: str) -> tuple[str, int]:
    """Remove consecutive repeats of paragraphs, lines and sentences (model loops), trailing spaces and extra blank lines."""
    removed = 0
    paras = [p.strip('\n') for p in re.split(r'\n[ \t]*\n', text.replace('\r\n', '\n')) if p.strip()]
    paras, n = _drop_repeats(paras, 3)
    removed += n
    clean = []
    for p in paras:
        lines, n = _drop_repeats([ln.rstrip() for ln in p.split('\n')], 5)
        removed += n
        kept = []
        for ln in lines:
            parts, n = _drop_repeats(SENTENCE.split(ln), 2, key=lambda s: s.strip() if len(s.strip()) >= 20 else object())
            removed += n
            kept.append(' '.join(parts).rstrip())
        clean.append('\n'.join(kept))
    return '\n\n'.join(clean), removed


@dataclass
class PageStats:
    chars: int = 0
    tables: int = 0
    charts: int = 0
    approx: int = 0  # values read off the scale ("≈ по шкале")
    suspect: int = 0  # chart blocks with made-up-looking values
    fallback: int = 0  # pages that got the fallback OCR appended
    empty: int = 0
    repeats: int = 0

    def __iadd__(self, other: 'PageStats') -> 'PageStats':
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self


YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)|['’](\d{2})(?!\d)")


def odd_year(block: str, report_year: int | None) -> int | None:
    """A year on the axes or in the values outside 2020..report year + 3: the model misread the dates."""
    part = block.split('Значения', 1)[1].split('Что видно', 1)[0] if 'Значения' in block else ''
    axes = ' '.join(ln for ln in block.split('\n') if ln.lstrip(' *').startswith('Ось'))
    for m in YEAR.finditer(f'{axes} {part}'):
        y = int(m.group(1)) if m.group(1) else 2000 + int(m.group(2))
        if y < 2020 or report_year and y > report_year + 3:
            return y
    return None


def suspect_reason(block: str, report_year: int | None = None) -> str:
    """Why the values of a [График] block look made up, '' if they look fine."""
    part = block.split('Значения', 1)[1].split('Что видно', 1)[0] if 'Значения' in block else ''
    values = [m.group() for m in VALUE.finditer(part) if part[max(0, m.start() - 1):m.start()] not in "'’"]  # not сен'23
    repeated = [v for v in dict.fromkeys(values) if values.count(v) >= 4]
    if repeated:
        return f'значение {repeated[0]} повторяется {values.count(repeated[0])} раз'
    long = [v for v in values if len(re.split(r'[.,]', v + '.')[1]) >= 5]
    if long:
        return f'число {long[0]} с лишними знаками'
    year = odd_year(block, report_year)
    return f'на осях или в значениях {year} год, вне периода отчёта' if year else ''


CHART_FIELD = re.compile(r'^\s*\**\s*(Тип|Ось|Ряды|Значения|Что видно|Источник|[-–•*]\s|\s)')


def close_charts(text: str) -> str:
    """Add the [/График] the model forgot: before the next block, or at a blank line not followed by another field."""
    lines, out, inside = text.split('\n'), [], False
    for i, line in enumerate(lines):
        s = line.strip()
        if inside and s.startswith('[/График]'):
            inside = False
        elif inside and s.startswith('[График]'):
            out.append('[/График]')
        elif inside and not s:
            nxt = next((ln for ln in lines[i + 1:] if ln.strip()), None)
            if nxt is None or not (CHART_FIELD.match(nxt) or nxt.strip().startswith('[/График]')) or nxt.strip().startswith('[График]'):
                out.append('[/График]')
                inside = False
        elif s.startswith('[График]'):
            inside = True
        out.append(line)
    return '\n'.join(out + (['[/График]'] if inside else []))


def flag_charts(text: str, report_year: int | None = None) -> tuple[str, int]:
    n = 0

    def flag(m):
        nonlocal n
        why = suspect_reason(m.group(), report_year)
        if not why:
            return m.group()
        n += 1
        return m.group().replace('[/График]', SUSPECT_NOTE.format(why=why) + '\n[/График]')
    return CHART_BLOCK.sub(flag, text), n


def count_tables(text: str) -> int:
    rows = [ln.strip().startswith('|') for ln in text.split('\n')]
    return sum(1 for i, r in enumerate(rows) if r and (i == 0 or not rows[i - 1]))


def count_charts(text: str) -> int:
    """[График] blocks, plus paragraphs outside them that mention a chart."""
    outside = re.sub(r'\[График\].*?(\[/График\]|$)', '', text, flags=re.S)
    return text.count('[График]') + sum(1 for p in re.split(r'\n\s*\n', outside) if CHART_WORDS.search(p))


def broken_reason(text: str, fallback: str) -> str:
    """Why the main OCR of a page looks worse than the fallback, '' if it does not."""
    if len(fallback.strip()) > 300 and len(text.strip()) < 0.75 * len(fallback.strip()):
        return f'{len(text.strip())} символов против {len(fallback.strip())}'
    found = [m for m in GARBLED.finditer(text)]
    loops = [m.group()[:12] for m in found if m.group(1)]
    if loops:
        return f'зацикливание «{loops[0]}…»'
    return f'{len(found)} чисел с буквами вместо цифр' if len(found) >= 5 else ''


LOOP_NOTE = '[зацикливание распознавания вырезано]'
EMPTY_TABLE = 'значения не распознаны'


def cut_phrase_loops(text: str, max_words: int = 6, times: int = 5) -> tuple[str, str]:
    """A phrase of 1..max_words words repeated `times`+ in a row is cut to one occurrence; returns (text, the phrase)."""
    words = list(re.finditer(r'\S+', text))
    for n in range(1, max_words + 1):
        i = 0
        while i + n * times <= len(words):
            unit = [w.group() for w in words[i:i + n]]
            if not any(re.search(r'[A-Za-zА-Яа-яЁё]{2}', u) for u in unit):
                i += 1  # numbers and «| — |» repeat in real tables; repeated values are the ⚠ check's job
                continue
            k = 1
            while i + (k + 1) * n <= len(words) and [w.group() for w in words[i + k * n:i + (k + 1) * n]] == unit:
                k += 1
            if k >= times:
                start, end = words[i + n].start() if n else 0, words[i + k * n - 1].end()
                phrase = ' '.join(unit)
                cut = text[:words[i + n - 1].end()] + f' {LOOP_NOTE}' + text[end:]
                return cut_phrase_loops(cut, max_words, times)[0], phrase
            i += 1
    return text, ''


def drop_empty_tables(text: str) -> str:
    """A Markdown table whose value cells are all «—» carries nothing: one line instead of it."""
    out, table = [], []
    for line in text.split('\n') + ['']:
        if line.strip().startswith('|'):
            table.append(line)
            continue
        if table:
            rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in table]
            values = [c for r in rows[1:] for c in r[1:] if not re.fullmatch(r':?-{2,}:?', c)]
            out += [EMPTY_TABLE] if values and all(c in ('—', '-', '–', '') for c in values) else table
            table = []
        out.append(line)
    return '\n'.join(out[:-1])


WORD = re.compile(r'[a-zа-яё0-9]+(?:[.,][0-9]+)?', re.I)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def doubled(a: set, b: set, threshold: float) -> bool:
    """a repeats b: word Jaccard >= threshold, or 80% of a's words are in b (b may carry a loop note or more text)."""
    return jaccard(a, b) >= threshold or bool(a) and len(a & b) / len(a) >= 0.8


def only_new(old: str, main: str, threshold: float = 0.6) -> str:
    """Paragraphs and lines of the fallback OCR that the main text does not already have (word Jaccard < threshold)."""
    words = lambda t: set(WORD.findall(t.lower()))
    main_paras = [words(p) for p in re.split(r'\n\s*\n', main) if p.strip()]
    main_lines = [words(ln) for ln in main.split('\n') if len(words(ln)) >= 3]
    kept = []
    for para in re.split(r'\n\s*\n', old):
        if not para.strip() or any(doubled(words(para), m, threshold) for m in main_paras):
            continue
        lines = [ln for ln in para.split('\n')
                 if len(words(ln)) < 3 or not any(doubled(words(ln), m, threshold) for m in main_lines)]
        if any(words(ln) for ln in lines):
            kept.append('\n'.join(lines))
    return '\n\n'.join(kept)


def clean_page(raw: str, fallback: str = '', report_year: int | None = None) -> tuple[str, PageStats]:
    """Page text after dedup and chart flags, with the fallback OCR appended if the main one looks broken."""
    raw, loop = cut_phrase_loops(raw or '')
    text, repeats = dedup(drop_empty_tables(raw))
    text, suspect = flag_charts(close_charts(text), report_year)
    stats = PageStats(chars=len(text), tables=count_tables(text), charts=count_charts(text), approx=text.count('≈ по шкале'),
                      suspect=suspect, empty=int(not text.strip()), repeats=repeats)
    old = dedup(fallback)[0] if fallback else ''
    why = (f'зацикливание «{loop[:40]}»' if loop else broken_reason(text, old)) if old.strip() else ''
    new = only_new(old, text) if why else ''
    if new.strip():
        text = f'{text.strip()}\n\n{FALLBACK_NOTE.format(why=why)}\n\n{new.strip()}'
        stats.fallback = 1
    return text.strip(), stats
