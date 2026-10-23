"""Analytic-retrieve v1, chunking: retrievable fragments of the parser (fragments.use_fragment) -> chunks of about 300 e5
tokens, and context windows around found chunks. Independent of the insight agent: any consumer (the insight tools, a
candidate generator over the bank's views) gets chunks / passages from `search.BankAnalytics`.

- Text: the report's kept paragraphs, plus short headline fragments (a thesis line above a paragraph), form one stream per
  stretch between tables; langchain's RecursiveCharacterTextSplitter cuts it by meaning: paragraph, line, sentence («. »),
  clause, word; length = e5 tokens (CHUNK_TOKENS = 300 leaves room for the report title that goes in front of the text at
  indexing, e5 reads 512), OVERLAP_TOKENS = 50. A sentence the OCR broke at a page border is rejoined by the stream.
- Tables: cut by rows to TABLE_TOKENS with the title and header row repeated (a long macro table does not fit 512 tokens).
- A chunk knows its stream offsets, so a window of neighbours is cut out of the stream without the overlap repeated.
chunk_tokens = 0: one chunk per kept fragment (the pre-chunking index, kept for the evaluation scripts)."""
import re
from collections import defaultdict
from datetime import date

from langchain_text_splitters import RecursiveCharacterTextSplitter

from stage5_insight_detection.bank_analytics.fragments import CAPTION, prose_reason, use_fragment

CHUNK_TOKENS, OVERLAP_TOKENS, TABLE_TOKENS = 300, 50, 320
SEPARATORS = ['\n\n', '\n', '. ', '; ', ', ', ' ', '']
NOT_HEADING = re.compile(r'Альфа-КИБ \||Центр макроэкономического анализа|Внизу страницы|^\W*Приложение|значения не распознаны|Контакты'
                         r'|— (красный|серый|зел[её]ный|синий)', re.I)


def is_heading(f: dict) -> bool:
    """A short text fragment that is a headline / thesis line, not a caption, footer or legend: it travels with the paragraph below."""
    t = f['text'].strip()
    return (f['kind'] == 'text' and prose_reason(f) == 'short' and not CAPTION.match(t) and not NOT_HEADING.search(t)
            and len(re.findall(r'[а-яё]{3,}', t.lower())) >= 3)


def _table_chunks(f: dict, token_len, limit: int) -> list[str]:
    cells = lambda r: [x.strip() for x in r.strip().strip('|').split('|')]
    lines = f['text'].split('\n')
    rows = [r for r in lines[1:] if r.strip().startswith('|') and not re.match(r'^\|[-:| ]+$', r.strip()) and any(cells(r))]
    if len(rows) < 2:
        return [f['text'].strip()]
    head = [lines[0].strip(), rows[0], '|' + '---|' * len(cells(rows[0]))]
    out, cur = [], []
    for r in rows[1:]:
        if cur and token_len('\n'.join(head + cur + [r])) > limit:
            out.append('\n'.join(head + cur)); cur = []
        cur.append(r)
    return out + ['\n'.join(head + cur)]


def build_chunks(rows: list[dict], token_len, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS,
                 keep=use_fragment, *, as_of: date | None = None) -> tuple[list[dict], dict]:
    """rows: parser fragments in file order (headers repaired). Returns (chunks in reading order, stream texts by id).
    Chunk: id, report, date, stale, page, pages, kind (text | table), text, frag_ids, seg, start, end (stream offsets)."""
    from stage5_insight_detection.bank_analytics.library import freshness, stance_of

    by_report = defaultdict(list)
    for f in rows:
        by_report[f['report']].append(f)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_tokens, chunk_overlap=overlap_tokens, length_function=token_len,
                                              separators=SEPARATORS, add_start_index=True) if chunk_tokens else None
    chunks, segments = [], {}
    as_of = as_of or date.today()
    for report, frs in by_report.items():
        try:
            when = date.fromisoformat(frs[0].get('date') or '')
        except ValueError:
            when = None
        fresh, age = freshness(when, as_of)
        meta = {'report': report, 'date': frs[0].get('date') or '', 'freshness': fresh,
                'age_days': age, 'stale': fresh in ('stale', 'unknown')}
        # Keep source attribution, not just report membership. Statistics and quoted CB views are not our forecast.
        views = {f['id']: f['own_view'] if 'own_view' in f else stance_of(f['text'])[0] for f in frs}
        stem = re.sub(r'^bank:|:p\d+\.\d+$', '', frs[0]['id'])
        parts, first = [], len(chunks)

        def add(text, kind, page, fids, **extra):
            view_ids = [fid for fid in fids if views[fid]]
            chunks.append({**meta, 'id': f'{fids[0]}+c{len(chunks) - first + 1}', 'kind': kind, 'text': text, 'page': page,
                           'own_view': bool(view_ids), 'view_frag_ids': view_ids,
                           'pages': sorted({page, *extra.pop('pages', [])}), 'frag_ids': fids, 'seg': None, 'start': 0, 'end': 0, **extra})

        def flush():
            if not parts:
                return
            text, spans = '', []
            for t, page, fid in parts:
                text += '\n\n' if text else ''
                spans.append((len(text), len(text) + len(t), page, fid)); text += t
            seg = f'{stem}:s{len(segments)}'; segments[seg] = text
            for d in splitter.create_documents([text]):
                s = d.metadata['start_index']; e = s + len(d.page_content)
                inside = [x for x in spans if x[0] < e and x[1] > s] or spans[:1]
                add(d.page_content, 'text', inside[0][2], [x[3] for x in inside], pages=[x[2] for x in inside], seg=seg, start=s, end=e)
            parts.clear()

        for f in frs:
            kind = f['kind']
            if keep(f) and kind == 'table' and splitter:
                flush()
                for t in _table_chunks(f, token_len, TABLE_TOKENS):
                    add(t, 'table', f['page'], [f['id']])
            elif keep(f) and kind == 'chart' and splitter:  # a table the parser took for a chart: its own stream
                flush(); parts.append((f['text'].strip(), f['page'], f['id'])); flush()
            elif keep(f) and not splitter:
                add(f['text'].strip(), kind, f['page'], [f['id']])
            elif keep(f) or (splitter and is_heading(f)):
                parts.append((f['text'].strip(), f['page'], f['id']))
        flush()
    return chunks, segments


def _merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def passages(chunks: list[dict], segments: dict, hits: list[int], context: int = 2) -> list[dict]:
    """Windows of `context` chunks above and below every hit (positions in `chunks`), overlapping windows of one report
    merged. Text is cut from the stream, so the overlap of neighbours is not repeated; the hit parts are marked
    «>>> найдено >>> … <<<». Returns [{report, date, pages, stale, ids (hit chunk ids), text}] in reading order."""
    ranges = []
    for i in sorted(set(hits)):
        lo, hi = max(0, i - context), min(len(chunks) - 1, i + context)
        while chunks[lo]['report'] != chunks[i]['report']: lo += 1
        while chunks[hi]['report'] != chunks[i]['report']: hi -= 1
        if ranges and chunks[ranges[-1][0]]['report'] == chunks[i]['report'] and lo <= ranges[-1][1] + 1:
            ranges[-1] = [ranges[-1][0], max(ranges[-1][1], hi), ranges[-1][2] | {i}]
        else:
            ranges.append([lo, hi, {i}])
    out = []
    for lo, hi, hit in ranges:
        pieces, j = [], lo
        while j <= hi:
            c = chunks[j]
            if c['seg'] is None:  # table (or whole-fragment chunk): shown as is
                pieces.append(f">>> найдено >>>\n{c['text']}\n<<<" if j in hit else c['text']); j += 1; continue
            k = j
            while k + 1 <= hi and chunks[k + 1]['seg'] == c['seg']: k += 1
            run = range(j, k + 1)
            ctx, hot = _merge([[chunks[x]['start'], chunks[x]['end']] for x in run]), _merge([[chunks[x]['start'], chunks[x]['end']] for x in run if x in hit])
            cold = []
            for s, e in ctx:
                cur = s
                for hs, he in hot:
                    if he <= cur or hs >= e: continue
                    if hs > cur: cold.append((cur, hs))
                    cur = max(cur, he)
                if cur < e: cold.append((cur, e))
            for s, e, is_hit in sorted([(s, e, False) for s, e in cold] + [(s, e, True) for s, e in hot]):
                t = segments[c['seg']][s:e].strip()
                if t: pieces.append(f'>>> найдено >>>\n{t}\n<<<' if is_hit else t)
            j = k + 1
        span = [chunks[x] for x in range(lo, hi + 1)]
        out.append({'report': chunks[lo]['report'], 'date': chunks[lo]['date'], 'stale': chunks[lo]['stale'],
                    'freshness': chunks[lo]['freshness'],
                    'own_view': any(chunks[i]['own_view'] for i in hit),
                    'pages': sorted({p for x in span for p in x['pages']}), 'ids': [chunks[i]['id'] for i in sorted(hit)],
                    'context_ids': [c['id'] for c in span],
                    'text': '\n\n'.join(pieces)})
    return out
