"""Digest of one bank report: summary, the bank's own views, key figures, risks, sectors and caveats, each with its page.

Input is the Markdown of export_text.py (data/analytics/text/<stem>.md). Pages go to the agent model (GLM, default
reasoning) as untrusted items; a long report is cut by pages into chunks of ~12k chars and the chunk answers are merged.
After the answer code checks every item: each of its numbers must occur on the page it cites (audit.numbers), otherwise
or without a page the item is `unverified`. Chart values the OCR model read off the scale (≈) or that export_text flagged
(⚠) proved made up on checked pages, so they are cut before the model sees the page and cannot pass the check either.
`--budget` (₽, the proxy's `usage.cost`) stops before the next chunk would pass it; a report cut short is not written.
Several chunks: their summaries are merged by one short call (prompts/digest_merge.md) into one summary; a sentence of
a chunk that claims something is absent («прогнозов нет») is dropped first, it is only true of its chunk, and a merged
sentence with a number none of the chunk summaries has is dropped after. Chunk summaries are kept in the JSON, so
`--merge-only` redoes the merge of a written digest without its chunks.
Output: <stem>.digest.json and a readable <stem>.digest.md beside the input.
python -X utf8 stage5_insight_detection/bank_analytics/digest.py [--text-dir data/analytics/text] [--budget 3] [--merge-only] [stem ...]
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, PromptLibrary, structured
from common.safety import UNTRUSTED_NOTE, render_untrusted
from stage5_insight_detection.bank_analytics.library import sentences, is_own_view
from common.quantities import quantities
from stage5_insight_detection.bank_analytics.export_text import TEXT_DIR

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
CHUNK_CHARS = 12000
PAGE_HEADER = re.compile(r'^## Страница (\d+)\s*$', re.M)
CHART_BLOCK = re.compile(r'\[График\].*?\[/График\]', re.S)
CHART_VALUES = re.compile(r'Значения:.*?(?=\nЧто видно|\nИсточник|\n⚠|\[/График\])', re.S)


class Item(BaseModel):
    page: int | None = Field(None, description='номер страницы, где это сказано')
    unverified: bool = Field(False, description='не заполняй: ставит проверка кодом')


class BankView(Item):
    indicator: str
    value: str = Field(description='число или диапазон как в тексте')
    unit: str = ''
    horizon: str = Field('', description='к какому сроку прогноз')
    previous: str = Field('', description='прежний прогноз Банка, если в тексте сказано, как он менялся')
    direction: Literal['up', 'down', 'flat', ''] = ''
    quote: str = Field('', description='короткая дословная цитата')


class KeyFigure(Item):
    indicator: str
    value: str
    unit: str = ''
    period: str = ''
    kind: Literal['actual', 'forecast'] = 'actual'


class Risk(Item):
    text: str


class Sector(Item):
    sector: str
    channel: str = Field(description='через что отчёт связывает сектор с выводом')
    direction: Literal['positive', 'negative', 'mixed', ''] = ''


class ReportDigest(BaseModel):
    summary: str = Field(description='3–5 предложений')
    bank_views: list[BankView] = []
    key_figures: list[KeyFigure] = []
    risks: list[Risk] = []
    sectors: list[Sector] = []
    nuances: list[Risk] = Field([], description='оговорки, условия, расхождения с консенсусом или ЦБ')


SECTIONS = ('bank_views', 'key_figures', 'risks', 'sectors', 'nuances')
ABSENT = re.compile(r'\b(нет|отсутству\w*|не содерж\w*|не привод\w*|не дан\w*)\b', re.I)


def drop_scale_values(text: str) -> str:
    """Values of a chart block read off the scale or flagged by export_text are removed; titles, axes, series stay."""
    def cut(m):
        block = m.group()
        if '≈' not in block and '⚠' not in block:
            return block
        return CHART_VALUES.sub('Значения: не приводятся (сняты моделью со шкалы, ненадёжны)', block)
    return CHART_BLOCK.sub(cut, text)


def split_pages(md: str) -> tuple[str, dict[int, str]]:
    """Header of export_text's Markdown and its pages by number."""
    parts = PAGE_HEADER.split(md)
    return parts[0], {int(n): drop_scale_values(text.strip()) for n, text in zip(parts[1::2], parts[2::2])}


def chunks(pages: dict[int, str], limit: int = CHUNK_CHARS) -> list[dict[int, str]]:
    out, cur, size = [], {}, 0
    for n, text in pages.items():
        if cur and size + len(text) > limit:
            out.append(cur)
            cur, size = {}, 0
        cur[n] = text
        size += len(text)
    return out + [cur] if cur else out


def verify(digest: ReportDigest, pages: dict[int, str]) -> ReportDigest:
    """An item is unverified if it has no known page or one of its numbers is not on that page."""
    for section in SECTIONS:
        for item in getattr(digest, section):
            fields = item.model_dump(exclude={'page', 'unverified'})
            text = ' '.join(str(v) for v in fields.values())
            page = trusted_page(pages.get(item.page, ''))
            if section == 'bank_views':
                page = '\n'.join(s for s in sentences(page) if is_own_view(s))
            said, allowed = quantities(text), quantities(page)
            item.unverified = not page.strip() or any(not any(q.matches(a) for a in allowed) for q in said)
    return digest


def trusted_page(text: str) -> str:
    """Axes and model descriptions are not chart data; keep only pixel-confirmed printed labels as numeric evidence."""
    def chart(m):
        return '\n'.join(ln for ln in m.group().splitlines() if ln.strip().startswith('Подписи на графике (сверены'))
    text = CHART_BLOCK.sub(chart, text)
    return re.sub(r'\|[^|\n]*\(\?\)[^|\n]*(?=\|)', '| [не подтверждено] ', text)


def merge(parts: list[ReportDigest]) -> ReportDigest:
    return ReportDigest(summary=' '.join(p.summary.strip() for p in parts),
                        **{s: [i for p in parts for i in getattr(p, s)] for s in SECTIONS})


def drop_absent(summary: str) -> str:
    """Sentences claiming that something is absent: true of one chunk only, never of the whole report."""
    return ' '.join(s for s in sentences(summary) if not ABSENT.search(s))


def keep_known_numbers(merged: str, parts: list[str]) -> str:
    """The merge may not bring numbers: a sentence with a number none of the parts has is dropped."""
    allowed = quantities(' '.join(parts))
    return ' '.join(s for s in sentences(merged) if all(any(q.matches(a) for a in allowed) for q in quantities(s)))


class CostMeter(BaseCallbackHandler):
    """Sums the proxy's per-call `usage.cost` (₽)."""

    def __init__(self):
        self.cost = 0.0

    def on_llm_end(self, response, **kwargs):
        self.cost += float(((response.llm_output or {}).get('token_usage') or {}).get('cost') or 0)


class BudgetExceeded(Exception):
    pass


class ReportDigester:
    def __init__(self, model=None, budget: float | None = None):
        self.meter, self.budget, self.max_chunk_cost = CostMeter(), budget, 0.0
        agent = ChatModels().agent if model is None else None
        self.chain = PROMPTS.chat('digest') | structured(model or agent(),
                                                         ReportDigest).with_config(callbacks=[self.meter])
        self.merger = (PROMPTS.chat('digest_merge') | (model or agent())).with_config(callbacks=[self.meter])

    def check_budget(self, title: str) -> None:
        if self.budget is not None and self.meter.cost + max(self.max_chunk_cost, 0.1) > self.budget:
            raise BudgetExceeded(f'{title}: spent {self.meter.cost:.2f} of {self.budget} ₽')

    def summary(self, parts: list[str], title: str, force: bool = False) -> str:
        """One summary of the chunk summaries; a single chunk keeps its own unless `force` (an old joined summary)."""
        if len(parts) < 2 and not force:
            return parts[0] if parts else ''
        parts = [drop_absent(p) for p in parts]
        self.check_budget(title)
        text = '\n\n'.join(render_untrusted(f'part{i}', f'сводка части {i}', p) for i, p in enumerate(parts, 1))
        merged = self.merger.invoke({'note': UNTRUSTED_NOTE, 'title': title, 'parts': text}).content
        return keep_known_numbers(str(merged), parts) or ' '.join(parts)

    def digest(self, md: str, title: str = '') -> tuple[ReportDigest, list[str]]:
        _, pages = split_pages(md)
        parts = []
        for chunk in chunks(pages):
            self.check_budget(title)
            before = self.meter.cost
            report = '\n'.join(render_untrusted(f'p{n}', f'{title}, страница {n}', text, body_limit=CHUNK_CHARS + 1000)
                               for n, text in chunk.items())
            answer = self.chain.invoke({'note': UNTRUSTED_NOTE, 'title': title, 'report': report})
            answer.summary = keep_known_numbers(answer.summary, [f'Страница {n}: {trusted_page(text)}' for n, text in chunk.items()])
            parts.append(answer if answer is not None else ReportDigest(summary=''))
            self.max_chunk_cost = max(self.max_chunk_cost, self.meter.cost - before)
        merged, summaries = merge(parts), [p.summary.strip() for p in parts]
        merged.summary = self.summary(summaries, title)
        return verify(merged, pages), summaries


def to_markdown(title: str, d: ReportDigest) -> str:
    mark = lambda i: f' (стр. {i.page if i.page is not None else "?"}){" ⚠ не подтверждено" if i.unverified else ""}'
    lines = [f'# Выжимка: {title}', '', d.summary, '', '## Позиция Банка']
    lines += [f'- {v.indicator}: {v.value} {v.unit}'.rstrip() + (f', {v.horizon}' if v.horizon else '') +
              (f'; прежде {v.previous}' if v.previous else '') + (f'; {v.direction}' if v.direction else '') +
              (f' — «{v.quote}»' if v.quote else '') + mark(v) for v in d.bank_views]
    lines += ['', '## Ключевые цифры']
    lines += [f'- {f.indicator}: {f.value} {f.unit}'.rstrip() + (f', {f.period}' if f.period else '') +
              f' [{"прогноз" if f.kind == "forecast" else "факт"}]' + mark(f) for f in d.key_figures]
    lines += ['', '## Риски'] + [f'- {r.text}{mark(r)}' for r in d.risks]
    lines += ['', '## Отрасли'] + [f'- {s.sector}: {s.channel}' + (f' ({s.direction})' if s.direction else '') + mark(s)
                                    for s in d.sectors]
    lines += ['', '## Оговорки'] + [f'- {n.text}{mark(n)}' for n in d.nuances]
    return '\n'.join(lines) + '\n'


def digest_file(md_path: Path, digester: ReportDigester, merge_only: bool = False) -> ReportDigest:
    """merge_only: take the written digest, redo only the summary merge and the page check (no chunk calls)."""
    md = md_path.read_text(encoding='utf-8')
    title = md.split('\n', 1)[0].lstrip('# ').strip() or md_path.stem
    if merge_only:
        old = json.loads(md_path.with_suffix('.digest.json').read_text(encoding='utf-8'))
        source_hash = old.pop('source_sha256', None)
        if source_hash and source_hash != hashlib.sha256(md.encode()).hexdigest():
            raise ValueError('Report text changed: regenerate the digest without --merge-only')
        summaries = old.pop('chunk_summaries', None) or [old['summary']]  # older digests kept only the joined summary
        d = ReportDigest(**old)
        d.summary = digester.summary(summaries, title, force=True).strip()
        d = verify(d, split_pages(md)[1])
    else:
        d, summaries = digester.digest(md, title)
    md_path.with_suffix('.digest.json').write_text(json.dumps(d.model_dump() | {'chunk_summaries': summaries,
                                                              'source_sha256': hashlib.sha256(md.encode()).hexdigest()}, ensure_ascii=False,
                                                              indent=1), encoding='utf-8')
    md_path.with_suffix('.digest.md').write_text(to_markdown(title, d), encoding='utf-8')
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--text-dir', type=Path, default=TEXT_DIR)
    ap.add_argument('--budget', type=float, default=None, help='stop before the spend of this run passes it, ₽')
    ap.add_argument('--merge-only', action='store_true', help='redo only the summary merge of written digests')
    ap.add_argument('stems', nargs='*')
    a = ap.parse_args()
    files = [a.text_dir / f'{s}.md' for s in a.stems] or sorted(p for p in a.text_dir.glob('*.md') if '.digest' not in p.name)
    digester = ReportDigester(budget=a.budget)
    for f in files:
        before = digester.meter.cost
        try:
            d = digest_file(f, digester, a.merge_only)
        except BudgetExceeded as e:
            print('budget reached:', e)
            break
        print(f.stem[:50], ' '.join(f'{s}={len(getattr(d, s))}/{sum(i.unverified for i in getattr(d, s))}u' for s in SECTIONS),
              f'{digester.meter.cost - before:.2f} ₽', flush=True)
    print(f'spent {digester.meter.cost:.2f} ₽')
