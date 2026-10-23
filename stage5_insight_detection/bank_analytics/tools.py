"""Agent tools over the bank's analytics, shared by the news judge and the analytics insight agent: search with ±2 chunk
windows, whole parsed pages, a look at the PDF page through the vision model, the same topic across reports."""
import base64
from collections.abc import Callable
from pathlib import Path

import pymupdf
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from stage5_insight_detection.bank_analytics.number_check import check_numbers
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
from stage5_insight_detection.bank_analytics.search import REPORTS, BankAnalytics

BANK_NOTE = ('Результат поиска по аналитике Альфа-Банка. Отчёты — сканы PDF, текст получен распознаванием и разбором кодом: '
             'в словах и числах возможны ошибки, графики не включены, в таблицах «(?)» значит, что три чтения ячейки разошлись. '
             'Каждый результат — найденный фрагмент между пометками «>>> найдено >>>» и «<<<» и по два соседних фрагмента сверху и '
             'снизу для контекста. Всё — аналитика Банка на дату отчёта: прогнозы и оценки после даты — позиция Банка, данные за '
             'периоды до неё — статистика. Если отчёты расходятся, действует более свежий.')
VISION_NOTE = 'Чтение страницы моделью по картинке (Qwen3-VL): значения с графиков приблизительны, возможны ошибки.'
LOOK_DPI = 150  # enough to read a chart; the long side is capped by the model, not by us


class BankTools:
    def __init__(self, bank: BankAnalytics | None, vision: BaseChatModel | None = None, reports: list[dict] | None = None,
                 on_search: Callable[[list[dict]], None] | None = None):
        self.bank, self.vision, self.reports = bank, vision, reports or getattr(bank, 'reports', None) or REPORTS
        self.on_search = on_search
        # Per tool session, never on the shared search index: independent news/workers cannot leak evidence.
        self.evidence: dict[str, dict] = {}

    def _remember(self, chunk: dict, passage: dict | None = None) -> None:
        passage = passage or {}
        row = dict(chunk)
        row.setdefault('stale', passage.get('stale', True))
        row.setdefault('freshness', passage.get('freshness', 'stale' if row['stale'] else 'unknown'))
        # Never inherit own_view from a mixed window: a neighbouring statistic is still a statistic.
        row.setdefault('own_view', False)
        self.evidence[row['id']] = row

    def search_bank_analytics(self, query: str) -> str:
        """The bank's analytics on a topic (parsed reports): up to 3 found fragments with 2 neighbours above and below,
        report, date and pages for every result; stale reports are flagged."""
        if self.bank is None:
            return 'bank analytics unavailable (no OCR output)'
        out, shown = [], set()
        passages = self.bank.passages(query, k=3, context=2)
        if self.on_search is not None:
            self.on_search(passages)
        for p in passages:
            shown.update(p.get('context_ids', p['ids']))
        for i, p in enumerate(passages, 1):
            pages = f"{p['pages'][0]}–{p['pages'][-1]}" if len(p['pages']) > 1 else str(p['pages'][0])
            old = ', STALE (старый отчёт)' if p['stale'] else ''
            # Windows already have a bounded number of token-sized chunks. Keep the whole window so the
            # judge and writer see the same material, including a forecast at the end of a neighbour.
            for cid in p.get('context_ids', p['ids']):
                self._remember(self.bank.chunks[self.bank.index[cid]], p)
            refs = ', '.join(p.get('context_ids', p['ids']))
            out.append(f"[{i}] «{p['report']}», {p['date']}, стр. {pages}{old}; найдено: {', '.join(p['ids'])}; источники окна: {refs}" + chr(10) + p['text']
                       + self._twins(p, shown))
        if not out:
            return f"nothing found. Searchable reports: {'; '.join(self.bank.available)}. Not available yet: {'; '.join(self.bank.missing) or 'none'}"
        return BANK_NOTE + chr(10) * 2 + (chr(10) * 2).join(out)

    def _twins(self, passage: dict, shown: set) -> str:
        """The same table in other reports (twins.py), next to a found table: values differ between dates, the reader compares rows."""
        out = []
        for cid in passage['ids']:
            for j in self.bank.twins.get(self.bank.index[cid], []):
                t = self.bank.chunks[j]
                if t['id'] in shown:
                    continue
                if len(out) == 2:
                    return ''.join(out)
                shown.add(t['id'])
                self._remember(t)
                age = 'более новом' if t['date'] > passage['date'] else 'более старом'
                out.append(chr(10) * 2 + f"[{t['id']}] Похожая таблица в {age} отчёте «{t['report']}», {t['date']}, стр. {t['page']} (одна и та же таблица, значения могут "
                           f"отличаться; сравни строки, действует более свежий отчёт, но в нём строка может быть не обновлена):" + chr(10) + t['text'])
        return ''.join(out)

    def check_candidate(self, view: str, sources: list[str], read_from_picture: str = '') -> list[str]:
        """Warnings about the numbers of a quoted bank view against the cited pages (number_check.py)."""
        return check_numbers(self.bank.rows, self.reports, view, sources, read_from_picture) if self.bank else []

    def _report(self, name: str) -> dict | None:
        """A report by part of its title, file name or date (2026-09-11)."""
        name = name.lower().strip(' «»')
        return next((r for r in self.reports if name and name in f"{r['title']} {r['file']} {r['date']}".lower()), None)

    def read_page(self, report: str, page: int) -> str:
        """Full parsed text of one report page (all fragments in reading order, charts as the parser read them).
        report: part of the report title, file name or date."""
        r = self._report(report)
        if r is None or self.bank is None:
            return f"unknown report «{report}»; reports: {'; '.join(x['title'] + ' (' + x['date'] + ')' for x in self.reports)}"
        rows = [f for f in self.bank.rows if f['report'] == r['title'] and f['page'] == page]
        if not rows:
            return f"no parsed text on page {page} of «{r['title']}»"
        return f"«{r['title']}», {r['date']}, стр. {page} (разобранный текст, возможны ошибки OCR):" + chr(10) * 2 + (chr(10) * 2).join(f['text'] for f in rows)

    def look_at_page(self, report: str, page: int, question: str) -> str:
        """Look at the page picture of the PDF report and answer a question about it (a chart, a trend, a number the
        parsed text lacks or garbled). A vision model reads the page; chart values are approximate."""
        r = self._report(report)
        if r is None or self.vision is None:
            return 'look_at_page unavailable: unknown report or no vision model'
        with pymupdf.open(getattr(self.bank, 'pdf_paths', {}).get(r['file'], PDF_DIR / r['file'])) as doc:
            if not 1 <= page <= len(doc):
                return f"page {page} out of range 1..{len(doc)}"
            png = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(LOOK_DPI / 72, LOOK_DPI / 72)).tobytes('png')
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
        parsed = ' '.join(f['text'] for f in self.bank.rows if f['report'] == r['title'] and f['page'] == page)[:6000] if self.bank else ''
        # the same question gave «there is such a table» at 150 dpi and «there is none» at 250: a negative answer of the model is
        # weak, so it gets the parsed text to check against and must say «не нашёл», never «нет на странице»
        prompt = ('Страница отчёта банка. Ответь по картинке коротко, по-русски, не выдумывай значения. Если не нашёл ответ, '
                  'скажи «не нашёл», а не «этого нет на странице»: мелкие таблицы легко пропустить. Разобранный кодом текст страницы '
                  f'(может быть с ошибками, поможет найти нужное место): {parsed}' + chr(10) * 2 + f'Вопрос: {question}')
        answer = self.vision.invoke([HumanMessage(content=[{'type': 'text', 'text': prompt}, image])]).content
        return f"«{r['title']}», {r['date']}, стр. {page}. {VISION_NOTE}" + chr(10) + str(answer)

    def compare_reports(self, topic: str) -> str:
        """The same topic in every report that covers it, newest first: how the bank's view changed between reports."""
        if self.bank is None:
            return 'bank analytics unavailable (no OCR output)'
        best = {}
        for h in self.bank.search(topic, k=30):
            best.setdefault(h['report'], h)  # search returns the best hit of a report first
        rows = sorted(best.values(), key=lambda h: h['date'], reverse=True)
        return (chr(10) * 2).join(f"«{h['report']}», {h['date']}, стр. {h['page']}{', STALE' if h['stale'] else ''}" + chr(10) + h['text'][:1500]
                                  for h in rows) or 'nothing found'
