"""Opt-in PDF preparation before the analyst loop: full OCR -> charts/tables -> fragments -> report digests."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from common.llm import ChatModels, StructuredOutputError
from common.news_corpus import write_json
from common.trace import AgentTrace
from stage5_insight_detection.bank_analytics.chart_novelty import mode_of
from stage5_insight_detection.bank_analytics.charts import read_pages as read_charts
from stage5_insight_detection.bank_analytics.digest import (
    BudgetExceeded, CostMeter, PROMPTS, ReportDigest, ReportDigester, digest_file,
)
from stage5_insight_detection.bank_analytics.export_text import export
from stage5_insight_detection.bank_analytics.library import BankLibrary, text_layer
from stage5_insight_detection.bank_analytics.ocr import AnalyticsOCR, BudgetExceeded as OCRBudgetExceeded, PDF_DIR
from stage5_insight_detection.bank_analytics.tables import run_pages as read_tables


@dataclass(frozen=True)
class PreparationOptions:
    ocr: bool = False
    charts: bool = False
    tables: bool = False
    digests: bool = False
    ocr_budget: float = 15.0
    charts_budget: float = 8.0
    tables_budget: float = 7.0
    digest_budget: float = 3.0
    chart_mode: str | None = None

    def __post_init__(self):
        # Freeze the environment fallback in the recipe, so queued jobs and cache keys keep the same mode.
        object.__setattr__(self, 'chart_mode', mode_of(self.chart_mode))
        for name in ('ocr_budget', 'charts_budget', 'tables_budget', 'digest_budget'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')

    @property
    def enabled(self):
        return self.ocr or self.charts or self.tables or self.digests


@dataclass
class PreparedBank:
    library: BankLibrary
    report: dict


def check_local_ocr() -> None:
    """Check the actual OCR backend before spending on vision calls; never install/download implicitly."""
    from PIL import Image
    from common.ocr_local import word_data
    word_data(Image.new('RGB', (64, 64), 'white'))


def prepare_bank(options: PreparationOptions, models: ChatModels, *, pdf_dir: Path = PDF_DIR,
                 trace: AgentTrace | None = None) -> PreparedBank:
    """Budgets are per stage for this run; partial results stay usable, with an explicit preparation report."""
    trace = trace or AgentTrace(folder=None)
    pdf_dir = Path(pdf_dir)
    report = {'status': 'complete', 'options': asdict(options), 'reports': 0, 'stages': []}

    def refresh():
        return BankLibrary.refresh(pdf_dir=pdf_dir, ocr_dir=pdf_dir / 'ocr', full_dir=pdf_dir / 'ocr_full',
                                   out=pdf_dir / 'fragments.jsonl', registry=pdf_dir / 'registry.json',
                                   charts=options.chart_mode)

    if not options.enabled:
        return PreparedBank(refresh(), dict(report, status='disabled'))
    pdfs = sorted(pdf_dir.glob('*.pdf'))
    report['reports'] = len(pdfs)
    if not pdfs:
        report.update(status='empty', message='No bank PDFs found')
        library = refresh()
        write_json(pdf_dir / 'preparation.json', report)
        trace.log('bank_preparation', **report)
        return PreparedBank(library, report)
    if (options.charts and options.chart_mode != 'off') or options.tables:
        check_local_ocr()
    callbacks = list(getattr(models, 'callbacks', None) or [])

    def stage(name, budget, work, cost):
        record = {'name': name, 'status': 'complete', 'budget': budget}
        print(f'Bank preparation: {name}', flush=True)
        try:
            details = work() or {}
            record.update(details)
            if details.get('errors'):
                record['status'] = 'partial'
        except (BudgetExceeded, OCRBudgetExceeded) as error:
            record.update(status='partial', reason='budget', message=str(error))
        record['cost'] = round(cost(), 4)
        report['stages'].append(record)
        if record['status'] != 'complete':
            report['status'] = 'partial'
            print(f'Bank preparation: {name}: {record.get("message") or record.get("errors")}', flush=True)
        trace.log('bank_preparation_stage', **record)

    ocr = AnalyticsOCR(model=models.vision().model_copy(update={'frequency_penalty': 0.2, 'max_retries': 0}),
                       out_dir=pdf_dir / 'ocr_full', prompt='ocr_page_full', budget=options.ocr_budget)
    all_pages = []
    scanned = {}
    for pdf in pdfs:
        layer, _ = text_layer(pdf)
        all_pages.extend((pdf.stem, n) for n in range(1, len(layer) + 1))
        scanned[pdf] = [n for n, text in enumerate(layer, 1) if len(text.strip()) < 50]

    def full_ocr():
        for pdf in pdfs:
            if scanned[pdf]:
                ocr.transcribe(pdf, only_pages=scanned[pdf])
        return {'pages': sum(map(len, scanned.values())), 'errors': [f'{name}:p{n}' for name, n in ocr.skipped]}
    stage('ocr', options.ocr_budget, full_ocr, lambda: ocr.cost)
    library = refresh()
    likely_charts = {(Path(f.file).stem, f.page) for f in library.fragments if f.kind == 'chart'}
    likely_tables = {(Path(f.file).stem, f.page) for f in library.fragments if f.kind == 'table'}
    if options.charts:
        meter = CostMeter()
        pages = sorted(all_pages, key=lambda pg: pg not in likely_charts)
        stage('charts', options.charts_budget,
              lambda: read_charts(pages, pdf_dir=pdf_dir, out_dir=pdf_dir / 'charts',
                                  budget=options.charts_budget, callbacks=[*callbacks, meter],
                                  charts=options.chart_mode), lambda: meter.cost)
    if options.tables:
        meter = CostMeter()
        pages = sorted(all_pages, key=lambda pg: pg not in likely_tables)
        def tables():
            rows = read_tables(pages, options.tables_budget, likely_tables, pdf_dir=pdf_dir,
                               out_dir=pdf_dir / 'tables', callbacks=[*callbacks, meter])
            done = {(r['stem'], r['page']) for r in rows}
            return {'pages': len(rows), 'errors': [f'{stem}:p{n}' for stem, n in pages if (stem, n) not in done]}
        stage('tables', options.tables_budget, tables, lambda: meter.cost)
    library = refresh()  # newly checked cells and printed/approximate chart values reach this very run
    if options.digests:
        text_dir = pdf_dir / 'text'
        for pdf in pdfs:
            export(pdf, pdf_dir / 'ocr_full', text_dir, pdf_dir / 'ocr', charts=options.chart_mode)
        recipe = hashlib.sha256((PROMPTS.text('digest') + PROMPTS.text('digest_merge') +
                                 json.dumps(ReportDigest.model_json_schema(), sort_keys=True) +
                                 str(getattr(getattr(models, 'settings', None), 'agent_model', 'provided-model'))).encode()).hexdigest()
        digester = None
        def digests():
            nonlocal digester
            done, cached, errors = 0, 0, []
            for pdf in pdfs:
                path = text_dir / f'{pdf.stem}.md'
                source_hash = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
                output = path.with_suffix('.digest.json')
                if output.exists():
                    try:
                        old = json.loads(output.read_text(encoding='utf-8'))
                        ReportDigest.model_validate(old)
                        if old.get('source_sha256') == source_hash and old.get('preparation_recipe') == recipe:
                            cached += 1
                            continue
                    except ValueError:
                        pass
                if digester is None:
                    digester = ReportDigester(models.agent(),
                                              budget=options.digest_budget)
                try:
                    digest_file(path, digester)
                except StructuredOutputError as error:
                    errors.append(f'{pdf.name}: {error}')
                    continue
                payload = json.loads(output.read_text(encoding='utf-8'))
                write_json(output, dict(payload, preparation_recipe=recipe))
                done += 1
            return {'reports': done, 'cached_reports': cached, 'errors': errors}
        stage('digests', options.digest_budget, digests, lambda: digester.meter.cost if digester else 0)
    report['fragments'] = len(library.fragments)
    write_json(pdf_dir / 'preparation.json', report)
    trace.log('bank_preparation', **report)
    return PreparedBank(library, report)
