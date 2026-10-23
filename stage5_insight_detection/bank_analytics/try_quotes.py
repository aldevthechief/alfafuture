"""Eyeball run of quotes.verify on real report pages (data/analytics/fragments.jsonl): python -X utf8 stage5_insight_detection/bank_analytics/try_quotes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.quotes import verify
from stage5_insight_detection.bank_analytics.search import REPORTS, BankAnalytics

R1 = 'Ключевая ставка ЦБ РФ: без изменения на 14,0%'
TACT = 'Тактический взгляд на 3-й квартал 2026 (долговой рынок)'
CASES = [
    ('exact', dict(report=R1, date='2026-09-11', page=1, quote='Мы ожидаем сохранения паузы на заседании 23 октября')),
    ('relocated page', dict(report=TACT, date='2026-07-22', page=21, quote='Негативное кредитное событие может спровоцировать эффект домино')),
    ('ocr words fixed', dict(report=R1, date='2026-09-11', page=1, quote='ЦБ РФ принял решение взять паузу и оставить ставку неизменной на уровне 14,0%')),
    ('wrong number', dict(report=R1, date='2026-09-11', page=1, quote='ЦБ РФ принял решение взять паузу и оставить ставку неизменной на уровне 13,5%')),
    ('garbled digits, agent wrote real digits', dict(report='Потребление на пике', date='2026-08-05', page=2, quote='ипотечный портфель в июне вырос на 1,2% м/м после роста на 0,5% м/м в мае на фоне скачка спроса населения на «семейную» ипотеку')),
    ('garbled digits copied as is', dict(report='Потребление на пике', date='2026-08-05', page=2, quote='ипотечный портфель в июне вырос на З,% м/м после роста на З,% м/м в мае на фоне скачка спроса населения на «семейную» ипотеку')),
    ('table ok', dict(report=R1, date='2026-09-11', page=2, quote='Курс, конец года, руб./$ 2027П 93,0', kind='table')),
    ('table wrong value', dict(report=R1, date='2026-09-11', page=2, quote='Курс, конец года, руб./$ 2027П 95,0', kind='table')),
    ('table, no period', dict(report=R1, date='2026-09-11', page=2, quote='Ставка, конец года, 14,0%', kind='table')),
    ('too short', dict(report=R1, date='2026-09-11', page=1, quote='ставка 14,0%')),
    ('invented', dict(report=R1, date='2026-09-11', page=1, quote='Банк ожидает снижения ставки до 8% уже в ноябре 2026 года благодаря притоку капитала')),
    ('other report', dict(report='Макростатистика 1П26', date='2026-07-31', page=1, quote='Мы ожидаем сохранения паузы на заседании 23 октября')),
    ('picture', dict(report='Потребление на пике', date='2026-08-05', page=3, quote='июнь 2026 ≈1,2% м/м', kind='picture')),
]

if __name__ == '__main__':
    rows = BankAnalytics(chunk_tokens=0).rows
    for name, ev in CASES:
        r = verify(rows, REPORTS, ev)
        print(f"{name}: {r['status']} page {r['cited_page']}->{r['page']} words-diff={r['differences'][:3]}\n    advice={[a[:260] for a in r['advice']]}\n    excerpt={r['source_excerpt'][:100]!r}")
