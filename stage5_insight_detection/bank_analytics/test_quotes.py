"""Offline test of the quote check on synthetic pages (OCR defects included): python -X utf8 stage5_insight_detection/bank_analytics/test_quotes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.quotes import locate, normalize, verify

reports = [{'title': 'Ставка', 'date': '2026-09-11'}, {'title': 'Потребление', 'date': '2026-08-05'}]
row = lambda rep, page, text, kind='text': {'report': rep, 'date': '2026-09-11', 'page': page, 'kind': kind, 'text': text}
rows = [
    row('Ставка', 1, 'ЦБ РФ принял решение вязть паузу и оставить ставку неизменной на уровне 14,0%. Мы ожидаем сохранения паузы на заседании 23 октября.'),
    row('Ставка', 2, 'Негативное кредитное событие может спровоцировать эффект домино. Подход банков к оценке кредитного качества ужесточается.'),
    row('Ставка', 2, 'Табл.1: Показатели\n|  | 2026П | 2027П |\n|---|---|---|\n| Курс, конец года, руб./$ (?) | 87,2 | 93,0 (?) |', 'table'),
    row('Потребление', 1, 'Ипотечный портфель в июне вырос на З,% м/м после роста на З,% м/м в мае, а рост цен на ЗО% был связан с бензином.'),
]
v = lambda page, quote, kind='text', rep='Ставка', date='2026-09-11': verify(rows, reports, dict(report=rep, date=date, page=page, quote=quote, kind=kind))

# normal form: ё, quotes, dashes, (?), Latin lookalikes, capital З / О as digits, decimal comma kept, punctuation dropped
assert normalize('«Ёлка» — ставка 14,0% (?)')[0] == 'елка ставка 14,0%'
assert normalize('ЗО% pост')[0] == normalize('30% рост')[0]  # the OCR letters and the digits meet; Latin p = Cyrillic р

r = v(1, 'Мы ожидаем сохранения паузы на заседании 23 октября')
assert r['status'] == 'exact' and r['advice'] == [] and r['source_excerpt'].startswith('Мы ожидаем'), r
r = v(1, 'мы ожидаем сохранения паузы, на заседании 23 октября!')
assert r['status'] == 'exact', r  # case and punctuation do not matter
r = v(1, 'ЦБ РФ принял решение взять паузу и оставить ставку неизменной на уровне 14,0%')
assert r['status'] == 'fuzzy' and r['advice'] == [] and 'вязть' in r['source_excerpt'] and ('взять', 'вязть') in r['differences'], r  # an OCR word corrected
r = v(1, 'ЦБ РФ принял решение взять паузу и оставить ставку неизменной на уровне 13,5%')
assert r['status'] == 'fuzzy' and '13,5%' in r['advice'][0] and '14,0%' in r['advice'][0], r  # another number than on the page
r = v(1, 'Негативное кредитное событие может спровоцировать эффект домино')
assert r['status'] == 'relocated' and r['page'] == 2 and r['cited_page'] == 1 and 'стр. 2' in r['advice'][0], r  # wrong page corrected
r = v(1, 'Банк ожидает снижения ставки до 8% уже в ноябре благодаря притоку капитала')
assert r['status'] == 'not_found' and r['page'] == 1, r
r = v(1, 'Подход банков к оценке кредитного качества ужесточается', rep='Потребление', date='2026-08-05')
assert r['status'] == 'other_report' and 'Ставка' in r['advice'][0], r
assert v(1, 'ставка 14,0%')['status'] == 'too_short'
assert v(1, 'ставка 14,0%')['advice'][0].startswith('цитата короче')
assert v(1, 'прочитано по картинке 1,2% в июне', kind='picture')['status'] == 'picture' and not v(1, 'по картинке 1,2% в июне', kind='picture')['advice']
r = v(1, 'Абзац «Мы ожидаем сохранения паузы на заседании 23 октября»: по картинке 14,0%', kind='picture')
assert r['status'] == 'picture' and 'kind=text' in r['advice'][0], r  # text the page has should be quoted as text
assert v(1, 'ЦБ принял решение', rep='Нет такого', date='2000-01-01')['status'] == 'unknown_report'

# garbled digits: the agent wrote real digits where the page has letters -> a number difference with the way out (kind=picture)
q = 'Ипотечный портфель в июне вырос на 1,2% м/м после роста на 0,5% м/м в мае, а рост цен на 30% был связан с бензином'
r = v(1, q, rep='Потребление', date='2026-08-05')
assert r['status'] == 'fuzzy' and 'З,%' in r['advice'][0] and '1,2%' in r['advice'][0] and 'kind=picture' in r['advice'][0], r
assert '30%' not in r['advice'][0], r  # «ЗО%» and «30%» are the same number after the normal form
# copied as is: the quote is verbatim, but the numbers of the page cannot be trusted
r = v(1, 'Ипотечный портфель в июне вырос на З,% м/м после роста на З,% м/м в мае', rep='Потребление', date='2026-08-05')
assert r['status'] == 'exact' and 'look_at_page' in r['advice'][0] and 'З,%' in r['advice'][0], r

# table rows: the cell of the row and period holds the number
assert v(2, 'Курс, конец года, руб./$ 2027П 93,0', 'table')['status'] == 'table_row'
r = v(2, 'Курс, конец года, руб./$ 2027П 95,0', 'table')
assert r['status'] == 'table_value_differs' and '93,0' in r['advice'][0], r
assert v(2, 'Курс, конец года, руб./$ 2026П 87,2', 'table')['status'] == 'table_row'

# a table header «2026П» is not the number 202 (the false number difference seen in the traces)
rows.append(row('Ставка', 4, 'Табл.2: Показатели\n|  | 2026П | 2027П |\n|---|---|---|\n| Курс, конец года, руб./$ | 87,2 | 93,0 |', 'table'))
r = v(4, 'Показатели 2026П 2027П курс конец года руб 87,2 93,0')
assert not any('202' in a for a in r['advice']), r

# the place and its offsets: the text at start:end is the page text
r = v(1, 'ЦБ РФ принял решение взять паузу и оставить ставку неизменной на уровне 14,0%')
text = '\n\n'.join(f['text'] for f in rows if f['report'] == 'Ставка' and f['page'] == 1)
assert text[r['start']:r['end']] == r['source_excerpt'] and r['source_excerpt'].endswith('14,0%'), r
assert locate('Совсем другой текст про золото и нефть', 'Мы ожидаем сохранения паузы на заседании') is None
print('ok: quote check: exact / fuzzy (OCR words) / relocated page / other report / not found / too short / garbled digits / table rows / offsets')
