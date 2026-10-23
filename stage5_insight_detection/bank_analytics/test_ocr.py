"""Offline check of how chart descriptions are put into the page text."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.ocr import looped, merge

page = 'Табл.1\n\nРис. 1: Инфляция,\n% н/н\nИсточники: Росстат\n\nТекст.\nРис. 2: Ставка, %\nИсточники: ЦБ РФ'
out = merge(page, [('Рис. 2: Ставка, %', 'Столбцы.\n\nСтавка 14,0%.'), ('Рис. 1: Инфляция, % н/н', 'Линия'),
                   ('Рис. 9: Нефть Brent', 'Brent')])
assert 'Рис. 1: Инфляция,\n% н/н\nИсточники: Росстат\n\n[Рисунок] Линия\n\nТекст.' in out, out  # wrapped title
assert 'Источники: ЦБ РФ\n\n[Рисунок] Столбцы.\nСтавка 14,0%.' in out, out  # title mid-paragraph, blank lines collapsed
assert out.endswith('\n\n[Рисунок] Brent'), out  # no paragraph with its title -> page end
assert looped('шапка\n' + '-7,8 трлн\n' * 12) and not looped('Источники: ЦБ РФ\n' * 6)
print('ok')
