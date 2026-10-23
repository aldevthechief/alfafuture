"""Offline check of selective chart processing (chart_novelty): the text about a chart, what it covers, the verdicts,
the forecast-first order, the three modes, the text export."""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics import chart_novelty as cn
from stage5_insight_detection.bank_analytics.export_text import render_page


def record(title, printed=(), tesseract=(), last=None, crop='Test_report_p2_1.png'):
    """A charts.py record: the model's printed labels, confirmed when Tesseract read the same number."""
    series = [{'name': 'ряд', 'printed_values': [{'x': x, 'value': v} for x, v in printed]}]
    dg = {'ok': True, 'x_labels': [], 'step': 1.0, 'series': [{'name': 'ряд', 'values': [], 'last': last}]} if last else {}
    return {'crop': crop, 'title': title, 'read': {'title': title, 'series': series}, 'digitized': dg,
            'code': {'numbers': list(tesseract)}}


# numbers of a text: precision, years, thousands, sign, values measured by code
nums = cn.text_numbers('В 2026 г. инфляция 6,32%, снижение на 0,2 п.п., выручка 69 572 млн руб.')
assert 2026 not in nums and 69572 in nums
assert cn.in_text('6,3', nums) and cn.in_text('−0,2', nums) and not cn.in_text('6,4', nums) and not cn.in_text('6,30', nums)
assert cn.in_text('≈ 6,4', nums) and not cn.in_text('≈ 7', nums)
assert cn.period_tokens('авг.26') >= {'август', '2026'} and '2к26' in cn.period_tokens('2К26')
assert 'ммк' in cn.stems('MMK') and 'динам' not in cn.stems('Динамика ВВП')

# the text about a chart: caption, references «см Рис.N» anywhere, paragraphs of its page and the page before
page1 = ('Инфляция замедлилась до 6,32% г/г в августе (см Рис. 1).\n\nКлючевая ставка 16%, мы ожидаем снижения ставки '
         'до 14% к концу 2026 г.\n\nПродажи автомобилей выросли.')
page2 = ('Рис. 1: Инфляция, % г/г\n[График] Рис. 1: Инфляция, % г/г\nЗначения: 6,3\n[/График]\n\n'
         'Рис. 2: Ключевая ставка, %\n[График] Рис. 2: Ключевая ставка, %\nЗначения: 16; 14 (2026П); 9,5 (2027П)\n[/График]\n\n'
         'Рис. 3: Курс юаня\n[График] Рис. 3: Курс юаня\nЗначения: не определить\n[/График]')
report = cn.Report('Test_report', [page1, page2], tables=False)
rel = report.related('Рис. 1: Инфляция, % г/г', 2)
assert rel['caption'] and rel['refs'] == 1 and 6.32 in rel['numbers'] and 16 not in rel['numbers']
assert report.related('Рис. 2: Ключевая ставка, %', 2)['paragraphs'] == 1
assert report.related('Рис. 3: Курс юаня', 2)['paragraphs'] == 0

# verdicts and what is covered
dup = record('Рис. 1: Инфляция, % г/г', [('авг.26', '6,3')], ['6,3'])
new = record('Рис. 2: Ключевая ставка, %', [('авг.26', '16'), ('июл.26', '18')], ['16', '18'])
ahead = record('Рис. 2: Ключевая ставка, %', [('2027П', '9,5')], ['9,5'])
told = record('Рис. 2: Ключевая ставка, %', [('2026П', '14')], ['14'])
alone = record('Рис. 3: Курс юаня', last=12.4, crop='Test_report_p2_3.png')
shape = record('Рис. 1: Инфляция, % г/г', last=6.3)
v = cn.novelty(dup, report, 2)
assert v['kind'] == 'duplicate' and v['coverage']['indicator'] and v['coverage']['period'] and v['coverage']['values']
v = cn.novelty(new, report, 2)
assert v['kind'] == 'new_values' and v['missing'] == ['18'] and v['priority'] == 1
v = cn.novelty(ahead, report, 2)
assert v['kind'] == 'forecast' and v['priority'] == 0 and v['coverage']['forecast'] is False
assert cn.novelty(told, report, 2)['kind'] == 'duplicate'  # «мы ожидаем ... до 14% к концу 2026 г.»
assert cn.novelty(alone, report, 2)['kind'] == 'only_chart' and cn.novelty(alone, report, 2)['new']
assert cn.novelty(shape, report, 2)['kind'] == 'history' and not cn.novelty(shape, report, 2)['new']
assert cn.novelty({'crop': 'x_p2_1.png', 'read': {'error': 'not JSON'}}, report, 2)['kind'] == 'empty'
caveat = record('Рис. 1: Инфляция*, % г/г', [('авг.26', '6,3')], ['6,3'])
assert cn.novelty(caveat, report, 2)['coverage']['caveats'] is False and 'оговорки' in cn.novelty(caveat, report, 2)['reason']

# before reading: the page OCR's blocks; a forecast first, then values, then what the text never names
facts = [cn.block_facts(b, 2026) for b in cn.BLOCK.findall(page2)]
assert facts[1]['values'] == ['16'] and facts[1]['forecast'] == ['14', '9,5'] and facts[1]['forecast_mark']
blocks = [cn.verdict(f, report.related(f['title'], 2)) for f in facts]
assert blocks[1]['missing_forecast'] == ['9,5']  # 14 is in the text («до 14% к концу 2026 г.»)
unknown = cn.block_facts('[График] Рис. 2: Ключевая ставка в 2026-2027 гг., %\nЗначения: 16\n[/График]', 2026)
assert unknown['forecast_mark'] and cn.verdict(unknown, report.related(unknown['title'], 2))['kind'] == 'forecast'
assert [b['kind'] for b in blocks] == ['duplicate', 'forecast', 'only_chart']
assert cn.page_priority(blocks) == 0 and cn.page_priority(blocks[:1]) == 9 and cn.page_priority(None) == 1
assert cn.worth_reading('Рис. 1: Инфляция, % г/г', blocks) is False and cn.worth_reading('Рис. 2: Ключевая ставка, %', blocks)
assert cn.worth_reading('Курс рубля к доллару', blocks) and cn.worth_reading('x', None)

# modes
os.environ.pop('BANK_CHARTS', None)
assert cn.mode_of() == 'all' and cn.mode_of('Selective') == 'selective'
os.environ['BANK_CHARTS'] = 'off'
assert cn.mode_of() == 'off' and cn.mode_of('all') == 'all'
os.environ.pop('BANK_CHARTS')
try:
    cn.mode_of('some')
    raise AssertionError('a wrong mode must fail')
except ValueError:
    pass

assert cn.select(page2, [dup, new], report, 2, 'all') == (page2, [dup, new])
out, keep = cn.select(page2, [dup, new], report, 2, 'selective')
assert keep == [new] and '[График] Рис. 2' in out and '[График] Рис. 1' not in out
assert f'{cn.SKIPPED} Рис. 1: Инфляция, % г/г: значения графика уже есть в связанном тексте.' in out
out, keep = cn.select(page2, [dup, new], report, 2, 'off')
assert keep == [] and '[График]' not in out and out.count(cn.SKIPPED) == 3

# the text export: the chart note is added only to the charts that are kept
md, _ = render_page(2, page2, charts=[dup, new], mode='selective', report=report)
assert md.count('Описание графика (составлено кодом)') == 1 and cn.SKIPPED in md
md, _ = render_page(2, page2, charts=[dup, new])
assert md.count('Описание графика (составлено кодом)') == 2 and cn.SKIPPED not in md
print('ok')
