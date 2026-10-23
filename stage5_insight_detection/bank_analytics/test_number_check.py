"""Offline test of the number check (parsing rules, wording, stale table value). python -X utf8 stage5_insight_detection/bank_analytics/test_number_check.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.number_check import check_numbers, numbers

n = numbers('Ставка 14,0% (было 13.5%), курс 95 руб./$; −4,0% г/г; 1 200 млрд; 11.09.2026; 4К26, 2027П, в 2026 г.; 2 заседания; 12 п.п.; 2,1–2,3')
assert set(n) == {'14', '13.5', '95', '4', '1200', '2.1', '2.3', '12'}, n  # years, dates, tags, bare small integers are not numbers; the sign is ignored
assert not numbers('(Макростатистика 1П26 от 31.07, отчёт от 07.09)') and '14' in numbers('ставка 14.00% и 13,5')
assert not numbers('на 31 июля и с 1 по 5 августа, заседание 23 октября') and '14' in numbers('ставка 14% с 23 октября')
assert not numbers('отчёт (2026-07-31), Потребление (2026-08-05)') and '14' in numbers('(2026-09-11) ставка 14%')
assert numbers('12,0%') == {'12': '12,0'} and numbers('12') == {'12': '12'} and '12' in numbers('12 %')

reports = [{'title': 'Макростатистика 1П26', 'date': '2026-07-31'}, {'title': 'Ключевая ставка ЦБ РФ', 'date': '2026-09-11'}]
table = lambda rep, date, fx: {'report': rep, 'date': date, 'page': 4, 'kind': 'table',
                               'text': f'Табл.2: Основные показатели\n|  | 2026П | 2027П |\n|---|---|---|\n| Курс, конец года, руб./$ (?) | 87,2 | {fx} |'}
rows = [table('Макростатистика 1П26', '2026-07-31', '95,0'), {**table('Ключевая ставка ЦБ РФ', '2026-09-11', '93,0 (?)'), 'page': 2},
        {'report': 'Макростатистика 1П26', 'date': '2026-07-31', 'page': 1, 'kind': 'text', 'text': 'Банк ждёт инфляцию 7,5% и рост ВВП 0,1%.'}]
src = ['Макростатистика 1П26, 2026-07-31, стр. 4']
w = check_numbers(rows, reports, 'Курс на конец 2027 г. — 95,0 руб./$', src)
assert len(w) == 1 and '93,0' in w[0] and 'более свежем' in w[0], w  # the old table value
assert check_numbers(rows, reports, 'Курс 2027: 93 руб./$ (было 95)', src + ['11.09.2026, стр. 2']) == []  # both values quoted, both cited
w = check_numbers(rows, reports, 'Инфляция 7,6%', ['Макростатистика 1П26, 31.07.2026, стр. 1'])
assert len(w) == 1 and 'числа 7,6'.replace('числа ', 'число ') in w[0] and 'нет в тексте' in w[0], w  # 7,6 is not on the page
assert check_numbers(rows, reports, 'Инфляция 7.5% и ВВП 0,1%', ['Макростатистика 1П26, 2026-07-31, стр. 1']) == []  # 7,5 = 7.5
assert 'не удалось разобрать' in check_numbers(rows, reports, 'Ставка 14%', ['где-то в отчёте'])[0]
print('ok: number parsing (comma/dot, sign, dates, years, tags), missing number, stale table value, unreadable source')
