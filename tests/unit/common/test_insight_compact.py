"""Screen version of an insight (24.10): code checks catch confidential company data and new numbers without flagging
news wording; sources come from the cited insight items, also when the model gave no ids."""
import pytest

from stage6_insight_generation.insight_compact.compact import COMPANY_DATA, RANK, CompactInsight, problems, with_sources

INSIGHT = {'what_happened': [{'id': 'e1', 'text': 'ДОМ.РФ разместил облигации на 10 млрд ₽, отсрочка платежей на 12 месяцев',
                              'sources': ['news:1']}],
           'why_it_matters': [{'id': 'w1', 'text': 'Гослизингодатели финансируют сделки дешевле рынка', 'sources': ['fact:3']}],
           'bank_view': [{'id': 'b1', 'text': 'Банк не видит драйверов снижения доходностей ОФЗ', 'sources': ['bank:r:p1']}]}


def compact(**kw):
    base = dict(title='t', lead='l', facts=[{'ids': ['e1'], 'text': 'ДОМ.РФ разместил облигации на 10 млрд ₽'}],
                why=[{'ids': ['w1'], 'label': 'Дешевле рынка', 'text': 'Сделки могут уходить к госигрокам'}],
                important_if=['Портфель компании — автотранспорт', 'Долг компании плавающий'])
    return with_sources(CompactInsight(**{**base, **kw}), INSIGHT)


def test_clean_compact_passes_and_news_wording_is_not_flagged():
    assert problems(compact(actions=['Проверить контрагентов до конца октября',
                                     'Учесть отсрочку платежей на 12 месяцев']), INSIGHT) == []


def test_ranks_company_figures_and_new_numbers_are_flagged():
    found = problems(compact(lead='Компания — 1-е место среди 55 компаний', actions=['73% исходящих платежей — автосервис']), INSIGHT)
    assert sum('конфиденциальные' in p for p in found) == 2 and any('числа 73' in p for p in found)


def test_uncovered_fact_is_flagged():
    assert any('e2' in p for p in problems(compact(facts=[{'ids': [], 'text': 'Новость'}]), {**INSIGHT, 'what_happened': [
        *INSIGHT['what_happened'], {'id': 'e2', 'text': 'Другое событие', 'sources': []}]}))


def test_plain_string_lines_get_ids_and_sources_from_the_closest_item():
    out = compact(bank_view=['Банк не видит драйверов снижения доходностей'])
    assert out['bank_view'][0]['ids'] == ['b1'] and out['bank_view'][0]['sources'] == ['bank:r:p1']
    assert out['why'][0]['sources'] == ['fact:3']


@pytest.mark.parametrize('text, leak', [
    ('Компания — 1-е место среди 55 компаний', True),
    ('73% исходящих платежей приходится на автосервис', True),
    ('Входящие платежи из региона уже составляют 170 млн ₽', True),
    ('Договоры с контрагентами-перевозчиками на 5,7 млрд ₽ в год', True),
    ('Отсрочка лизинговых платежей на 12 месяцев', False),
    ('Рост выручки в 2027 году придётся зарабатывать долей рынка', False),
    ('Отказы по автокредитам 69,3% сократят поток новых договоров и выручку', False),
    ('Проверить внешнеторговых контрагентов по спискам 22-го пакета', False),
])
def test_confidential_company_data(text, leak):
    assert bool(RANK.search(text) or COMPANY_DATA.search(text)) == leak
