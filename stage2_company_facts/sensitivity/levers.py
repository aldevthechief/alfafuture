"""Company sensitivity to the fixed list of levers (the 21 news channels A1-H1 of experiments/news_channels/channels.py).

A news item is scored once against the same levers (strength 0..1); importance for a company = sum over levers of
news strength x company sensitivity. The company side is built here from the profile only (no news, no LLM), so it
works for any news corpus:
- every external factor of the profile is mapped onto a lever (label/keyword rules first, then its topic);
- company rules: bank products (rate, financing, payment terms), FTS countries (currency, sanctions, export markets),
  ownership, headcount, scale.
Sensitivity per lever = max of the contributions; `reasons` keep the factor and fact ids behind it.
"""
import re

LEVERS = {  # code -> short name; full wording in experiments/news_channels/channels.py ITEMS
    'A1': 'Потребительский спрос', 'A2': 'Инвестиционный спрос компаний', 'A3': 'Госрасходы и госзаказ',
    'A4': 'Внешние рынки сбыта', 'B1': 'Цены на товары и сырьё', 'B2': 'Энергия и регулируемые тарифы',
    'B3': 'Логистика', 'B4': 'Труд', 'C1': 'Общие налоги и взносы', 'C2': 'Отраслевые сборы, пошлины, акцизы',
    'C3': 'Отраслевое регулирование', 'C4': 'Господдержка', 'C5': 'Собственность и корпоративное право',
    'D1': 'Ставки и стоимость кредита', 'D2': 'Доступ к финансированию', 'D3': 'Валютный курс и валютное регулирование',
    'D4': 'Платёжная дисциплина и оборотный капитал', 'E1': 'Санкции и внешние ограничения',
    'F1': 'Конкуренция и структура рынка', 'G1': 'Сбои и форс-мажор, погода', 'H1': 'Технологии и новые рынки',
}
# factor label / keywords -> lever, checked in order; the first match wins, otherwise TOPIC decides
LABEL_RULES = [
    (r'пошлин|акциз|утильсбор|демпфер|сбор', 'C2'), (r'санкц|недружеств', 'E1'), (r'иностранн', 'C5'),
    (r'урожай|погод|засух|заморозк', 'G1'), (r'госзаказ|госзакуп|нацпроект|омс|гособорон', 'A3'),
    (r'господдержк|грант|субсид|льгот', 'C4'), (r'тариф', 'B2'), (r'логистик|перевоз|грузопоток|транзит', 'B3'),
    (r'экспорт|внешн.* рын', 'A4'), (r'регулирован|маркировк|лиценз|реестр', 'C3'), (r'импортозамещ', 'F1'),
    (r'факторинг|отсрочк', 'D4'), (r'лизинг|автокредит|долгов|облигац', 'D2'), (r'курс', 'D3'),
    (r'жиль|ипотек|стройк', 'A1'), (r'цен', 'B1'),
]
TOPIC = {'ставка': 'D1', 'кредит': 'D2', 'курс': 'D3', 'инфляция': 'B1', 'потребление': 'A1', 'рынок_труда': 'B4',
         'энергия_и_сырьё': 'B1', 'строительство': 'A2', 'металлургия': 'B1', 'бюджет_и_налоги': 'C1',
         'облигации': 'D2', 'внешняя_торговля': 'C2', 'инвестиции': 'A2', 'золото': 'B1', 'промышленность': 'A2'}
BASE = {'D1': 0.3, 'C1': 0.3, 'B4': 0.2, 'B1': 0.2, 'D3': 0.1}  # any business, weakly
STAFF = {'Менее 15': 0.2, '16-100': 0.3, '101-250': 0.5, 'Более 251': 0.7}


def lever_of(factor: dict) -> str:
    text = (factor['label'] + ' ' + ' '.join(factor['keywords'][:3])).lower()
    return next((lv for rx, lv in LABEL_RULES if re.search(rx, text)), TOPIC.get(factor['topic'], 'F1'))


def sensitivity(profile: dict, staff: str | None = None) -> dict[str, dict]:
    """profile: stage-2 profile (card, scale, facts, external_factors). staff: FNS headcount bucket if known."""
    card = profile['card']
    fid = {}
    for f in profile['facts']:
        fid.setdefault(f['cat'], f['id'])
    out = {k: {'lever': k, 'name': v, 'weight': 0.0, 'reasons': []} for k, v in LEVERS.items()}

    def bump(lever, w, reason):
        x = out[lever]
        x['weight'] = round(max(x['weight'], w), 2)
        if reason and reason not in x['reasons']:
            x['reasons'].append(reason)

    for lv, w in BASE.items():
        bump(lv, w, 'base')
    for f in profile.get('external_factors', []):
        bump(lever_of(f), f['weight'], f['id'])

    bank = fid.get('bank', 'card.products')
    if {'loan', 'leasing'} & set(card['products']):
        bump('D1', 1.0, bank), bump('D2', 0.8, bank)
    if 'factoring' in card['products']:
        bump('D4', 0.8, bank)
    if 'guarantee' in card['products']:
        bump('A3', 0.5, bank)  # bank guarantees are typical for public procurement contracts
    if 'salary' in card['products']:
        bump('B4', 0.5, bank)

    trade = fid.get('trade', 'card.fts_countries')
    if card['fts_countries'] or card['fts_flag']:
        bump('D3', 0.8, trade), bump('A4', 0.5, trade)  # direction unknown: foreign markets are possible, not certain
        bump('E1', 0.5, trade)
    if any(f['rule'] == 'fts:unfriendly' for f in profile.get('external_factors', [])):
        bump('E1', 0.9, trade)
    own = next((f['id'] for f in profile['facts'] if f['cat'] == 'ownership' and 'иностран' in f['text'].lower()), None)
    if own:
        bump('C5', 0.8, own), bump('E1', 0.6, own)
    state = next((f['id'] for f in profile['facts'] if f['cat'] == 'ownership' and 'Государственное' in f['text']), None)
    if state:
        bump('A3', 0.6, state), bump('C5', 0.5, state)

    if staff in STAFF:
        bump('B4', STAFF[staff], fid.get('profile', 'card'))
    if profile['scale'].get('case_large') or profile['scale'].get('group_case_large'):
        bump('D2', 0.5, 'scale.case_large')  # large borrowers use bonds and syndicated loans
    return out


def vector(sens: dict[str, dict]) -> list[float]:
    """Sensitivity in LEVERS order: the company side of news x company matching."""
    return [sens[k]['weight'] for k in LEVERS]
