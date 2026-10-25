"""Client profile projection from the existing stage-2 profile; no generated demo fields."""
from backend.cabinet import company_name
from common.data import money

PRODUCTS = {'loan': 'Кредитные продукты', 'leasing': 'Лизинг', 'deposit': 'Депозиты',
            'guarantee': 'Банковские гарантии', 'factoring': 'Факторинг', 'rko': 'РКО', 'salary': 'Зарплатный проект'}


def display_profile(profile):
    card = profile['card']
    scale = profile.get('scale') or {}
    label = scale.get('class_ru') or 'Нет данных'
    if card.get('revenue') is not None:
        label += f" · выручка {money(card['revenue'])}"
    group = next((f['text'] for f in profile.get('facts', []) if f.get('cat') == 'group'), 'Нет данных')
    channels, seen = [], set()
    for factor in sorted(profile.get('external_factors', []), key=lambda f: -f.get('weight', 0)):
        if not factor.get('mechanism') or factor['label'] in seen:
            continue
        seen.add(factor['label'])
        channels.append(dict(name=factor['label'], why=factor['mechanism']))
    return dict(name=company_name(profile['id'], card.get('okved_name')), industry=card.get('okved_name') or 'Нет данных',
                scale=label[:1].upper() + label[1:], group=group, region=card.get('region'),
                bank_products=[PRODUCTS.get(p, p) for p in card.get('products', [])], channels=channels[:8], hypotheses=[])
