"""Size class of a company: 209-FZ / PP RF 265 revenue and headcount ceilings, the category is the higher of the two."""
import pandas as pd

LARGE = 15e9  # case: large business = revenue above 15 bn rub
REVENUE_CLASS = [(120e6, 'micro'), (800e6, 'small'), (2e9, 'medium')]
STAFF_CLASS = {'Менее 15': 'micro', '16-100': 'small', '101-250': 'medium', 'Более 251': 'large'}
ORDER = ['micro', 'small', 'medium', 'large']
CLASS_RU = {'micro': 'микропредприятие', 'small': 'малый бизнес', 'medium': 'средний бизнес', 'large': 'крупный бизнес'}


def size_class(revenue, staff: str) -> tuple[str, str]:
    """Higher of the revenue and headcount categories; (class, basis). Unknown or placeholder revenue -> headcount only."""
    by_staff = STAFF_CLASS.get(staff)
    by_rev = None
    if pd.notna(revenue) and revenue > 1000:  # 1000 rub is the organisers' placeholder
        by_rev = next((c for hi, c in REVENUE_CLASS if revenue <= hi), 'large')
    known = [c for c in (by_rev, by_staff) if c]
    if not known:
        return 'unknown', 'нет выручки и штата'
    cls = max(known, key=ORDER.index)
    basis = 'выручка и штат' if by_rev and by_staff and by_rev == by_staff else \
        'выручка' if cls == by_rev and by_rev else 'штат'
    return cls, basis
