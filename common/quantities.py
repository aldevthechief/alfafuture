"""Conservative quantity comparison: normalize scale and units, never keep a second unscaled reading."""
import re
from dataclasses import dataclass

NUMBER = re.compile(r'(?<![\w.])[-+−]?\d+(?:[.,]\d+)?')
THOUSANDS = re.compile(r'(?<![\d.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)')
SCALE = re.compile(r'\s*(тыс\.?|тысяч[а-я]*|млн\.?|миллион[а-я]*|млрд\.?|миллиард[а-я]*|трлн\.?|триллион[а-я]*)(?![а-яё])', re.I)
MULTIPLIERS = (('тыс', 1e3), ('млн', 1e6), ('миллион', 1e6), ('млрд', 1e9), ('миллиард', 1e9),
               ('трлн', 1e12), ('триллион', 1e12))
UNIT = re.compile(r'\s*(п\.?\s*п\.?|процентн[а-я]*\s+пункт[а-я]*|б\.?\s*п\.?|базисн[а-я]*\s+пункт[а-я]*|'
                  r'%|процент[а-я]*|₽|руб(?:\.|л[а-я]*)?|\$|долл(?:\.|ар[а-я]*)?|€|евро|'
                  r'кг|килограмм[а-я]*|т(?:онн[а-я]*)?|человек[а-я]*|сотрудник[а-я]*|шт\.?|штук[а-я]*)'
                  r'(?![а-яёa-z])', re.I)
PER_UNIT = re.compile(r'\s*(?:/|за\s+)\s*(т(?:онн[а-я]*)?|кг|килограмм[а-я]*|шт\.?|штук[а-я]*)(?![а-яё])', re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str = ''

    def matches(self, other: 'Quantity') -> bool:
        tolerance = max(0.051 if self.unit in ('%', 'pp') else 1e-9, 0.01 * abs(other.value))
        return self.unit == other.unit and abs(self.value - other.value) <= tolerance


def _unit(raw: str) -> tuple[str, float]:
    t = re.sub(r'[.\s]', '', raw.lower())
    if t.startswith(('пп', 'процентн')):
        return 'pp', 1
    if t.startswith(('бп', 'базисн')):
        return 'pp', 0.01
    if t == '%' or t.startswith('процент'):
        return '%', 1
    if t == '₽' or t.startswith('руб'):
        return 'RUB', 1
    if t == '$' or t.startswith('долл'):
        return 'USD', 1
    if t in ('€', 'евро'):
        return 'EUR', 1
    if t == 'кг' or t.startswith('килограмм'):
        return 'kg', 1
    if t == 'т' or t.startswith('тонн'):
        return 'kg', 1000
    if t.startswith(('человек', 'сотрудник')):
        return 'person', 1
    return 'item', 1


def quantity_mentions(text: str) -> list[tuple[Quantity, bool]]:
    """Dates/ordinals should be removed by the caller when their numbers are irrelevant to an audit."""
    text = THOUSANDS.sub(lambda m: re.sub(r'\s', '', m.group()), text)
    out = []
    for m in NUMBER.finditer(text):
        value = float(m.group().replace(',', '.').replace('−', '-'))
        if text[m.end():m.end() + 2] in ('-й', '-я', '-е', '-м'):
            continue
        end, unit = m.end(), ''
        scale = SCALE.match(text, end)
        if scale:
            value *= next(k for word, k in MULTIPLIERS if scale.group(1).lower().startswith(word))
            end = scale.end()
        suffix = UNIT.match(text, end)
        if suffix:
            unit, factor = _unit(suffix.group(1))
            value *= factor
            end = suffix.end()
        else:
            # Currency symbols also occur before the number: $10, ₽100.
            prefix = text[:m.start()].rstrip()[-1:]
            if prefix in ('₽', '$', '€'):
                unit, _ = _unit(prefix)
        per = PER_UNIT.match(text, end)
        if unit and per:
            denominator, factor = _unit(per.group(1))
            value /= factor
            unit += '/' + denominator
        if not scale and not unit and 1990 <= value <= 2100 and value.is_integer():
            continue
        rough = bool(re.search(r'(?:≈|около|примерно|приблизительно|порядка)\s*[₽$€]?\s*$', text[:m.start()], re.I))
        out.append((Quantity(value, unit), rough))
    return out


def quantities(text: str) -> list[Quantity]:
    return [q for q, _ in quantity_mentions(text)]
