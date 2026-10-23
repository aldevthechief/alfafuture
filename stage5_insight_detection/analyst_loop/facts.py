"""Computed facts: every number the insight may cite, with its formula and source file. Pure code, no LLM imports
(a test enforces it for the whole compute layer), so each figure on screen can show where it came from."""
from dataclasses import asdict, dataclass


@dataclass
class Fact:
    id: str  # fact:<slug>, the only form a claim may cite
    label: str
    value: float | str
    unit: str = ''
    formula: str = ''
    source: str = ''
    kind: str = 'fact'  # fact | hypothesis (an assumption the data cannot confirm)

    def text(self) -> str:
        v = f'{self.value:,.1f}'.replace(',', ' ').replace('.0', '') if isinstance(self.value, float) else str(self.value)
        return f'{self.label}: {v}{" " + self.unit if self.unit else ""}'

    def render(self) -> str:
        extra = f' [формула: {self.formula}]' if self.formula else ''
        tag = ' (ГИПОТЕЗА)' if self.kind == 'hypothesis' else ''
        return f'{self.id} {self.text()}{tag}{extra}'

    def dict(self) -> dict:
        return asdict(self)


def money_value(rub: float) -> tuple[float, str]:
    for div, unit in ((1e12, 'трлн ₽'), (1e9, 'млрд ₽'), (1e6, 'млн ₽'), (1e3, 'тыс. ₽')):
        if abs(rub) >= div:
            return round(rub / div, 1), unit
    return round(rub, 0), '₽'
