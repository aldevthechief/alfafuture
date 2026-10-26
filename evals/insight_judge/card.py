"""The card as the human labeller saw it (insight_judge_calibration/generate.py `compact`): title, essence, one action,
one bank thesis. The judge reads the same text, so its agreement with the human is not an artefact of a richer input."""


def clip(s: str, n: int) -> str:
    s = ' '.join((s or '').split())
    return s if len(s) <= n else s[:s.rfind(' ', 0, n)] + '…'


def _best_why(i: dict) -> str:
    skip = ('Основной вид деятельности', 'Годовая выручка', 'Крупный бизнес')  # raw profile lines of analytics insights
    texts = [c.get('text', '') for c in i.get('why_it_matters', [])]
    return max([t for t in texts if t and not t.startswith(skip)], key=len, default='')


def card_text(i: dict) -> str:
    """Insight (stage-6 dict, contract v2) -> the labeller's card."""
    why = _best_why(i)
    essence = ' '.join(c.get('text', '') for c in i.get('what_happened', [])[:2]).strip()
    essence = why if len(essence) < 60 else essence + (' ' + why if why else '')
    if len(essence) < 60:
        essence = i.get('ai_conclusion', '') or i.get('title', '')
    parts = ['**Заголовок:** ' + clip(i.get('title', ''), 150), '**Суть:** ' + clip(essence, 340)]
    if i.get('actions'):
        parts.append('**Действие:** ' + clip(i['actions'][0], 150))
    if bank := [c.get('text', '') for c in i.get('bank_view', [])][:1]:
        parts.append('**Позиция Банка:** ' + clip(bank[0], 200))
    return '\n'.join(parts)
