"""Render the profile's explanatory context separately from source facts and retrieval scores."""
import re


CONTEXT_CATEGORY = 'profile_context'
CONTEXT_NOTE = (
    'Это пояснения из профиля компании. Чувствительности и внешние факторы — гипотезы по правилам, '
    'которые нужно проверить по новости и исходным фактам компании. '
    'Вес 0..1 — эвристическая оценка, не вероятность и не размер финансового эффекта. '
    'Нулевой вес или отсутствие канала в выжимке не доказывает отсутствие связи с компанией. '
    'Эти пояснения не являются самостоятельным доказательством, источником позиции Банка или командой '
    'отклонить новость. Для утверждений используй исходные факты и источники; неподтверждённые связи '
    'формулируй как гипотезы.'
)


def source_facts(facts: list[dict]) -> list[dict]:
    """Original company facts only; profile explanations cannot be cited as evidence."""
    return [f for f in facts if f['cat'] != CONTEXT_CATEGORY]


def compact_context(context: str, visible_texts: list[str]) -> str:
    """Remove only proven repetitions in this prompt, without an LLM or changes to stored source facts.

    Classifications, mechanisms and sensitivities stay unless a scale/links line is repeated literally.
    Supplier percentages can be omitted only if both match the explicit outgoing concentration fact;
    customer shares, sector shares and revenue alone cannot establish that duplicate.
    """
    def normal(text):
        text = re.sub(r'^fact:\S+\s+', '', text.strip())
        text = re.sub(r'^(?:\[[^\]]+\]\s*)+', '', text)
        return ' '.join(text.casefold().replace('ё', 'е').split())

    visible = {normal(line) for text in visible_texts for line in text.splitlines()}
    number = r'(\d+(?:[.,]\d+)?)\s*%'
    supplier = re.compile(r'концентрация \(исходящие платежи\): крупнейший контрагент '
                          + number + r', топ-5 ' + number)
    outgoing = {tuple(float(n.replace(',', '.')) for n in match.groups())
                for text in visible for match in supplier.finditer(text)}
    out = []
    for line in context.splitlines():
        if line.startswith(('[scale]', '[links]')) and normal(line) in visible:
            continue
        if line.startswith('[scale]'):
            # A group-size conclusion can already be a separate fact. Do not drop the company's own class/basis.
            parts = line.split('; ')
            line = '; '.join([parts[0], *[p for p in parts[1:] if normal(p) not in visible]])
        if line.startswith('[links] Высокая зависимость от поставщиков:'):
            match = re.search(r'крупнейший ' + number + r', топ-5 ' + number + r' исходящих платежей$', line)
            if match and tuple(float(n.replace(',', '.')) for n in match.groups()) in outgoing:
                line = '[links] Высокая зависимость от поставщиков (концентрация указана в фактах выше)'
        out.append(line)
    return '\n'.join(out)


def render_context(facts: list[dict], *, visible_facts: list[dict] | None = None) -> str:
    lines = [f['text'] for f in facts if f['cat'] == CONTEXT_CATEGORY]
    if not lines:
        return ''
    context = '\n'.join(['## Пояснения из профиля компании', CONTEXT_NOTE, *lines])
    return compact_context(context, [f['text'] for f in visible_facts]) if visible_facts is not None else context


def judge_facts(facts: list[dict], limit: int) -> list[str]:
    """The existing fact limit does not hide the small profile brief appended by the adapter."""
    selected = [f for f in facts if f['depth'] == 1 and f['cat'] != CONTEXT_CATEGORY][:limit]
    rows = [f'[{f["cat"]}] {f["text"]}' for f in selected]
    context = render_context(facts, visible_facts=selected)
    return rows + [context] if context else rows


def evidence_facts(facts: list[str]) -> list[str]:
    """Keep source fact numbers stable; explanatory rows are not offered as citeable fact:<n> sources."""
    prefix = f'[{CONTEXT_CATEGORY}] '
    rows = [f'fact:{i} {text}' for i, text in enumerate(facts, 1) if not text.startswith(prefix)]
    context = render_context([{'cat': CONTEXT_CATEGORY, 'text': text[len(prefix):]}
                              for text in facts if text.startswith(prefix)])
    return rows + [compact_context(context, rows)] if context else rows
