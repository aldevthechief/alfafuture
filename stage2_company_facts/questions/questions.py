"""Questions to the user about the company, made from what the last analysis had to assume.

In the demo of 24.10 45% of the reasoning claims carried «если …» about facts the bank's data cannot show (a floating-rate
debt, imports in the portfolio, the share of public customers, an own fleet or subcontractors), and the writer added 3-5
hypotheses per card. Each such assumption is one yes / no question; the answer goes to the profile as «со слов
пользователя» and the next analysis reasons from it instead of «если». The frontend shows them in «Профиль компании».
- sources: the writer's hypotheses (already statements) and «Если <condition>, …» clauses of why / risks / opportunities;
- similar assumptions are merged (stems overlap), ranked by how many cards rest on them; at most `top`;
- the id is stable (hash of the normalised stems), so an answer stays attached across analyses.
No LLM, no case data: only the texts of the cards.
"""
import hashlib
import re

from common.text import stems

IF = re.compile(r'(?:^|[.;!?]\s+|\(\s*)[Ее]сли\s+([^,;:()]{12,170}?)(?=,|;|:)')
BLOCKS = ('why_it_matters', 'risks', 'opportunities')
GENERIC = re.compile(r'^(это|так|нет|да|ставк\w* (снизит|вырастет))\b', re.I)


def _statement(text: str) -> str:
    s = re.sub(r'\s+', ' ', text).strip()
    s = re.split(r'(?<=[^\d\s][.!?])\s+(?=[А-ЯЁA-Z])', s)[0]  # a hypothesis adds «Подтвердят …»: keep the claim
    s = re.split(r'\s*[:;]\s', s)[0].rstrip('.;:, —')  # «…: рынок на 67% плавающий», «; подтвердит …»: the reason, not the claim
    s = re.sub(r'^(что|то)\s+', '', s)
    first = s.split(' ', 1)[0]
    return s if len(first) > 1 and first.isupper() else s[:1].lower() + s[1:]  # keep «НДС», «ЦБ»


def _question(statement: str) -> str:
    return f'Верно ли, что {statement}?'


def _key(statement: str) -> frozenset:
    return frozenset(stems(statement))


def assumptions(results: list[dict]) -> list[dict]:
    """[{statement, insights: [titles]}] in order of appearance, not merged."""
    out = []
    for r in results:
        ins = r.get('insight')
        if not isinstance(ins, dict):
            continue
        title = ins.get('title', '')
        for h in ins.get('hypotheses', []):
            out.append({'statement': _statement(h.get('text', '')), 'insight': title, 'kind': 'hypothesis'})
        for block in BLOCKS:
            for c in ins.get(block, []):
                for m in IF.finditer(c.get('text', '')):
                    out.append({'statement': _statement(m.group(1)), 'insight': title, 'kind': 'if'})
    # fewer than 4 words: a clause cut at an inner dash («если автотехника — предмет лизинга»), not a question
    return [a for a in out if 12 <= len(a['statement']) <= 200 and len(a['statement'].split()) >= 4
            and not GENERIC.search(a['statement'])]


def questions(results: list[dict], top: int = 5, merge: float = 0.6) -> list[dict]:
    """The frontend contract of profile.hypotheses: {id, question, text, for, impact} + insights it affects.
    merge: share of the shorter statement's stems found in the other one («долг плавающий» ~ «долг по плавающей ставке, >30%»)."""
    groups = []  # [key stems, statement, set of insights, hypothesis?]
    for a in assumptions(results):
        k = _key(a['statement'])
        if not k:
            continue
        for g in groups:
            if len(k & g[0]) / min(len(k), len(g[0])) >= merge:
                g[2].add(a['insight'])
                g[3] = g[3] or a['kind'] == 'hypothesis'
                break
        else:
            groups.append([k, a['statement'], {a['insight']}, a['kind'] == 'hypothesis'])
    # cards resting on it first; on a tie a writer's hypothesis (a full statement) before an «если» fragment
    groups.sort(key=lambda g: (-len(g[2]), not g[3], len(g[1])))
    out = []
    for k, statement, cards, _ in groups[:top]:
        n = len(cards)
        out.append({'id': 'q_' + hashlib.sha1(' '.join(sorted(k)).encode()).hexdigest()[:12],
                    'question': _question(statement), 'text': statement[:1].upper() + statement[1:],
                    'for': f'от этого зависят выводы {n} {"инсайта" if n == 1 else "инсайтов"}',
                    'impact': 'Ответ уберёт это допущение из следующего анализа: выводы станут точнее и увереннее.',
                    'insights': sorted(cards)})
    return out
