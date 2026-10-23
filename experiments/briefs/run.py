"""Company-brief variants for the stage-3/4 judge and ranker 1 (README.md here): (a) current brief.md, (b) brief_channels.md,
(c) brief_strength.md, (d) deterministic template from stage-2 fact texts, (e) = (d) + LLM hypotheses (brief_hypotheses.md).
python -X utf8 experiments/briefs/run.py [--llm]  →  experiments/briefs/briefs.json (without --llm: only (d) and cached answers)."""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from langchain_core.output_parsers import StrOutputParser

from common.llm import ChatModels, PromptLibrary
from stage4_news_relevance.batch_ranker.ranker import BatchRanker

IDS = ['U_106783', 'U_31505', 'U_42222', 'U_3263', 'U_44767']
PROMPTS = PromptLibrary(ROOT / 'stage4_news_relevance/batch_ranker/prompts')


def load_facts() -> dict[str, list[dict]]:
    """Saved stage-2 facts of the golden / large companies (data/companies); facts_for() builds any other company."""
    out = {}
    for f in ('facts_large.jsonl', 'facts.jsonl'):
        for line in open(ROOT / 'data/companies' / f, encoding='utf8'):
            d = json.loads(line)
            out[d['id']] = d['facts']
    return out


def facts_for(u: str) -> list[dict]:
    from stage2_company_facts.deterministic_facts.facts import build
    return load_facts().get(u) or build([u])[u]


def strength(p: float) -> str:
    return 'сильная' if p >= 30 else 'средняя' if p >= 5 else 'слабая'


def pct(s: str) -> float:
    m = re.search(r'(<?)(\d+)% суммы', s)
    return 0.5 if m and m[1] else float(m[2]) if m else 0.0


def deterministic(facts: list[dict]) -> str:
    """Code-built brief: fact texts regrouped by impact channel, each counterparty link with its share of the flow and a
    strength label; FTS countries always «не измерена». shortcut: parses stage-2 fact texts, breaks if facts.py wording
    changes; build from features.parquet if this format wins."""
    by = {}
    for f in facts:
        by.setdefault(f['cat'], []).append(f['text'])
    first = lambda cat, pre='': next((t for t in by.get(cat, []) if t.startswith(pre)), None)
    act = first('activity', 'Основной')
    lines = [f"Отрасль: {act.split(': ', 1)[1]} ({re.search(r'ОКВЭД [\d.]+', act)[0]}). "
             f"{first('scale', 'Годовая')}; {first('profile', 'Численность').replace('Численность по ФНС: ', 'штат ')}"]
    if r := first('rank'):
        lines.append(f'Место: {r.split(";")[0]}.')
    if any('иностран' in t for t in by.get('ownership', [])):
        lines.append('Собственник: иностранное участие.')

    for cat, title, who in (('payments_in', 'Покупатели (входящие платежи)', 'плательщиков'),
                            ('payments_out', 'Поставщики (исходящие платежи)', 'получателей')):
        ts = by.get(cat, [])
        if not ts:
            lines.append(f'{title}: нет данных (платежи видны в основном у клиентов Альфа-Банка).')
            continue
        tot = next(t for t in ts if re.match(r'(Входящие|Исходящие) платежи: \d+ контраг', t))
        share = tot.split('; это ')[1] if '; это ' in tot else ''
        conc = next((t.split(': ', 1)[1] for t in ts if t.startswith('Концентрация')), '')
        known = next((re.search(r'известен для (.+? суммы)', t)[1] for t in ts if 'известен для' in t), '?')
        lines.append(f'{title}: {share}. Концентрация: {conc}. Отрасль {who} известна для {known}.')
        m = re.match(r'(<?)(\d+)% годовой', share)
        if m and (m[1] or int(m[2]) < 30):  # a small visible flow: its shares say little about the business
            lines.append(f'  - виден только поток в {m[1]}{m[2]}% выручки: доли ниже описывают его, а не весь бизнес')
        elif 'раз больше' in share:
            lines.append('  - поток во много раз больше выручки: аномалия: обороты не отражены в выручке (проектная компания, транзит или неполная выручка в данных)')
        if big := next((t for t in ts if t.startswith('Крупнейший контрагент')), None):
            lines.append(f'  - крупнейший: {big.split(": ", 1)[1]}')
        elif conc:  # facts.py names the largest counterparty only when it is in the company table
            lines.append('  - крупнейший: контрагент вне базы, отрасль неизвестна')
        if dep := next((t for t in ts if t.startswith('Доля компании')), None):
            p = int(re.search(r': <?(\d+)%', dep)[1])  # '<1%' too
            if p >= 50:
                lines.append(f'  - крупнейший плательщик отправляет этой компании {p}% всех своих платежей' if cat == 'payments_in' else
                             f'  - крупнейший получатель получает от этой компании {p}% всех своих поступлений')
        for t in [t for t in ts if ' с ОКВЭД ' in t and pct(t) >= 1][:4]:
            m = re.search(r'— (.+?) с ОКВЭД ([\d.]+) «(.+?)»: (.+)$', t)
            lines.append(f'  - {m[3]} ({m[2]}, {m[1]}): {m[4]} — {strength(pct(t))}')
        if grp := next((t for t in ts if 'внутри своей группы' in t), None):
            lines.append(f'  - внутри группы: {re.search(r": (.+? суммы\))", grp)[1]}')

    if prods := first('bank', 'Продукты'):
        credit = any(w in prods for w in ('кредит', 'лизинг'))
        lines.append(f"Альфа-Банк: {prods.split(': ', 1)[1]}{' — чувствительность к ставке' if credit else ''}.")
    countries = [re.search(r'из страны (.+?) \(', t)[1] for t in by.get('trade', []) if 'из страны' in t]
    if countries:
        lines.append(f"ВЭД (ФТС): {', '.join(countries)} — сила не измерена: направление (ввоз или вывоз) и объём неизвестны, "
                     'это не доля бизнеса.')
    else:
        lines.append('ВЭД: в базе ФТС нет.')
    geo = first('geo').split(': ', 1)[1]
    other = [re.search(r'(\d+%|<1%) известной', t)[1] for t in by.get('payments_in', []) + by.get('payments_out', [])
             if 'других регионов' in t]
    lines.append(f"Регион: {geo}" + (f"; доля контрагентов из других регионов: входящие {other[0]}, исходящие {other[1]}"
                                     if len(other) == 2 else '') + '.')
    if g := by.get('group'):
        acts = [re.sub(r'^В группе ', '', t) for t in by.get('group_activity', [])[:3]]
        gp = first('group_payments')
        lines.append(f"Группа: {g[0].replace('Входит в группу', 'группа')}; {g[1][0].lower() + g[1][1:]}."
                     + (f" Другие компании: {'; '.join(acts)}." if acts else '') + (f' {gp}.' if gp else ''))
    lines.append('Неизвестно: направление и объём ВЭД; отрасль контрагентов вне названной доли суммы; период платежей; '
                 'структура выручки по продуктам.')
    return '\n'.join(lines)


def main(llm: bool):
    facts = load_facts()
    det = {u: deterministic(facts[u]) for u in IDS}
    out = {u: {'d': det[u]} for u in IDS}
    if llm:
        model = ChatModels().cheap(max_tokens=2000)
        ranker = BatchRanker(model)
        text = lambda u: '\n'.join(f"[{f['cat']}] {f['text']}" for f in facts[u])
        chain = lambda name: PROMPTS.chat(name) | model | StrOutputParser()
        jobs = {(u, 'a'): lambda u=u: ranker.brief(facts[u]) for u in IDS}
        jobs |= {(u, 'b'): lambda u=u: chain('brief_channels').invoke({'facts': text(u)}) for u in IDS}
        jobs |= {(u, 'c'): lambda u=u: chain('brief_strength').invoke({'facts': text(u)}) for u in IDS}
        # keep only hypothesis lines: Gemma sometimes echoes the skeleton first; FTS has no direction, so a line that says
        # export or import breaks the prompt's rule and is dropped
        hyp = lambda u: [l for l in chain('brief_hypotheses').invoke({'skeleton': det[u]}).splitlines()
                         if l.startswith('- ') and 'уверенность' in l and not re.search(r'(?i)экспорт|импорт', l)]
        jobs |= {(u, 'e'): lambda u=u: det[u] + '\nГипотезы о бизнес-модели:\n' + '\n'.join(hyp(u)) for u in IDS}
        with ThreadPoolExecutor(8) as pool:
            futs = {k: pool.submit(f) for k, f in jobs.items()}
        for (u, v), fut in futs.items():
            out[u][v] = fut.result()
    path = Path(__file__).with_name('briefs.json')
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf8')
    for u in IDS:
        print(f'===== {u}\n{out[u]["d"]}\n')


if __name__ == '__main__':
    main('--llm' in sys.argv)
