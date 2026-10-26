"""Insight judge: the labelling task is parsed as the human saw it, both orders are merged, kappa / Bradley–Terry are right."""
import asyncio
import csv

from common.llm.testing import ScriptedChatModel, calls, tool_call
from evals.insight_judge.calibrate import agreement, bradley_terry, kappa, human_ratings, pointwise_check, position_bias
from evals.insight_judge.card import card_text
from evals.insight_judge.judge import InsightJudge, PairVerdict, merge
from evals.insight_judge.pairs import Pair, load_human, load_pairs

MD = """# Калибровка судьи инсайтов — U_1 — тест

Компания и роль: **U_1 · cfo**

**Экспозиция:** лизинговая компания.

---

## Пары для сравнения

### Пара 1-P01
**A.**
**Заголовок:** ДОМ.РФ
**Суть:** облигации

**B.**
**Заголовок:** Ставка
**Суть:** 14%

---

### Пара 1-P02
**A.**
**Заголовок:** Курс

**B.**
**Заголовок:** Налоги

---

## Бланк ответов
"""


def verdict(**kw):
    base = dict(relevance='A', significance='A', clarity='A', grounding='A', winner='A', confidence=3)
    return PairVerdict(**{**base, **kw})


def test_pairs_are_parsed_with_exposure_and_both_cards(tmp_path):
    (tmp_path / 'U_1_cfo.md').write_text(MD, encoding='utf-8')
    pairs = load_pairs(tmp_path)
    assert [p.id for p in pairs] == ['1-P01', '1-P02'] and pairs[0].company == 'U_1' and pairs[0].role == 'cfo'
    assert 'лизинговая компания' in pairs[0].exposure and pairs[0].a.startswith('**Заголовок:** ДОМ.РФ') and 'Ставка' in pairs[0].b


def test_human_sheet_reads_cyrillic_letters_and_skips_unfilled_rows(tmp_path):
    sheet = tmp_path / 's.csv'
    with open(sheet, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'r', 's', 'c', 'g', 'w', 'conf', 'err', 'comment'])
        w.writerow(['1-P01', 'А', 'В', '=', 'a', 'А', '3', '1', 'число'])  # Cyrillic А, В
        w.writerow(['1-P02', '', '', '', '', '', '', '', ''])
        w.writerow(['1-P03', 'A', 'x', 'A', 'A', 'A', '2', '0', ''])  # an invalid cell: not labelled
    human = load_human(sheet)
    assert list(human) == ['1-P01']
    assert human['1-P01'] | {} == dict(relevance='A', significance='B', clarity='=', grounding='A', winner='A', confidence=3,
                                       factual_error=True, comment='число')


def test_both_orders_must_agree_otherwise_tie():
    first = verdict(relevance='A', clarity='A', winner='A')
    swapped = verdict(relevance='B', clarity='A', winner='B')  # in the swapped order «B» is the original A
    out = merge(first, swapped)
    assert out['relevance'] == 'A' and out['winner'] == 'A'  # consistent
    assert out['clarity'] == '=' and out['agree']['clarity'] is False  # the judge just prefers the first position


def test_judge_asks_in_both_orders():
    model = ScriptedChatModel(responses=[calls(tool_call('PairVerdict', relevance='A', significance='A', clarity='A', grounding='A',
                                                         winner='A', confidence=3)),
                                         calls(tool_call('PairVerdict', relevance='B', significance='B', clarity='B', grounding='B',
                                                         winner='B', confidence=2))])
    out = asyncio.run(InsightJudge(model).ajudge_pair('exposure', 'card a', 'card b'))
    assert out['winner'] == 'A' and out['confidence'] == 2 and len(out['raw']) == 2


def test_kappa():
    assert kappa(list('AABB'), list('AABB')) == 1.0
    assert kappa(list('AABB'), list('BBAA')) == -1.0
    assert kappa(list('AAAA'), list('AAAA')) is None  # single label: undefined
    assert abs(kappa(list('AAB='), list('AAB=')) - 1) < 1e-9


def test_agreement_per_axis_and_position_bias():
    human = {'p1': dict(relevance='A', significance='A', clarity='B', grounding='=', winner='A', confidence=3, factual_error=False),
             'p2': dict(relevance='B', significance='B', clarity='B', grounding='B', winner='B', confidence=1, factual_error=True)}
    judged = {i: dict(h, raw=[h, h], agree=dict.fromkeys(('relevance', 'significance', 'clarity', 'grounding', 'winner'), True),
                      factual_error=h['factual_error']) for i, h in human.items()}
    judged['p1'] = dict(judged['p1'], clarity='A')  # one disagreement
    report = agreement(human, judged)
    assert report['relevance']['agree'] == 1.0 and report['clarity']['agree'] == 0.5 and report['winner']['n_sure'] == 1
    assert report['factual_error'] == dict(human=1, judge=1, both=1)
    assert position_bias(judged)['relevance']['both_orders_agree'] == 1.0


def test_bradley_terry_ranks_by_wins_and_stays_finite_for_unbeaten_cards():
    games = [('a', 'b', 1.0), ('a', 'c', 1.0), ('b', 'c', 1.0), ('c', 'd', 0.5)]
    r = bradley_terry(games)
    assert r['a'] > r['b'] > r['c'] and abs(r['c'] - r['d']) < 0.5 and max(map(abs, r.values())) < 10


def test_pointwise_judge_ranks_like_the_human_and_predicts_pairs():
    pairs = [Pair(f'p{i}', 'U_1', 'cfo', 'x', a, b) for i, (a, b) in enumerate([('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')])]
    human = {p.id: dict(relevance='A', significance='A', clarity='A', grounding='A', winner='A', confidence=3, factual_error=False)
             for p in pairs}  # a > b > c > d in every comparison
    scores = {c: dict.fromkeys(('relevance', 'significance', 'clarity', 'grounding'), s) for c, s in zip('abcd', (5, 4, 3, 2))}
    out = pointwise_check(pairs, human, scores)
    assert out['spearman'] == 1.0 and out['as_pairwise']['winner']['agree'] == 1.0
    assert out['release_thresholds'][0]['threshold'] == 2.0 and out['release_thresholds'][0]['passed'] == 1.0
    assert set(human_ratings(pairs, human)) == set('abcd')


def test_card_text_is_what_the_labeller_saw():
    insight = dict(title='Заголовок инсайта', what_happened=[dict(text='ДОМ.РФ разместил облигации на 10 млрд ₽ ' * 3)],
                   why_it_matters=[dict(text='Годовая выручка 32 млрд'), dict(text='Это давит на маржу компании')],
                   actions=['Пересчитать бюджет'], bank_view=[dict(text='Ставка 14% до конца года')])
    card = card_text(insight)
    assert card.startswith('**Заголовок:** Заголовок инсайта') and '**Действие:** Пересчитать бюджет' in card
    assert 'Годовая выручка' not in card and '**Позиция Банка:** Ставка 14%' in card


def test_hybrid_uses_pointwise_scores_only_where_the_pairwise_judge_fell_back_to_a_tie():
    from evals.insight_judge.calibrate import derive_pairs, hybrid
    pairs = [Pair('p', 'U_1', 'cfo', 'x', 'a', 'b')]
    scores = {'a': dict(relevance=2, significance=4, clarity=3, grounding=3), 'b': dict(relevance=5, significance=2, clarity=3, grounding=3)}
    derived = derive_pairs(pairs, scores)['p']
    assert derived['relevance'] == 'B' and derived['significance'] == 'A' and derived['clarity'] == '=' and derived['winner'] == 'B'  # means 3.0 vs 3.25
    judged = {'p': dict(relevance='=', significance='B', clarity='=', grounding='=', winner='=', factual_error=False)}
    mixed = hybrid(judged, {'p': derived})['p']
    assert mixed['relevance'] == 'B'  # a tie of the pairwise judge, decided by the scores
    assert mixed['significance'] == 'B'  # the pairwise verdict stands against the scores
    assert mixed['clarity'] == '='
