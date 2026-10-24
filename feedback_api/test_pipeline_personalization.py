"""Offline test: the user's reactions change what the judge engine selects. Stage 3, stage 4 and the judge are stubs
with equal scores, so only personalization decides; no LLM, no case data. python -X utf8 feedback_api/test_pipeline_personalization.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
import pipeline
from common.personalization import Personalization, Reaction

NEWS = pd.DataFrame([
    {'id': 1, 'title': 'Биржевые цены на бензин выросли', 'text': 'бензин дизельное топливо АЗС'},
    {'id': 2, 'title': 'ЦБ сохранил ключевую ставку', 'text': 'ставка кредиты заемщики'},
    {'id': 3, 'title': 'Налог на прибыль повысят', 'text': 'налоги НДС налоговая нагрузка'},
    {'id': 4, 'title': 'Ипотека для семей с детьми', 'text': 'семейная ипотека застройщики'},
])
judged = []


class Relevance:
    def score(self, news, facts, role):
        return SimpleNamespace(score=2, channel='', model_dump=lambda: {'score': 2})


class Judge:
    def decide(self, news_id, *a, **k):
        judged.append(news_id)
        return SimpleNamespace(is_insight=False, bank_evidence=[], calculation_facts=[], calculation_bank=[],
                               model_dump=lambda: {'is_insight': False})


def run(personalization=None):
    judged.clear()
    p = pipeline.InsightPipeline.__new__(pipeline.InsightPipeline)
    p.relevance, p.judge, p.bank = Relevance(), Judge(), None
    p.consequences = SimpleNamespace(extract=lambda news: {})
    p.related = SimpleNamespace(event_plausibility=lambda i: 1.0)
    # __wrapped__: the run without @pinned_run, which would open the real news corpus
    pipeline.InsightPipeline.run.__wrapped__(p, 'U_1', 'cfo', n_candidates=4, n_judge=2, personalization=personalization)
    return list(judged)


pipeline.build = lambda ids, use_profiles=False: {ids[0]: [{'cat': 'bank', 'text': 'кредиты', 'depth': 1}]}
pipeline.candidates = lambda *a, **k: (NEWS.copy(), None)

base = run()
assert base == [1, 2], base  # equal scores: the stage-3 order wins

pers = Personalization.from_reactions('demo', 'U_1', 'cfo', [
    Reaction(kind='dislike', news_id=90, channel='fuel', title='Бензин дорожает', text='Бензин дорожает бензин АЗС'),
    Reaction(kind='like', news_id=91, channel='rate', title='Ставка ЦБ сохранена', text='Ставка ЦБ сохранена ставка кредиты'),
    Reaction(kind='already_known', news_id=3, title='Налог на прибыль'),
])
personal = run(pers)
assert personal[0] == 2 and 1 not in personal and 3 not in personal, personal
print(f'ok: without feedback judged {base}; after like(rate) + dislike(fuel) + already_known(news 3) judged {personal}')
