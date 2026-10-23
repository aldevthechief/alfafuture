"""Offline regressions for merged retrieval, event preparation, numeric evidence and publication gates.

python -X utf8 -m unittest test_merge_regressions
All companies/news/models here are synthetic; no encoder download or LLM calls.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import pipeline
from common.quantities import Quantity, quantities
from common.search import StemIndex, corpus_fingerprint
from stage3_news_search import retrieval
from stage4_news_relevance.plausibility import events
from stage4_news_relevance.relevance_scoring.relevance import Relevance
from stage5_insight_detection.analyst_loop.audit import audit
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.impact import indicator_delta, assess
from stage5_insight_detection.analyst_loop.loop import AnalystLoop
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.analyst_loop.verdict import InsightBrief
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.insight_judge.judge import InsightDecision
from stage6_insight_generation.insight_writer.insight import Claim, Insight, Review
from common.trace import AgentTrace
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor

ROOT = Path(__file__).parent
FACTS = [{'cat': 'activity', 'text': 'Основной вид деятельности: строительство', 'depth': 1,
          'entities': ['41.20', 'Строительство зданий']}]
NEWS = pd.DataFrame([
    {'id': 41, 'title': 'ЦБ повысил ключевую ставку до 15%', 'text': 'Ключевая ставка повышена на 1 п.п.',
     'source': 'Источник А', 'date': '2026-10-17', 'tier': 1, 'url': 'https://example.com/a'},
    {'id': 42, 'title': 'Банк России повысил ставку до 15%', 'text': 'Ставка выросла на 1 п.п.',
     'source': 'Источник Б', 'date': '2026-10-17', 'tier': 1, 'url': 'https://example.com/b'}])
EVENTS = pd.DataFrame([{'event': 999, 'news_ids': [41, 42], 'n_sources': 2, 'generators': [
    {'reason': 'own_industry', 'query': 'строительство', 'rank': 0, 'sim': 0.8}], 'best_rank': 0}])


def insight():
    return Insight(title='Синтетический инсайт', role='cfo', what_happened=[], why_it_matters=[],
                   actions=['Проверить данные'], bank_view=[], ai_conclusion='Проверить условия', confidence='low')


def profile_snapshot(company=None, facts=None):
    """The Kirill loader's input, kept synthetic so loop tests never read case CSVs."""
    company = company or CompanyProfile('synthetic', '41.20', 'Строительство', 'Москва', 1e9, products=['loan'])
    names = ('id', 'okved', 'okved_name', 'region', 'revenue', 'group_revenue', 'products', 'fts_countries',
             'fts_flag', 'pay_out', 'pay_in', 'pay_out_total', 'pay_in_total')
    return {'card': {k: getattr(company, k) for k in names}, 'facts': FACTS if facts is None else facts,
            'scale': dict(class_ru='синтетическая компания', class_basis='тест', case_large=False, group_case_large=False),
            'sensitivity': {}, 'external_factors': []}


def brief():
    return InsightBrief(refs=['news:1', 'fact:1'], facts=[], consequences=[], risks=[], bank=[], hypotheses=[],
                        impact={'level': 'low', 'criteria': 'test'}, not_bank_position=True, warnings=[])


class QuantityTests(unittest.TestCase):
    def test_dates_and_only_scaled_value(self):
        from stage5_insight_detection.analyst_loop.audit import numbers
        self.assertEqual(numbers('от 02.09.2026, на 2027–2028 гг., 4К26: рост 0,9%'), {0.9})
        self.assertEqual(quantities('69 тысяч'), [Quantity(69000)])


class DeltaTests(unittest.TestCase):
    def event(self, text):
        return EventCard.from_news(dict(NEWS.iloc[0], title=text, text=''))

    def test_final_level_is_not_delta(self):
        for text in ['Инфляция снизилась до 8%', 'Инфляция составила 8%', 'Инфляция выросла до 8%']:
            with self.subTest(text=text):
                event = self.event(text)
                self.assertIsNone(indicator_delta('инфляция', event, []))
                profile = CompanyProfile('synthetic', '47.11', 'Торговля', 'Москва', 1e9)
                consequences, _ = assess(event, profile, [])
                self.assertTrue(consequences)
                self.assertTrue(all(c.estimate is None for c in consequences))

    def test_explicit_changes(self):
        cases = [('Инфляция снизилась на 2%', 'инфляция', -2, '%'),
                 ('Инфляция снизилась с 10% до 8%', 'инфляция', -2, 'п.п.'),
                 ('Инфляция снизилась на 2 п.п. до 8%', 'инфляция', -2, 'п.п.'),
                 ('Ключевая ставка снизилась на 100 б.п. до 14%', 'ставка', -1, 'п.п.'),
                 ('Инфляция снизилась на -2%', 'инфляция', -2, '%')]
        for text, topic, value, unit in cases:
            with self.subTest(text=text):
                delta = indicator_delta(topic, self.event(text), [])
                self.assertIsNotNone(delta)
                self.assertEqual(delta[:2], (value, unit))


class EventCacheTests(unittest.TestCase):
    def test_missing_reused_and_changed_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.parquet'
            with patch.object(events, 'build_events', wraps=events.build_events) as build:
                first = events.ensure_events(NEWS, path)
                self.assertTrue(path.exists())
                events.ensure_events(NEWS.copy(), path)
                self.assertEqual(build.call_count, 1)
                for column, value in [('text', 'Совсем другой текст'), ('date', '2026-10-20'),
                                      ('source', 'Новый источник'), ('tier', 2), ('id', 1234)]:
                    changed = NEWS.copy()
                    changed.loc[0, column] = value
                    fresh = events.ensure_events(changed, path)
                    self.assertEqual(fresh.attrs['corpus_fingerprint'], corpus_fingerprint(changed))
                self.assertEqual(build.call_count, 6)
                self.assertEqual(first.news_ids.map(len).sum(), len(NEWS))

    def test_legacy_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.parquet'
            EVENTS.to_parquet(path)
            out = events.ensure_events(NEWS, path)
            self.assertIn('corpus_fingerprint', out.attrs)
            self.assertTrue(events.ensure_events(NEWS.head(0), path).empty)

    def test_related_news_builds_missing_events(self):
        from stage5_insight_detection.news_enrichment.related_news import RelatedNews
        with tempfile.TemporaryDirectory() as tmp, patch(
                'stage5_insight_detection.news_enrichment.related_news.news_index', return_value=(NEWS, None)):
            path = Path(tmp) / 'events.parquet'
            related = RelatedNews(path)
            self.assertTrue(path.exists())
            self.assertIsNotNone(related.event_plausibility(41))

    def test_process_reloads_updated_corpus(self):
        import common.search as search
        with tempfile.TemporaryDirectory() as tmp, patch.object(search, 'DATA', Path(tmp)):
            folder = Path(tmp) / 'news'
            folder.mkdir()
            NEWS.to_parquet(folder / 'news.parquet')
            self.assertEqual(search.news_pool().id.tolist(), [41, 42])
            NEWS.head(1).to_parquet(folder / 'news.parquet')
            self.assertEqual(search.news_pool().id.tolist(), [41])

    def test_lexical_index_reloads_updated_corpus(self):
        import common.search as search
        rows = pd.DataFrame([dict(NEWS.iloc[0], id=i, title=word, text='экономика')
                             for i, word in enumerate(['яблоки', 'груши', 'кредиты', 'налоги'] * 2)])
        with tempfile.TemporaryDirectory() as tmp, patch.object(search, 'DATA', Path(tmp)):
            path = Path(tmp) / 'news' / 'news.parquet'
            path.parent.mkdir()
            rows.to_parquet(path)
            before, index = search.news_index()
            pd.concat([rows, rows.head(1).assign(id=99)]).to_parquet(path)
            after, updated = search.news_index()
            self.assertEqual(len(before), 8)
            self.assertEqual(len(after), 9)
            self.assertIsNot(index, updated)

    def test_related_refreshes_membership_between_batches(self):
        from stage5_insight_detection.news_enrichment.related_news import RelatedNews
        with tempfile.TemporaryDirectory() as tmp, patch(
                'stage5_insight_detection.news_enrichment.related_news.news_index', return_value=(NEWS, None)) as index:
            related = RelatedNews(Path(tmp) / 'events.parquet')
            updated = NEWS.copy()
            updated.loc[0, 'id'] = 1234
            index.return_value = (updated, None)
            related.refresh()
            self.assertNotIn(41, related.row_of)
            self.assertIn(1234, related.row_of)
            self.assertIsNotNone(related.event_plausibility(1234))

    def test_embeddings_rebuild_when_text_changes_with_same_ids(self):
        import common.embed as embed
        with tempfile.TemporaryDirectory() as tmp, patch.object(embed, 'DATA', Path(tmp)), patch.object(
                embed, 'news_pool', return_value=NEWS) as pool, patch.object(
                embed, 'encode', side_effect=lambda texts, *a, **kw: np.ones((len(texts), 3))) as encode:
            embed.news_embeddings.cache_clear()
            embed.news_embeddings('synthetic')
            embed.news_embeddings('synthetic')
            self.assertEqual(encode.call_count, 1)
            # A new process may use the persisted cache if its fingerprint matches.
            embed.news_embeddings.cache_clear()
            embed.news_embeddings('synthetic')
            self.assertEqual(encode.call_count, 1)
            updated = NEWS.copy()
            updated.loc[0, 'text'] = 'Изменённый текст при прежнем ID'
            pool.return_value = updated
            embed.news_embeddings('synthetic')
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(len(encode.call_args.args[0]), 1)
            embed.news_embeddings.cache_clear()


class RetrievalTests(unittest.TestCase):
    def test_adapter_preserves_identity_membership_and_provenance(self):
        row = retrieval.event_news(EVENTS, NEWS).iloc[0]
        self.assertEqual(row.id, 41)
        self.assertEqual(row.event_id, 999)
        self.assertEqual(row.event_news_ids, [41, 42])
        self.assertEqual(row.sources, ['Источник А', 'Источник Б'])
        self.assertEqual(row.text, NEWS.iloc[0].text)
        self.assertEqual(row.generators, EVENTS.iloc[0].generators)

    def test_hybrid_generator_runs_with_offline_embeddings(self):
        from stage3_news_search.candidate_generators import generators as module
        index = StemIndex(NEWS.title.tolist(), min_df=1, max_df=1.0)
        with patch.object(module, 'news_index', return_value=(NEWS, index)), patch.object(
                module, 'events_frame', return_value=(EVENTS, np.array([0, 0]))), patch(
                'common.embed.news_embeddings', return_value=np.ones((2, 1))), patch(
                'common.embed.encode', side_effect=lambda texts: np.ones((len(texts), 1))):
            out, gens = module.candidates(FACTS, role='cfo', n=1, l=1)
        self.assertEqual(out.event.tolist(), [999])
        self.assertTrue(gens)

    def test_both_pipeline_engines_use_default_retrieval_and_pass_news_rows(self):
        with patch.object(retrieval, 'v2_candidates', return_value=(EVENTS, [])) as v2, patch.object(
                retrieval, 'news_pool', return_value=NEWS), patch.object(pipeline, 'build', return_value={'synthetic': FACTS}):
            runner = pipeline.InsightPipeline.__new__(pipeline.InsightPipeline)
            runner.relevance = SimpleNamespace(score=Mock(return_value=Relevance(score=2, role_fit=2)))
            runner.consequences = SimpleNamespace(extract=Mock(return_value={}))
            runner.related = SimpleNamespace(event_plausibility=lambda i: 1, row_of={})
            runner.bank = None
            runner.judge = SimpleNamespace(decide=Mock(return_value=InsightDecision(
                is_insight=True, score=2, company_link='test', mechanism='test')))
            runner.writer = SimpleNamespace(write=Mock(return_value=(insight(), Review(ok=False, issues=['bad']))))
            out = runner.run('synthetic', 'cfo', 1, 1)
            self.assertEqual(out[0]['news_id'], 41)
            self.assertEqual(out[0]['retrieval']['event_id'], 999)
            self.assertNotIn('insight', out[0])
            self.assertEqual(out[0]['status'], 'review_rejected')
            runner.writer.write.return_value = (insight(), Review(ok=True))
            self.assertIn('insight', runner.run('synthetic', 'cfo', 1, 1)[0])
            self.assertTrue(v2.called)
            models = Mock()
            with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'DATA', Path(tmp)), patch(
                    'common.trace.AgentTrace', return_value=AgentTrace(folder=None)), patch(
                    'stage5_insight_detection.analyst_loop.pool.RelatedNewsPool'), patch(
                    'stage2_company_facts.profile.integration.load_profile', return_value=profile_snapshot()), patch(
                    'stage4_news_relevance.news_classifier.classifier.NewsClassifier') as classifier, patch(
                    'stage5_insight_detection.bank_analytics.library.FRAGMENTS') as fragments, patch(
                    'stage5_insight_detection.bank_analytics.library.BankLibrary.load'), patch(
                    'stage5_insight_detection.analyst_loop.loop.AnalystLoop') as loop:
                classifier.return_value.classify.return_value = [dict(id=41, importance=4, affected_roles=['cfo'])]
                fragments.exists.return_value = True
                loop.return_value.run.return_value = {'news_id': 41, 'title': 'synthetic',
                                                      'verdict': {'status': 'reject', 'reasons': []}}
                out = pipeline.run_loop('synthetic', 'cfo', models, 1, 1)
                passed = loop.return_value.run.call_args.args[0]
                self.assertEqual(passed['id'], 41)
                self.assertEqual(passed['event_news_ids'], [41, 42])
                self.assertEqual(passed['sources'], ['Источник А', 'Источник Б'])
                self.assertEqual(out[0]['retrieval']['event_id'], 999)

    def test_explicit_lexical_mode(self):
        with patch.object(retrieval, 'lexical_candidates', return_value=(NEWS, [])) as lexical, patch.object(
                retrieval, 'hybrid_candidates') as hybrid:
            retrieval.candidates('synthetic', 'cfo', 1, facts=FACTS, method='lexical')
            lexical.assert_called_once_with('synthetic', 'cfo', top=1)
            hybrid.assert_not_called()


class PublicationTests(unittest.TestCase):
    def test_scripted_pipeline_keeps_only_approved_insights(self):
        for ok in (False, True):
            with self.subTest(review_ok=ok), tempfile.TemporaryDirectory() as tmp, patch.object(
                    retrieval, 'v2_candidates', return_value=(EVENTS, [])), patch.object(
                    retrieval, 'news_pool', return_value=NEWS), patch.object(
                    pipeline, 'build', return_value={'synthetic': FACTS}):
                cheap = ScriptedChatModel(responses=[
                    calls(tool_call('Relevance', score=2, role_fit=2)),
                    calls(tool_call('NewsConsequences', economic=True, event='Ставка выросла', status='fact', significance=2))])
                script = [calls(tool_call('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')),
                          calls(tool_call('Insight', **insight().model_dump())),
                          calls(tool_call('Review', ok=ok, issues=[] if ok else ['Ошибка']))]
                if not ok:  # two rewrites, each checked again
                    script += [calls(tool_call('Insight', **insight().model_dump())),
                               calls(tool_call('Review', ok=False, issues=['Ошибка осталась']))] * 2
                agent = ScriptedChatModel(responses=script)
                related = SimpleNamespace(event_plausibility=lambda i: 1, row_of={},
                                          get=lambda i: NEWS[NEWS.id == i].iloc[0].to_dict())
                runner = pipeline.InsightPipeline(cheap, agent, related=related, consequences=ConsequenceExtractor(
                    cheap, cache=JsonlCache('consequences', folder=Path(tmp))))
                out = runner.run('synthetic', 'cfo', 1, 1)
                self.assertEqual(out[0]['news_id'], 41)
                self.assertEqual(out[0]['retrieval']['event_id'], 999)
                self.assertEqual('insight' in out[0], ok)
                self.assertEqual('draft' in out[0], not ok)
                self.assertEqual(cheap.i, 0)
                self.assertEqual(agent.i, 0)

    def test_analyst_loop_blocks_failed_review(self):
        syn = ROOT / 'stage5_insight_detection/analyst_loop/synthetic'
        news = json.loads((syn / 'news.json').read_text(encoding='utf-8'))['news']
        company = {k: v for k, v in json.loads((syn / 'company.json').read_text(encoding='utf-8')).items()
                   if not k.startswith('_')}
        pages = json.loads((syn / 'bank_pages.json').read_text(encoding='utf-8'))['reports']
        library = BankLibrary.from_pages(pages, date(2026, 10, 20))
        pool = ListNewsPool(news)
        for ok in (False, True):
            with self.subTest(review_ok=ok):
                writer = SimpleNamespace(write=Mock(return_value=(insight(), Review(ok=ok, issues=[] if ok else ['bad']))))
                loop = AnalystLoop(library, pool, writer=writer, trace=AgentTrace(folder=None), explain=False)
                out = loop.run(pool.get(101), CompanyProfile(**company), 'cfo', {'importance': 4})
                writer.write.assert_called_once()
                self.assertEqual('insight' in out, ok)
                self.assertEqual('draft' in out, not ok)
                self.assertEqual(out['status'], 'ready' if ok else 'review_rejected')
                if not ok:
                    from stage6_insight_generation.insight_card.render import render
                    card = render(out)
                    self.assertIn('Инсайт не прошёл проверку', card)
                    self.assertNotIn('Что сделать', card)


class DateTests(unittest.TestCase):
    def test_mixed_page_and_fallback_formats(self):
        # The collector is a standalone script with sibling imports.
        folder = ROOT / 'stage1_news_pool/web_scraping'
        with patch.object(sys, 'path', [str(folder), *sys.path]):
            spec = importlib.util.spec_from_file_location('collector_build', folder / 'build.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        today = pd.Timestamp.now(tz='UTC').normalize()
        df = pd.DataFrame({'published_page': [today.isoformat(), today.strftime('%Y-%m-%d'), 'invalid', None],
                           'published': [None, None, today.strftime('%a, %d %b %Y %H:%M:%S GMT'), today.isoformat()]})
        out = module.parse_dates(df, 3)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(d == today.date() for d in out.date))
        df = pd.DataFrame({'published_page': [today.isoformat(), (today - pd.Timedelta(days=20)).isoformat(), None],
                           'published': [(today - pd.Timedelta(days=20)).isoformat(), today.isoformat(), 'bad']})
        self.assertEqual(module.parse_dates(df, 3).index.tolist(), [0])


if __name__ == '__main__':
    unittest.main()
