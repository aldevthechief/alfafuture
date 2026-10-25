"""Deterministic hybrid/reranking/fact-index regressions; no downloads, API keys or real news required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from common.fusion import reciprocal_rank_fusion
from common.news_corpus import stable_id
from common.news_fact_index import FactStore, FactIndex
from common.news_retrieval import hybrid_search
from common.search import StemIndex, news_document
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage1_news_pool.updater import refresh_once
from stage4_news_relevance.semantic_ranker.ranker import SemanticRanker
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage5_insight_detection.news_enrichment.related_news import RelatedNews


def article(slug, title, text):
    url = f'https://example.org/{slug}'
    return dict(id=stable_id(url), url=url, title=title, text=text, source='test', tier=1, date='2026-10-22')


NEWS = pd.DataFrame([
    article('a', 'Ставка', 'Банк изменил ставку кредитования.'),
    article('b', 'Условия финансирования', 'Заемные средства стали дешевле для предприятий.'),
    article('c', 'Отраслевые события', 'Завод увеличил выпуск продукции.')])


def extracted(row, *, country='Китай'):
    quote = row['text']
    return dict(facts=[dict(text='Расширение производства', participants=['Завод'], industries=['машиностроение'],
                            countries=[country], commodities=['оборудование'], change='рост выпуска',
                            evidence=dict(quote=quote, start=0, end=len(quote)))])


class FusionTests(unittest.TestCase):
    def test_shared_rrf_preserves_ties_and_ignores_missing_hits(self):
        a, b = np.array([2., 2., 0.]), np.array([0., 1., 2.])
        expected = 1 / (60 + pd.Series(a).rank(ascending=False)) + 1 / (60 + pd.Series(b).rank(ascending=False))
        np.testing.assert_allclose(reciprocal_rank_fusion(a, b), expected)
        out = reciprocal_rank_fusion([1, np.nan], [np.nan, np.nan])
        self.assertGreater(out[0], 0)
        self.assertEqual(out[1], 0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([1], [1, 2])


class StoreTests(unittest.TestCase):
    def test_incremental_changes_failures_restarts_and_deleted_articles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'facts.sqlite3'
            store = FactStore(path, version='test1')
            extractor = Mock()
            extractor.extract.side_effect = extracted
            first = store.refresh(NEWS, extractor)
            self.assertEqual(first['processed'], 3)
            original_bytes = store.records(NEWS)[str(NEWS.iloc[1].id)]
            self.assertEqual(FactStore(path, version='test1').refresh(NEWS, extractor)['reused'], 3)
            self.assertEqual(extractor.extract.call_count, 3)
            changed = NEWS.copy()
            changed.loc[0, 'text'] = 'Банк повысил ставку.'
            self.assertNotIn(str(changed.iloc[0].id), store.records(changed))
            extractor.extract.side_effect = RuntimeError('temporary error')
            result = store.refresh(changed, extractor)
            self.assertEqual(len(result['failed']), 1)
            self.assertEqual(result['reused'], 2)
            self.assertNotIn(str(changed.iloc[0].id), store.records(changed))
            extractor.extract.side_effect = extracted
            self.assertEqual(store.refresh(changed, extractor)['processed'], 1)
            self.assertEqual(store.records(changed)[str(changed.iloc[1].id)], original_bytes)
            self.assertEqual(len(store.records(changed.head(1))), 1)
            self.assertEqual(FactStore(path, version='test2').records(changed), {})
            self.assertEqual(FactStore(path, version='test2').refresh(changed, extractor, limit=1)['pending'], 2)

    def test_no_supported_facts_and_limits_do_not_publish_fabricated_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactStore(Path(tmp) / 'facts.sqlite3', version='test')
            fake = extracted(NEWS.iloc[0])
            fake['facts'][0]['evidence']['quote'] = 'fabrication'
            extractor = Mock(extract=Mock(return_value=fake))
            report = store.refresh(NEWS, extractor, limit=1)
            self.assertEqual(len(report['failed']), 1)
            self.assertEqual(report['pending'], 2)
            self.assertEqual(store.records(NEWS), {})
            extractor.extract.return_value = dict(facts=[])
            self.assertEqual(store.refresh(NEWS, extractor)['processed'], 3)
            self.assertEqual(store.refresh(NEWS, extractor)['reused'], 3)

    def test_changed_row_does_not_destroy_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactStore(Path(tmp) / 'facts.sqlite3', version='test')
            model = Mock(extract=Mock(side_effect=extracted))
            store.refresh(NEWS.head(1), model)
            changed = NEWS.head(1).assign(text='Обновленная статья')
            store.refresh(changed, model)
            self.assertEqual(store.records(NEWS)[str(NEWS.iloc[0].id)][0]['evidence']['quote'], NEWS.iloc[0].text)
            self.assertEqual(store.records(changed)[str(NEWS.iloc[0].id)][0]['evidence']['quote'], 'Обновленная статья')

    def test_updater_retries_fact_backlog_even_without_new_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            extractor = Mock(extract=Mock(side_effect=extracted))
            initial = refresh_once(folder, collector=lambda *a, **kw: NEWS, enrich_facts=True, fact_limit=1, extractor=extractor)
            self.assertEqual(initial['facts']['processed'], 1)
            second = refresh_once(folder, collector=lambda *a, **kw: None, enrich_facts=True, extractor=extractor)
            self.assertEqual(second['facts']['processed'], 2)
            self.assertEqual(second['facts']['reused'], 1)

    def test_extractor_preserves_structured_fields_and_literal_evidence(self):
        source = NEWS.iloc[2].to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            model = ScriptedChatModel(responses=[calls(tool_call('NewsConsequences', economic=True,
                event='Рост выпуска', status='fact', significance=2, facts=[dict(text='Рост выпуска', quote=source['text'],
                participants=['Завод'], industries=['промышленность'], countries=[], commodities=['продукция'], change='рост')]))])
            result = ConsequenceExtractor(model, cache=JsonlCache('test', folder=Path(tmp))).extract(source)
            self.assertEqual(result['facts'][0]['industries'], ['промышленность'])
            self.assertEqual(result['facts'][0]['evidence']['quote'], source['text'])


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.index = StemIndex(news_document(NEWS).tolist(), min_df=1, max_df=1.0)
        self.empty_store = SimpleNamespace(records=lambda news: {})

    def test_semantic_only_and_fact_only_hits_are_combined_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactStore(Path(tmp) / 'facts.sqlite3', version='test')
            store.refresh(NEWS.tail(1), Mock(extract=Mock(side_effect=extracted)))
            with patch('common.embed.news_embeddings', return_value=np.array([[.1], [.9], [.0]])), \
                 patch('common.embed.encode', return_value=np.array([[1.]])):
                result = hybrid_search(NEWS, self.index, 'ставка машиностроение', k=3, rerank=False, fact_store=store)
            self.assertEqual(set(result.id), set(NEWS.id))
            self.assertTrue(result.id.is_unique)
            by_id = result.set_index('id')
            self.assertIn('semantic', by_id.loc[NEWS.iloc[1].id, 'search_channels'])
            self.assertNotIn('lexical', by_id.loc[NEWS.iloc[1].id, 'search_channels'])
            self.assertIn('facts', by_id.loc[NEWS.iloc[2].id, 'search_channels'])
            evidence = by_id.loc[NEWS.iloc[2].id, 'matched_facts'][0]['evidence']
            self.assertEqual(evidence['quote'], NEWS.iloc[2].text)
            self.assertEqual(by_id.loc[NEWS.iloc[2].id, 'matched_fragments'][0]['origin'], 'fact')

    def test_reranker_scores_joint_pairs_only_for_shortlist(self):
        model = Mock(predict=Mock(return_value=np.array([.1, .9])))
        ranker = SemanticRanker(model, max_candidates=2)
        rows = NEWS.copy()
        rows['retrieval_score'] = [.9, .8, .7]
        result = ranker.rank('дешевое финансирование', rows, top=1)
        self.assertEqual(result.iloc[0].id, NEWS.iloc[1].id)
        pairs = model.predict.call_args.args[0]
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair[0] == 'дешевое финансирование' for pair in pairs))
        self.assertIn(NEWS.iloc[1].text, pairs[1][1])
        self.assertEqual(result.iloc[0].retrieval_score, .8)

    def test_related_news_uses_cross_encoder_after_fusion(self):
        model = Mock(predict=Mock(return_value=[.1, .9]))
        with tempfile.TemporaryDirectory() as tmp, patch('stage5_insight_detection.news_enrichment.related_news.news_index',
                return_value=(NEWS, self.index)):
            related = RelatedNews(Path(tmp) / 'events.parquet', ranker=SemanticRanker(model, max_candidates=2),
                                  candidate_limit=2, fact_store=self.empty_store, rerank=True)
            default = RelatedNews(Path(tmp) / 'events.parquet', ranker=SemanticRanker(model, max_candidates=2),
                                  candidate_limit=2, fact_store=self.empty_store)
            with patch('common.embed.news_embeddings', return_value=np.array([[.9], [.8], [0.]])), \
                 patch('common.embed.encode', return_value=np.array([[1.]])):
                result = related.search('ставка', 1)
                calls = model.predict.call_args.args[0]
                model.predict.reset_mock()
                off = default.search('ставка', 1)
        self.assertEqual(result.iloc[0].id, NEWS.iloc[1].id)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.attrs['search_diagnostics']['rerank'], 'ok')
        # off by default (24.10): the cross-encoder is never called, so its model is never loaded
        model.predict.assert_not_called()
        self.assertNotIn('rerank', off.attrs['search_diagnostics'])

    def test_unavailable_models_fall_back_with_explicit_diagnostics(self):
        ranker = SemanticRanker(Mock(predict=Mock(side_effect=RuntimeError('not installed'))))
        with patch('stage3_news_search.retrieval_v2.hybrid.doc_vectors', side_effect=OSError('offline')), self.assertWarns(RuntimeWarning):
            result = hybrid_search(NEWS, self.index, 'ставка', fact_store=self.empty_store, ranker=ranker)
        self.assertEqual(result.iloc[0].id, NEWS.iloc[0].id)
        self.assertIn('unavailable', result.attrs['search_diagnostics']['semantic'])
        self.assertIn('unavailable', result.attrs['search_diagnostics']['rerank'])

    def test_empty_query_never_loads_models_and_wrong_scores_are_rejected(self):
        with patch('stage3_news_search.retrieval_v2.hybrid.doc_vectors', side_effect=AssertionError('must stay lazy')):
            self.assertTrue(hybrid_search(NEWS, self.index, '', fact_store=self.empty_store).empty)
            self.assertTrue(hybrid_search(NEWS.head(0), self.index, 'query', fact_store=self.empty_store).empty)
        ranker = SemanticRanker(Mock(predict=Mock(return_value=[float('nan')])))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            ranker.rank('query', NEWS.head(1))


if __name__ == '__main__':
    unittest.main()
