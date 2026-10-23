"""Synthetic rebuilds: ID migration, snapshot consistency and invalidation. No encoder/LLM/network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from common import news_corpus as corpus
from common.search import corpus_fingerprint
from common.llm import JsonlCache
from stage3_news_search.retrieval import event_news
from stage4_news_relevance.plausibility.events import ensure_events
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor, NewsConsequences
from stage4_news_relevance.plausibility.event_summary import EventSummarizer, EventSummary


def news():
    return pd.DataFrame([
        dict(id=0, url='https://example.com/a', title='ЦБ повысил ключевую ставку до 15%',
             text='Ключевая ставка повышена на 1 п.п.', date='2026-10-17', source='Источник А', tier=1),
        dict(id=1, url='https://example.com/b', title='Банк России повысил ставку до 15%',
             text='Ставка выросла на 1 п.п.', date='2026-10-17', source='Источник Б', tier=1)])


def result():
    return {'news_id': 0, 'title': 'Синтетика', 'decision': {'related_news_ids': [1]},
            'insight': {'what_happened': [{'text': 'Ставка повысилась', 'sources': ['news:0', 'news:1']}]}}


class ReferencesTests(unittest.TestCase):
    def test_id_migration_resolves_url_and_content(self):
        original = news()
        bound = corpus.bind_result(result(), original)
        updated = original.assign(id=original.url.map(corpus.stable_id))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            corpus.write_json(path, [bound])
            loaded = corpus.load_results(path, updated)[0]
        self.assertEqual(loaded['insight'], bound['insight'])  # historical claims are immutable
        ref = loaded['resolved_news_references']['news:0']
        self.assertEqual(ref['status'], 'remapped')
        self.assertEqual(ref['current_id'], str(corpus.stable_id(original.iloc[0].url)))

    def test_recycled_id_never_retargets_another_url(self):
        ref = corpus.news_reference(news().iloc[0].to_dict())
        updated = news().iloc[[1]].assign(id=0)
        self.assertEqual(corpus.resolve_reference(ref, updated)['status'], 'missing')

    def test_changed_content_and_removed_articles_are_stale(self):
        original = news()
        bound = corpus.bind_result(result(), original)
        for updated in [original.head(1), original.assign(text=['Новый текст', original.iloc[1].text])]:
            with self.subTest(rows=len(updated)):
                self.assertFalse(corpus.result_references(bound, updated)['ok'])
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / 'result.json'
                    corpus.write_json(p, [bound])
                    with self.assertRaises(corpus.StaleNewsReferences):
                        corpus.load_results(p, updated)

    def test_legacy_missing_and_partial_references_are_not_trusted(self):
        original = news()
        self.assertFalse(corpus.result_references(result(), original)['ok'])
        bound = corpus.bind_result(result(), original)
        del bound['news_references']['news:1']
        self.assertFalse(corpus.result_references(bound, original)['ok'])
        with self.assertRaises(corpus.StaleNewsReferences):
            corpus.bind_result(dict(result(), news_id=999), original)

    def test_saved_history_reads_original_snapshot_after_rebuild(self):
        original = news()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'news'
            p = Path(tmp) / 'insights' / 'result.json'
            corpus.save_results(p, [result()], original, folder)
            updated = original.assign(text='Изменённая статья')
            with self.assertRaises(corpus.StaleNewsReferences):
                corpus.load_results(p, updated)
            old = corpus.load_history(p, folder)[0]
            self.assertEqual(old['historical_news'][0]['text'], original.iloc[0].text)
            with self.assertRaises(corpus.StaleNewsReferences):
                corpus.save_results(p, [old], updated, folder)

    def test_golden_pairs_remap_without_rewriting_labels(self):
        original = news()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.json'
            corpus.write_json(path, {'pairs': {'synthetic': [0, 1]}})
            corpus.bind_pairs(path, original)
            updated = original.assign(id=original.url.map(corpus.stable_id))
            self.assertEqual(corpus.load_pairs(path, updated)['synthetic'], updated.id.tolist())
            with self.assertRaises(corpus.StaleNewsReferences):
                corpus.load_pairs(path, updated.head(1))
            with self.assertRaises(corpus.StaleNewsReferences):
                corpus.bind_pairs(path, updated)


class PublishTests(unittest.TestCase):
    def test_publish_builds_events_migration_and_saved_reference_report(self):
        original = news()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'news'
            folder.mkdir()
            original.to_parquet(folder / 'news.parquet', index=False)
            path = Path(tmp) / 'insights' / 'result.json'
            corpus.save_results(path, [result()], original, folder)
            legacy = path.parent / 'legacy.json'
            corpus.write_json(legacy, [result()])
            corpus.write_json(Path(tmp) / 'llm_runs/manifest.json', {'corpus_fingerprint': corpus_fingerprint(original)})
            updated = original.assign(id=original.url.map(corpus.stable_id))
            report = corpus.publish_news(updated, folder)
            self.assertEqual(report['migration']['remapped'], 2)
            self.assertIn('stale', report['frozen_llm_queues'])
            active = pd.read_parquet(folder / 'news.parquet')
            self.assertEqual(active.attrs['id_scheme'], corpus.ID_SCHEME)
            ev = pd.read_parquet(folder / 'events.parquet')
            self.assertEqual(ev.attrs['corpus_fingerprint'], corpus_fingerprint(updated))
            self.assertEqual(set(i for ids in ev.news_ids for i in ids), set(updated.id))
            states = {r['file']: r['ok'] for r in report['saved_results']}
            self.assertEqual(states, {'legacy.json': False, 'result.json': True})
            self.assertTrue(path.with_suffix('.references.json').exists())
            # Snapshots and label files have never been overwritten with the new IDs.
            self.assertEqual(corpus.load_history(path, folder)[0]['news_id'], 0)

    def test_preparation_failure_keeps_active_corpus(self):
        original = news()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original.to_parquet(folder / 'news.parquet', index=False)
            before = (folder / 'news.parquet').read_bytes()
            updated = original.assign(id=original.url.map(corpus.stable_id))
            with patch('stage4_news_relevance.plausibility.events.ensure_events', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaises(RuntimeError):
                    corpus.publish_news(updated, folder)
            self.assertEqual((folder / 'news.parquet').read_bytes(), before)

    def test_fingerprint_covers_urls_and_events_reject_mixed_snapshot(self):
        original = news()
        updated = original.assign(url=['https://example.com/new', original.iloc[1].url])
        self.assertNotEqual(corpus_fingerprint(original), corpus_fingerprint(updated))
        with tempfile.TemporaryDirectory() as tmp:
            ev = ensure_events(original, Path(tmp) / 'events.parquet')
        with self.assertRaises(corpus.StaleNewsReferences):
            event_news(ev, updated)

    def test_experiment_labels_are_blocked_after_corpus_change(self):
        import common.data
        original = news()
        with tempfile.TemporaryDirectory() as tmp, patch.object(common.data, 'DATA', Path(tmp)):
            folder = Path(tmp) / 'news'
            folder.mkdir()
            original.to_parquet(folder / 'news.parquet', index=False)
            with patch('stage4_news_relevance.plausibility.events.DATA', Path(tmp)):
                ev = ensure_events(original, folder / 'events.parquet')
                path = Path(tmp) / 'experiment.sources.json'
                corpus.write_json(path, corpus.experiment_binding(original, ev))
                corpus.assert_experiment_corpus(path)
                original.assign(id=[10, 11]).to_parquet(folder / 'news.parquet', index=False)
                with self.assertRaises(corpus.StaleNewsReferences):
                    corpus.assert_experiment_corpus(path)


class DerivedCacheTests(unittest.TestCase):
    def test_consequences_cache_uses_content_not_recycled_id(self):
        first = news().iloc[0].to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            extractor = ConsequenceExtractor.__new__(ConsequenceExtractor)
            extractor.cache = JsonlCache('consequences', folder=Path(tmp))
            response = NewsConsequences(economic=True, event='Синтетика', status='fact', significance=2)
            extractor.chain = Mock(invoke=Mock(return_value=response))
            extractor.extract(first)
            extractor.extract(dict(first, id=999))
            self.assertEqual(extractor.chain.invoke.call_count, 1)
            extractor.extract(dict(first, text='Изменённый текст при прежнем ID'))
            self.assertEqual(extractor.chain.invoke.call_count, 2)

    def test_event_summary_invalidates_when_article_changes(self):
        reports = news().to_dict('records')
        with tempfile.TemporaryDirectory() as tmp:
            summarizer = EventSummarizer.__new__(EventSummarizer)
            summarizer.min_reports = 2
            summarizer.cache = JsonlCache('summary', folder=Path(tmp))
            summarizer.chain = Mock(invoke=Mock(return_value=EventSummary(title='Событие', summary='Синтетика')))
            summarizer.summarize(reports)
            summarizer.summarize([dict(r, id=r['id'] + 100) for r in reports[::-1]])
            self.assertEqual(summarizer.chain.invoke.call_count, 1)
            summarizer.summarize([dict(reports[0], title='Изменённый заголовок'), reports[1]])
            self.assertEqual(summarizer.chain.invoke.call_count, 2)

    def test_vectors_remain_on_the_requested_snapshot_during_rebuild(self):
        import common.embed as embed
        original = news()
        updated = original.assign(id=original.url.map(corpus.stable_id))
        with tempfile.TemporaryDirectory() as tmp, patch.object(embed, 'DATA', Path(tmp)), patch.object(
                embed, 'news_pool', return_value=updated), patch.object(embed, 'encode', return_value=np.ones((2, 3))):
            embed.news_embeddings.cache_clear()
            embed.news_embeddings('synthetic', news=original)
            embed.news_embeddings('synthetic', news=updated)
            for frame in (original, updated):
                path = Path(tmp) / 'news/versions' / corpus_fingerprint(frame) / 'emb_synthetic.npz'
                with np.load(path, allow_pickle=False) as z:
                    self.assertEqual(z['ids'].tolist(), frame.id.astype(str).tolist())
                    self.assertEqual(str(z['fingerprint']), corpus_fingerprint(frame))
            embed.news_embeddings.cache_clear()


if __name__ == '__main__':
    unittest.main()
