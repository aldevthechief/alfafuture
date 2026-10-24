"""Offline integration regressions for chunk retrieval, literal evidence and independent updates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from common.chunk_search import ChunkIndex, article_context, attach_fragments, chunk_index
from common.news_chunks import split_into_chunks
from common.news_corpus import stable_id
from common.news_evidence import verify_claim_evidence
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage1_news_pool.updater import merge_news, refresh_once, update_lock, serve
from stage3_news_search.retrieval import event_news
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage4_news_relevance.relevance_scoring.relevance import RelevanceScorer
from stage4_news_relevance.news_classifier.classifier import NewsClassifier
from stage5_insight_detection.analyst_loop.event import EventCard
from stage6_insight_generation.insight_writer.insight import Claim, Insight, SourceExcerpt
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter

QUOTE = 'Ключевая ставка снижена на 2 п.п. до 14%.'


def row(url='https://example.org/a', text=QUOTE):
    return dict(id=stable_id(url), url=url, title='Обзор экономики', text=text,
                date='2026-10-22', source='cbr', tier=1)


def insight(evidence):
    return Insight(title='Ставка', role='cfo', what_happened=[Claim(text='Ставка снижена',
                   sources=[f"news:{row()['id']}"], evidence=evidence)], why_it_matters=[],
                   actions=[], bank_view=[], ai_conclusion='Проверить условия', confidence='low')


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.news = pd.DataFrame([row(text=('Общие сведения. ' * 300) + QUOTE + (' Прочие темы.' * 300)),
                                  row('https://example.org/b', 'Погода и урожай.')])

    def test_middle_found_with_source_offsets_and_reaches_both_scorers(self):
        index = ChunkIndex(self.news)
        self.assertGreater(index.article_scores(['ключевая ставка'])[0, 0], 0)
        selected = attach_fragments(self.news, ['ключевая ставка']).iloc[0].to_dict()
        self.assertIn(QUOTE, article_context(selected))
        for fragment in selected['matched_fragments']:
            self.assertEqual(fragment['text'], selected['text'][fragment['start']:fragment['end']])
        scorer = RelevanceScorer.__new__(RelevanceScorer)
        scorer.chain = Mock()
        scorer.score(selected, 'кредит', 'cfo')
        self.assertIn(QUOTE, scorer.chain.invoke.call_args.args[0]['text'])
        classifier = NewsClassifier.__new__(NewsClassifier)
        classifier.trace = Mock()
        self.assertIn(QUOTE, classifier._item(0, selected))
        self.assertIn('ставка', EventCard.from_news(selected).topics)
        self.assertIn(QUOTE, Evidence([selected], []).render())

    def test_best_event_member_retains_own_identity(self):
        events = pd.DataFrame([dict(event=1, news_ids=self.news.id.tolist()[::-1], n_sources=1,
                                    generators=[dict(query='ключевая ставка')], best_rank=0)])
        out = event_news(events, self.news).iloc[0]
        self.assertEqual(out.id, self.news.iloc[0].id)
        self.assertIn(QUOTE, article_context(out.to_dict()))

    def test_empty_unicode_boundaries_and_cache_invalidation(self):
        self.assertEqual(split_into_chunks(''), [])
        with self.assertRaises(ValueError):
            split_into_chunks('abc', 3, 3)
        text = 'Длинная статья 📰 ' * 30
        for start, end, fragment in split_into_chunks(text, 37, 5):
            self.assertEqual(text[start:end], fragment)
        old = chunk_index(self.news)
        changed = self.news.copy()
        changed.loc[0, 'text'] = 'Погода'
        self.assertIsNot(old, chunk_index(changed))
        blank = ChunkIndex(pd.DataFrame([row(text='!!!')]))
        self.assertEqual(blank.fragments(row()['id'], ['ставка']), [])


class EvidenceTests(unittest.TestCase):
    def test_quote_is_literal_and_offsets_are_recomputed(self):
        source = row(text='Вступление. ' + QUOTE)
        claim = insight([SourceExcerpt(source=f"news:{source['id']}", quote=QUOTE, start=0, end=1, url='fake')])
        self.assertEqual(verify_claim_evidence(claim, [source]), [])
        excerpt = claim.what_happened[0].evidence[0]
        self.assertEqual(excerpt.start, len('Вступление. '))
        self.assertEqual(source['text'][excerpt.start:excerpt.end], QUOTE)
        self.assertEqual(excerpt.url, source['url'])
        claim.what_happened[0].evidence[0].quote = 'Выдуманный факт'
        self.assertTrue(verify_claim_evidence(claim, [source]))
        self.assertTrue(claim.what_happened[0].unconfirmed)

    def test_missing_quote_marks_the_claim_but_does_not_reject(self):
        draft = insight([])
        model = ScriptedChatModel(responses=[calls(tool_call('Insight', **draft.model_dump())),
                                            calls(tool_call('Review', ok=True))])
        result, review = InsightWriter(model).write('test', 'cfo', '', Evidence([row()], []))
        self.assertTrue(review.ok)
        self.assertTrue(result.what_happened[0].unconfirmed)
        self.assertEqual(model.i, 0)

    def test_extractor_reads_beyond_lead_and_rejects_fabricated_quotes(self):
        source = row(text=('Вступление. ' * 300) + QUOTE)
        def answer(inputs):
            facts = [dict(text='Выдумано', quote='НЕТ В СТАТЬЕ')]
            if QUOTE in inputs['text']:
                facts.append(dict(text='Ставка снижена', quote=QUOTE))
            return SimpleNamespace(model_dump=lambda: dict(economic=True, event='Ставка', status='fact',
                                    significance=2, affected=[], entities={}, facts=facts))
        with tempfile.TemporaryDirectory() as tmp:
            extractor = ConsequenceExtractor.__new__(ConsequenceExtractor)
            extractor.cache = JsonlCache('test', folder=Path(tmp))
            extractor.chain = Mock()
            extractor.chain.invoke.side_effect = answer
            result = extractor.extract(source)
            self.assertEqual(len(result['facts']), 1)
            self.assertTrue(result['rejected_facts'])
            excerpt = result['facts'][0]['evidence']
            self.assertEqual(source['text'][excerpt['start']:excerpt['end']], QUOTE)
            count = extractor.chain.invoke.call_count
            extractor.extract(source)
            self.assertEqual(extractor.chain.invoke.call_count, count)
            migrated = extractor.extract(dict(source, id=999))
            self.assertEqual(extractor.chain.invoke.call_count, count)
            self.assertEqual(migrated['facts'][0]['evidence']['source'], 'news:999')


class UpdaterTests(unittest.TestCase):
    def test_merge_preserves_history_updates_url_and_is_idempotent(self):
        old = pd.DataFrame([row(), row('https://example.org/b')])
        incoming = pd.DataFrame([row('https://example.org/a?utm_source=x', 'Новый текст')])
        merged = merge_news(old, incoming)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.loc[merged.url == row()['url'], 'text'].iloc[0], 'Новый текст')
        pd.testing.assert_frame_equal(merged, merge_news(merged, incoming))

    def test_publication_reload_failure_and_empty_collect(self):
        import common.search as search
        with tempfile.TemporaryDirectory() as tmp, patch.object(search, 'DATA', Path(tmp)):
            folder = Path(tmp) / 'news'
            first = refresh_once(folder, collector=lambda *a, **k: pd.DataFrame([row()]))
            self.assertEqual(first['status'], 'updated')
            self.assertEqual(len(search.news_pool()), 1)
            self.assertEqual(len(search.news_index()[0]), 1)
            second = refresh_once(folder, collector=lambda *a, **k: pd.DataFrame([row('https://example.org/b')]))
            self.assertEqual(second['status'], 'updated')
            self.assertEqual(len(search.news_pool()), 2)
            before = (folder / 'news.parquet').read_bytes()
            self.assertEqual(refresh_once(folder, collector=lambda *a, **k: None)['status'], 'unchanged')
            def fail(*a, **k):
                raise RuntimeError('network failed')
            with self.assertRaisesRegex(RuntimeError, 'network failed'):
                refresh_once(folder, collector=fail)
            self.assertEqual(before, (folder / 'news.parquet').read_bytes())
            self.assertEqual(json.loads((folder / 'updater_status.json').read_text())['status'], 'failed')
            same = refresh_once(folder, collector=lambda *a, **k: pd.DataFrame([row('https://example.org/b')]))
            self.assertEqual(same['status'], 'unchanged')

    def test_single_writer_lock_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            with update_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    with update_lock(Path(tmp)):
                        self.fail('second writer acquired lock')
            with update_lock(Path(tmp)):
                pass

    def test_periodic_retry_after_failure_and_shutdown(self):
        stop = Mock()
        stop.is_set.side_effect = [False, False, True]
        with patch('stage1_news_pool.updater.refresh_once', side_effect=[RuntimeError('offline'), {}]) as refresh:
            with self.assertLogs('stage1_news_pool.updater', level='ERROR'):
                serve(10, stop)
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(stop.wait.call_count, 2)

    def test_cli_help_is_offline_and_pipeline_exposes_retrieval(self):
        root = Path(__file__).resolve().parents[3]
        for command in ([sys.executable, '-X', 'utf8', 'pipeline.py', '--help'],
                        [sys.executable, '-X', 'utf8', '-m', 'stage1_news_pool.updater', '--help']):
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--retrieval' if 'pipeline.py' in command else '--interval', result.stdout)


if __name__ == '__main__':
    unittest.main()
