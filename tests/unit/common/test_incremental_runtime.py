"""Offline checks of per-article reuse and a corpus update during analysis."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from common.news_snapshot import news_snapshot
from common.incremental_vectors import vectors
from common.search import news_document
from stage1_news_pool.incremental import fetch_change
from test_hybrid_news_search import NEWS

class IncrementalTests(unittest.TestCase):
    def test_vectors_encode_only_new_or_changed_text_and_preserve_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            encoder=Mock(side_effect=lambda texts,*args,**kw: np.array([[len(t),1.] for t in texts],dtype=np.float32))
            first=vectors(NEWS,'test',folder,encoder,news_document(NEWS).tolist())
            self.assertEqual(len(encoder.call_args.args[0]),3)
            reordered=NEWS.iloc[::-1]
            second=vectors(reordered,'test',folder,encoder,news_document(reordered).tolist())
            self.assertEqual(encoder.call_count,1)
            np.testing.assert_array_equal(second,first[::-1])
            changed=NEWS.copy()
            changed.loc[1,'text']='Совсем другой текст статьи'
            third=vectors(changed,'test',folder,encoder,news_document(changed).tolist())
            self.assertEqual(len(encoder.call_args.args[0]),1)
            np.testing.assert_array_equal(third[[0,2]],first[[0,2]])
            changed.loc[0,'id']=999
            vectors(changed,'test',folder,encoder,news_document(changed).tolist())
            self.assertEqual(encoder.call_count,2)
            vectors(changed,'another-model',folder,encoder,news_document(changed).tolist())
            self.assertEqual(len(encoder.call_args.args[0]),3)

    def test_conditional_http_and_body_hash_skip_parser(self):
        record={'url':'https://example.org/news','source':'test'}
        old={'etag':'old-tag','modified':'yesterday','digest':hashlib.sha256(b'page').hexdigest()}
        session=Mock()
        parser=Mock(return_value={'text':'new article','title':'new'})
        session.get.return_value=SimpleNamespace(status_code=304)
        self.assertIsNone(fetch_change(record,old,session,parser)[0])
        self.assertEqual(session.get.call_args.kwargs['headers']['If-None-Match'],'old-tag')
        parser.assert_not_called()
        response=Mock(status_code=200,content=b'page',text='page',headers={},encoding='utf-8')
        session.get.return_value=response
        self.assertIsNone(fetch_change(record,old,session,parser)[0])
        parser.assert_not_called()
        response.content=b'new page'
        raw,state=fetch_change(record,old,session,parser)
        self.assertEqual(raw['text'],'new article')
        parser.assert_called_once()
        session.get.return_value=SimpleNamespace(status_code=304)
        with self.assertRaises(ValueError):
            fetch_change(record,None,session,parser)

    def test_analysis_keeps_old_snapshot_while_live_corpus_changes(self):
        import common.search as search
        with tempfile.TemporaryDirectory() as tmp,patch.object(search,'DATA',Path(tmp)):
            path=Path(tmp)/'news/news.parquet'
            path.parent.mkdir()
            NEWS.to_parquet(path)
            with news_snapshot():
                original=search.news_pool()
                index=search.news_index()
                NEWS.head(1).to_parquet(path)
                self.assertIs(search.news_pool(),original)
                self.assertIs(search.news_index(),index)
                with news_snapshot():
                    self.assertEqual(len(search.news_pool()),3)
            self.assertEqual(len(search.news_pool()),1)
            self.assertEqual(len(search.news_index()[0]),1)

    def test_collector_reuses_cleaned_articles_and_recovers_unpublished_state(self):
        from stage1_news_pool.incremental import collect_changed
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            stage=folder/'staging'
            stage.mkdir()
            records=[dict(url=NEWS.iloc[0].url,source='test')]
            frame=NEWS.head(1)
            def build():
                frame.to_parquet(stage/'news.parquet')
            fetch=Mock(return_value=(dict(url=NEWS.iloc[0].url,text='raw'),dict(digest='hash',etag='tag')))
            with patch('stage1_news_pool.incremental.fetch_change',fetch), patch('stage1_news_pool.incremental.ProcessPoolExecutor'):
                first=collect_changed(stage,records,build)
                # Before publication, a failed cycle must not reuse its prepared-page acknowledgement.
                collect_changed(stage,records,build)
                self.assertIsNone(fetch.call_args.args[1])
                first.to_parquet(folder/'news.parquet')
                (stage/'news.parquet').unlink()
                fetch.side_effect=lambda rec,old,**kw: (None,old)
                build_mock=Mock(side_effect=AssertionError('unchanged article must not be cleaned'))
                self.assertIsNone(collect_changed(stage,records,build_mock))
                self.assertEqual(fetch.call_args.args[1]['etag'],'tag')
                build_mock.assert_not_called()

    def test_snapshot_released_after_exception(self):
        from common.news_snapshot import ACTIVE
        with self.assertRaises(RuntimeError):
            with news_snapshot(NEWS):
                raise RuntimeError('failed analysis')
        self.assertIsNone(ACTIVE.get())

if __name__=='__main__':
    unittest.main()
