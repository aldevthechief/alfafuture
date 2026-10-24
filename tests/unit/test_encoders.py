"""Encoder initialization must be serialized across all model names, without downloading weights."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.encoders import clear_encoder_cache, load_encoder


def model(*, meta_parameter=False, meta_buffer=False):
    return SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=meta_parameter)],
                           buffers=lambda: [SimpleNamespace(is_meta=meta_buffer)])


class EncoderTests(unittest.TestCase):
    def setUp(self):
        clear_encoder_cache()
        self.torch = patch.dict('sys.modules', {'torch': SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False), float16='float16')})
        self.torch.start()

    def tearDown(self):
        clear_encoder_cache()
        self.torch.stop()

    def test_concurrent_requests_share_materialized_model(self):
        instance = model()
        start = threading.Barrier(8)

        def request():
            start.wait(timeout=10)
            return load_encoder('e5', device='cpu')

        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(
                SentenceTransformer=Mock(return_value=instance))}):
            import sentence_transformers
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: request(), range(8)))
            sentence_transformers.SentenceTransformer.assert_called_once()
        self.assertTrue(all(result is instance for result in results))
        self.assertEqual(instance.max_seq_length, 512)

    def test_different_models_also_serialize_global_initialization(self):
        first_entered, release_first, second_requested, second_entered = (
            threading.Event() for _ in range(4))

        def construct(name, **kwargs):
            if name == 'e5':
                first_entered.set()
                if not release_first.wait(timeout=10):
                    raise TimeoutError('Test did not release first constructor')
            else:
                second_entered.set()
            return model()

        def second():
            second_requested.set()
            return load_encoder('qwen', device='cpu')

        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(
                SentenceTransformer=construct)}), ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(load_encoder, 'e5', device='cpu')
            try:
                self.assertTrue(first_entered.wait(timeout=10))
                other = executor.submit(second)
                self.assertTrue(second_requested.wait(timeout=10))
                self.assertFalse(second_entered.wait(timeout=0.1))
            finally:
                release_first.set()
            self.assertIsNotNone(first.result(timeout=10))
            self.assertIsNotNone(other.result(timeout=10))
        self.assertTrue(second_entered.is_set())

    def test_incompatible_tokenizer_or_length_does_not_reuse_model(self):
        factory = Mock(side_effect=lambda *a, **kw: model())
        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(
                SentenceTransformer=factory)}):
            default = load_encoder('e5', device='cpu')
            self.assertIs(default, load_encoder('e5', device='cpu', tokenizer_kwargs={}))
            left = load_encoder('e5', device='cpu', tokenizer_kwargs={'padding_side': 'left'})
            short = load_encoder('e5', device='cpu', max_seq_length=128)
        self.assertIsNot(default, left)
        self.assertIsNot(default, short)
        self.assertEqual(factory.call_count, 3)

    def test_failed_or_meta_load_is_not_cached(self):
        for bad in [OSError('incomplete checkpoint'), model(meta_parameter=True), model(meta_buffer=True)]:
            with self.subTest(bad=bad):
                clear_encoder_cache()
                good = model()
                factory = Mock(side_effect=[bad, good])
                with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(
                        SentenceTransformer=factory)}):
                    with self.assertRaises((OSError, RuntimeError)):
                        load_encoder('e5', device='cpu')
                    self.assertIs(load_encoder('e5', device='cpu'), good)
                    self.assertIs(load_encoder('e5', device='cpu'), good)
                self.assertEqual(factory.call_count, 2)


if __name__ == '__main__':
    unittest.main()
