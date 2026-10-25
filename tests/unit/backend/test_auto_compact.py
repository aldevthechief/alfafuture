"""A finished analysis gets its screen versions in a background thread, all insights in parallel, computed once."""
import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.schemas import AnalysisRequest
from backend.service import Service

ROWS = [{'status': 'ready', 'news_id': n, 'insight': {'title': f'i{n}'}} for n in (1, 2, 3)] + [{'status': 'judged'}]


class Model(SimpleNamespace):
    def model_copy(self, update):
        return self


def test_analysis_is_compacted_in_background_in_parallel_once():
    running = []

    async def acompact(self, insight):
        running.append(insight['title'])
        await asyncio.sleep(0.2)  # all three in flight together: ~0.2 s total, not 0.6
        return {'title': 'short ' + insight['title']}

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(Service, 'run', lambda self, r: {'status': 'succeeded', 'results': ROWS}), \
         patch('stage6_insight_generation.insight_compact.compact.InsightCompactor.__init__', lambda self, m: None), \
         patch('stage6_insight_generation.insight_compact.compact.InsightCompactor.acompact', acompact):
        service = Service(Path(tmp), models_factory=lambda: SimpleNamespace(agent=lambda temperature: Model()), auto_compact=True)
        job = service.analyze(AnalysisRequest(company_id='U_1', role='cfo'))
        path = Path(tmp) / 'compact' / f'{job["id"]}.json'
        start = time.time()
        while not path.exists() and time.time() - start < 5:
            time.sleep(0.05)
        assert path.exists()
        done = service.compact(job['id'])  # the API call after it: read from disk, nothing recomputed
        assert sorted(done) == ['row-0-1', 'row-1-2', 'row-2-3'] and done['row-1-2'] == {'title': 'short i2'}  # ids as api.js toFeed gives them
        assert sorted(running) == ['i1', 'i2', 'i3']
        service.close()
