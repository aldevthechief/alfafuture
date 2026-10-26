"""Derived assessments reach cached previews/details/exports without changing source caches or calling LLMs."""
import tempfile
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import CACHE,NEWS_CACHE
from backend.service import Service


def test_saved_feed_detail_and_export_share_the_assessment_and_keep_original_files():
    paths=[CACHE/'U_14366_cfo.json',NEWS_CACHE/'U_14366_cfo.json']
    before={path:path.read_bytes() for path in paths if path.exists()}
    with tempfile.TemporaryDirectory() as tmp:
        models=Mock(side_effect=AssertionError('No LLM calls on cabinet reads'))
        service=Service(Path(tmp),models_factory=models,auto_compact=False)
        with TestClient(create_app(service)) as client:
            headers={'X-User-ID':'reader'}
            feed=client.get('/v1/cabinet/feed?company_id=U_14366&role=cfo',headers=headers).json()
            assert feed['insights']
            export=client.get('/v1/exports/insights?company_id=U_14366&role=cfo&format=json',headers=headers).json()
            exported={row['insight_id']:row for row in export['tables']['insights']}
            for card in feed['insights']:
                assert card['confidence_basis']
                detail=client.get(f"/v1/jobs/{card['job_id']}/insights/{card['result_index']}?insight_id={card['id']}",headers=headers).json()
                row=detail['result']['results'][0]
                assert row['insight']['confidence']==card['confidence']
                assert row['business_value']['confidence_basis']==card['confidence_basis']
                assert row.get('evidence')
                assert exported[card['id']]['confidence']==card['confidence']
                assert exported[card['id']]['bank_product']==row['insight']['bank_product']
            models.assert_not_called()
    assert all(path.read_bytes()==original for path,original in before.items())
