"""Prepared data and optional presentation endpoints must remain read-only and reader-scoped."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import NEWS_CACHE, cached_job
from backend.service import Service
from common.evidence import EvidenceSnapshot
from scripts_import_precomputed import checked

GOOD=['Почему вырос риск?','Что проверить первым?','Когда пересчитать сценарий?']


def test_suggestions_are_private_and_cached_reads_never_initialize_models(tmp_path):
    models=Mock(side_effect=AssertionError('No inference on read'))
    service=Service(tmp_path,models_factory=models)
    job=service.agents.submit('analysis',dict(company_id='U_1',role='cfo',user_id='alice'),
        lambda:dict(results=[dict(status='ready',insight=dict(id='i1',title='Test',role='cfo'))]))
    service.agents.wait(job['id'])
    folder=tmp_path/'suggested';folder.mkdir()
    path=folder/f'{job["id"]}.json'
    path.write_text(json.dumps(dict(i1=GOOD,other=GOOD)),encoding='utf-8')
    with TestClient(create_app(service)) as client:
        url=f'/v1/jobs/{job["id"]}/suggested'
        assert client.get(url,headers={'X-User-ID':'bob'}).status_code==404
        assert client.get(url).status_code==404
        assert client.get(url,headers={'X-User-ID':'alice'}).json()==dict(i1=GOOD)
        path.write_text(json.dumps(dict(i1=['Repeated?']*3)))
        assert client.get(url,headers={'X-User-ID':'alice'}).json()=={}
        path.write_text('{broken')
        assert client.get(url,headers={'X-User-ID':'alice'}).json()=={}
    models.assert_not_called()


def test_bundled_snapshot_manifest_matches_frozen_reader_evidence_and_saved_questions(tmp_path):
    manifest=json.loads((NEWS_CACHE/'precomputed-manifest.json').read_text(encoding='utf-8'))
    models=Mock(side_effect=AssertionError('Prepared feeds require no models'))
    service=Service(tmp_path,models_factory=models)
    try:
        for name,counts in manifest['feeds'].items():
            company,_,role=name.rpartition('_')
            job=cached_job(f'demo-{company}-{role}',tmp_path)
            rows=job['result']['results']
            # Runtime/static analytics may add cards; every imported row is independently checked.
            source=json.loads((NEWS_CACHE/f'{name}.json').read_text(encoding='utf-8'))
            assert len(source['result']['results'])==counts['cards']
            for row in source['result']['results']:
                assert checked(row,company,role) is None
                ev=EvidenceSnapshot.model_validate(row['evidence'])
                assert (ev.company_id,ev.role)==(company,role)
            assert len(rows)>=counts['cards']
            saved=service.suggest(job['id'],job=job,cache_only=True)
            assert len(saved)==counts['questions']
            assert len(service.compact(job['id'],job=job,cache_only=True))==counts['compact']
        # Existing CEO cards must also receive incoming questions, even though their feed was not imported again.
        job=cached_job('demo-U_14366-ceo',tmp_path)
        saved=service.suggest(job['id'],job=job,cache_only=True)
        assert {'85b35bb40d0d','9fabeb378408','6d51facc3b20'} <= saved.keys()
        assert not (tmp_path/'suggested').exists()
        assert not (tmp_path/'compact').exists()
    finally:service.close()
    models.assert_not_called()
