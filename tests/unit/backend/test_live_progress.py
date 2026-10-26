"""Live checked cards, their durable stage and stream access; no model/network calls."""
import asyncio
import copy
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import cabinet_feed
from backend.insight_previews import metadata, preview_feed
from backend.jobs import Jobs
from backend.progress import RunProgress
from backend.schemas import ExportQuery
from backend.service import Service
from common import progress


def row(key):
    return dict(status='ready',news_id=int(key),insight=dict(id=f'card-{key}',title=f'Инсайт {key}',role='cfo',
        why_it_matters=[dict(text='Факт компании',sources=['fact:1'])],actions=['Проверить'],origin='news'),
        evidence=dict(company_id='U_live',role='cfo',facts=[dict(id='fact:1',text='Факт компании')],news=[],bank=[]))


def test_partial_cards_survive_failure_restart_and_keep_their_identity():
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp),models_factory=Mock(side_effect=AssertionError('LLM forbidden')),auto_compact=False)
        first=threading.Event();resume=threading.Event()
        def work():
            progress.listen(RunProgress(progress.results,lambda value:value))
            progress.enter('generating')
            progress.emit('row',key=(1,2),row=row('2'))
            first.set();resume.wait(5)
            progress.emit('row',key=(1,1),row=row('1'))
            raise RuntimeError('Synthetic failure after two checked cards')
        job=service.agents.submit('analysis',dict(company_id='U_live',role='cfo',user_id='alice'),work)
        assert first.wait(5)
        try:
            with TestClient(create_app(service)) as client:
                headers={'X-User-ID':'alice'}
                path=f"/v1/jobs/{job['id']}"
                current=service.agents.get(job['id'])
                assert current['status']=='running'
                assert len(preview_feed(current)['insights'])==1
                assert 'partial_results' not in metadata(current)
                assert 'result' not in metadata(current)
                assert client.get(path+'/events',headers={'X-User-ID':'bob'}).status_code==404
                assert client.get(path+'/insights',headers={'X-User-ID':'bob'}).status_code==404
                for role,user,expected in [('cfo','alice',1),('ceo','alice',0),('cfo','bob',0)]:
                    feed=cabinet_feed('U_live',role,[current],user,service.root)
                    assert len(feed['insights'])==expected
                # Export and compact readers accept ready rows while the job is still running.
                tables,_,_=service.export_tables(ExportQuery(kind='insights',job_id=job['id']),'alice')
                assert len(tables['insights'][1])==1
                resume.set()
                final=service.agents.wait(job['id'])
                assert final['status']=='failed' and final['stage']['code']=='failed'
                feed=client.get('/v1/cabinet/feed?company_id=U_live&role=cfo',headers=headers).json()
                assert {card['id'] for card in feed['insights']}=={'card-1','card-2'}
                # The second card moved from index 0 to 1; opening the original preview still opens card 2.
                opened=client.get(path+'/insights/0?insight_id=card-2',headers=headers).json()
                assert opened['result']['results'][0]['insight']['id']=='card-2'
                events=client.get(path+'/events',headers=headers)
                assert events.status_code==200 and events.headers['x-accel-buffering']=='no'
                payload=json.loads(events.text.split('data: ',1)[1].strip())
                assert payload['status']=='failed' and len(payload['feed']['insights'])==2
                assert 'partial_results' not in payload and 'result' not in payload
        finally:
            resume.set();service.close()
        reloaded=Jobs(Path(tmp)/'analyses')
        try:
            assert len(preview_feed(reloaded.get(job['id']))['insights'])==2
        finally: reloaded.close()


def test_sse_delivers_a_card_before_the_job_completes_and_reconnects_from_snapshot():
    async def scenario(service,app,job,ready,finish):
        endpoint=next(route.endpoint for route in app.routes if getattr(route,'path','')=='/v1/jobs/{job_id}/events')
        response=await endpoint(job['id'],x_user_id='alice',heartbeat=1)
        stream=response.body_iterator
        initial=await anext(stream)
        assert '"status": "running"' in initial
        ready.set()
        update=await asyncio.wait_for(anext(stream),3)
        assert 'card-1' in update and '"status": "running"' in update
        assert service.agents.get(job['id'])['status']=='running'
        await stream.aclose()
        # Reconnecting needs no missed event log: the first event has every ready card.
        response=await endpoint(job['id'],x_user_id='alice',heartbeat=1)
        stream=response.body_iterator
        assert 'card-1' in await anext(stream)
        finish.set()
        terminal=''
        for _ in range(8):
            terminal=await asyncio.wait_for(anext(stream),3)
            if '"status": "succeeded"' in terminal: break
        assert '"status": "succeeded"' in terminal and 'card-1' in terminal
        await stream.aclose()
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp),models_factory=lambda:None,auto_compact=False)
        ready=threading.Event();finish=threading.Event();started=threading.Event()
        def work():
            progress.enter('generating');started.set();ready.wait(5)
            progress.results([row('1')]);finish.wait(5)
            return dict(results=[row('1')])
        job=service.agents.submit('analysis',dict(company_id='U_live',role='cfo',user_id='alice'),work)
        assert started.wait(5)
        app=create_app(service);app.state.service=service
        try: asyncio.run(scenario(service,app,job,ready,finish))
        finally: ready.set();finish.set();service.close()


def test_restart_of_running_job_preserves_checked_cards_and_marks_stage_interrupted():
    with tempfile.TemporaryDirectory() as tmp:
        stage=progress.Progress();stage.enter('verifying')
        Path(tmp,'old.json').write_text(json.dumps(dict(id='old',kind='analysis',status='running',
            request=dict(company_id='U_live',role='cfo',user_id='alice'),created_at='2026-10-26',
            result=None,error=None,partial_results=[row('1')],stage=stage.snapshot())),encoding='utf-8')
        jobs=Jobs(Path(tmp))
        try:
            job=jobs.get('old')
            assert job['status']=='interrupted' and job['stage']['code']=='interrupted'
            assert len(preview_feed(job)['insights'])==1
        finally: jobs.close()


def test_collector_keeps_order_and_only_publishes_bank_cards_after_final_merge():
    published=[]
    collector=RunProgress(published.append,copy.deepcopy)
    collector('row',key=(1,2),row=row('2'))
    collector('analytics',run=object(),error='')
    collector('row',key=(1,1),row=row('1'))
    assert [value['insight']['id'] for value in published[-1]]==['card-1','card-2']
    merged=[row('1'),row('2'),dict(row('3'),source='bank_analytics')]
    collector('merged',rows=merged)
    assert published[-1]==merged
