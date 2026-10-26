"""The current async graph publishes checked writers, keeps the dedup gate and final bank quota."""
import asyncio
import contextvars
import copy
from types import SimpleNamespace

from backend.progress import RunProgress
from common import progress
from pipeline import InsightPipeline, build_e2e_graph


def test_async_writers_emit_after_audit_in_rank_order_without_waiting_for_other_writers():
    async def scenario():
        pipe=InsightPipeline.__new__(InsightPipeline)
        gate=asyncio.Event()
        first=asyncio.Event()
        published=[]
        collector=RunProgress(published.append,copy.deepcopy)
        p=progress.Progress(on_results=progress.results)
        # Plain publisher here so the test can inspect updates without a job queue.
        p.on_event=collector;p.on_results=published.append
        progress.bind(p)
        pipe._record=lambda news,**result:dict(news_id=news['id'],**result)
        async def write(item):
            progress.phase('write')
            if item['pos']==0: await gate.wait()
            progress.phase('check')
            progress.phase('audit')
            return dict(status='ready',insight=dict(id=f"card-{item['pos']}")),item['news']
        pipe._write=write
        def collect(event,**data):
            collector(event,**data)
            if event=='row': first.set()
        p.on_event=collect
        p.expect(2)
        try:
            tasks=[asyncio.create_task(pipe._write_node(dict(pos=n,news={'id':n},ctx=contextvars.copy_context()))) for n in range(2)]
            await asyncio.wait_for(first.wait(),3)
            assert published[-1][0]['insight']['id']=='card-1'
            assert not tasks[0].done()
            gate.set();await asyncio.gather(*tasks)
            assert [row['insight']['id'] for row in published[-1]]==['card-0','card-1']
            assert p.snapshot()['code']=='verifying'
        finally:
            gate.set();progress.bind(None)
    asyncio.run(scenario())


def test_graph_live_merge_uses_the_existing_cross_source_dedup_and_saves_identical_rows():
    published=[];saved=[]
    p=progress.Progress(on_results=published.append)
    p.on_event=RunProgress(progress.results,copy.deepcopy)
    progress.bind(p)
    news=dict(status='ready',insight=dict(id='news'))
    bank=dict(status='ready',insight=dict(id='bank-surviving-dedup'))
    def run(*args,**kwargs):
        progress.enter('generating')
        progress.emit('row',key=(1,0),row=news)
        return [news]
    def merge_bank(run,error,rows):
        assert rows==[news]
        assert published==[[news]]  # no tentative bank card published before dedup/quota
        return [bank]
    try:
        graph=build_e2e_graph(SimpleNamespace(run=run),with_analytics=True,
            run_analytics_fn=lambda:('bank-run',''),analytics_rows_fn=merge_bank,
            migrate_fn=copy.deepcopy,save_fn=lambda rows:saved.append(rows))
        result=graph.invoke(dict(company_id='U_1',role='cfo',n_candidates=2,n_judge=2,retrieval='lexical',workers=2))
        assert published[-1]==saved[0]==result['results']==[news,bank]
        assert p.snapshot()['code']=='finalizing'
    finally: progress.bind(None)
