"""Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from typing import Literal

import pymupdf
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.jobs import QueueFull
from backend.cabinet import Precomputed, cached_job, cabinet_feed, catalog as cabinet_catalog, search_companies
from backend.insight_previews import detail, metadata, preview_feed
from backend.schemas import AnalysisRequest, AskRequest, BatchRequest, PreparationRequest, SearchRequest, JobResponse
from backend.service import Service
from backend.payments import PaymentAnalytics
from common.roles import roles
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR

def create_app(service=None, cors_origins=None, payment_analytics=None, staff_token=None):
    payment_reports = payment_analytics if payment_analytics is not None else PaymentAnalytics()
    @asynccontextmanager
    async def lifespan(app):
        app.state.service=service or Service(checkpointing=True, agent_workers=1, auto_compact=True)
        # Warm/import each original report independently in the background; uploaded PDFs never touch that folder.
        if (service is None and os.environ.get('CABINET_MODE', 'full') != 'server'
                and os.environ.get('API_PREPARE_ON_START', '1') == '1'):
            for pdf in sorted(PDF_DIR.glob('*.pdf')):
                try: app.state.service.prepare(pdf,PreparationOptions(),PDF_DIR)
                except QueueFull: break
        app.state.precomputed=Precomputed(cabinet_jobs, lambda: app.state.service.root).start()
        try:
            yield
        finally:
            import asyncio
            await asyncio.to_thread(app.state.precomputed.stop)
            await asyncio.to_thread(app.state.service.close)

    app=FastAPI(title='AlfaFuture Insight API',version='1.0.0',lifespan=lifespan,
                description='Background agent analyses and PDF preparation. Poll job status/results. Swagger exposes every pipeline switch.')

    origins=cors_origins if cors_origins is not None else [origin.strip() for origin in os.environ.get(
        'API_CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000').split(',') if origin.strip()]
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,
                       allow_methods=['GET','POST','OPTIONS'],allow_headers=['Content-Type','Authorization','X-User-ID'],
                       expose_headers=['Content-Disposition'])

    User = Annotated[str | None, Header(min_length=1, max_length=100, pattern=r'^[\w.@-]+$')]

    def with_user(request, user_id):
        if user_id is None:
            return request
        if request.user_id is not None and request.user_id != user_id:
            raise HTTPException(422, 'user_id and X-User-ID must match')
        return request.model_copy(update={'user_id':user_id})

    def svc(): return app.state.service
    from backend.export_routes import create_export_router
    app.include_router(create_export_router(svc,staff_token))
    def require_inference():
        if os.environ.get('CABINET_MODE', 'full') == 'server':
            raise HTTPException(403, 'This server only serves prepared demo accounts')
    def submit(work):
        try: return work()
        except QueueFull as exc: raise HTTPException(429,str(exc)) from exc

    def prepare_paths(paths,options,seed=None):
        require_inference()
        jobs=[]
        for i,path in enumerate(paths):
            try: jobs.append(svc().prepare(path,options,seed))
            except QueueFull:
                return dict(jobs=jobs,remaining_files=[p.name for p in paths[i:]],queue_full=True)
        return dict(jobs=jobs,remaining_files=[],queue_full=False)

    @app.get('/health',tags=['system'])
    def health(): return {'status':'ok'}

    @app.get('/v1/options',tags=['system'])
    def options(): return dict(roles=roles(),engines=['judge'],retrieval=['v2','v3','hybrid','lexical'],sources=['news','analytics','both'],
                              chart_modes=['all','selective','off'])

    @app.post('/v1/analyses',status_code=202,response_model=JobResponse,tags=['agents'],summary='Run one company/role analysis with all CLI flags')
    def analyze(request: AnalysisRequest, x_user_id: User = None):
        require_inference()
        request = with_user(request, x_user_id)
        return submit(lambda:svc().analyze(request))

    @app.post('/v1/analyses/batch',status_code=202,tags=['agents'],summary='Queue independent companies/roles/scenarios in parallel')
    def batch(request: BatchRequest, x_user_id: User = None):
        require_inference()
        # Every item has its own durable status; a queue limit returns accepted jobs explicitly.
        jobs=[]
        items = [with_user(item, x_user_id) for item in request.items]
        for item in items:
            try: jobs.append(svc().analyze(item))
            except QueueFull: return dict(jobs=jobs,remaining=len(request.items)-len(jobs),queue_full=True)
        return dict(jobs=jobs,remaining=0)

    @app.get('/v1/jobs/{job_id}',response_model=JobResponse,tags=['jobs'],summary='Status, result or error of an analysis/PDF job')
    def job(job_id: str, x_user_id: User = None, include_result: bool = True):
        for queue in (svc().agents,svc().pdfs):
            try:
                result = queue.get(job_id, project=None if include_result else metadata)
                if result['request'].get('user_id') not in (None, x_user_id):
                    raise HTTPException(404,'Unknown job')
                return result
            except KeyError: pass
        raise HTTPException(404,'Unknown job')

    def insight_projection(job_id, user_id, project):
        def owned(job):
            if job['request'].get('user_id') not in (None, user_id):
                raise HTTPException(404, 'Unknown job')
            return project(job)
        try:
            if job_id.startswith('demo-'):
                return project(cached_job(job_id, svc().root))
            return svc().agents.get(job_id, project=owned)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, 'Unknown insight')

    @app.get('/v1/jobs/{job_id}/events',tags=['jobs'],response_class=StreamingResponse,
             summary='Live stages and checked insight previews; full snapshot on reconnect')
    async def job_events(job_id: str, x_user_id: User = None, heartbeat: float = Query(10,ge=1,le=60)):
        import asyncio
        from backend.insight_previews import DONE
        # The same ownership check protects previews, full cards and the event stream.
        insight_projection(job_id,x_user_id,metadata)
        if job_id.startswith('demo-'):
            raise HTTPException(404,'Saved demo feeds do not have a live event stream')
        def project(job):
            from backend.cabinet import cabinet_feed
            request=job['request']
            # Apply the same source checks/dedup as the normal cabinet feed.
            feed=cabinet_feed(request['company_id'],request['role'],[job],x_user_id,svc().root)
            own=preview_feed(job)
            own['insights']=[card for card in feed['insights'] if card['job_id']==job_id]
            return dict(metadata(job),feed=own)
        async def events():
            version=-1
            while True:
                seen,item=await asyncio.to_thread(svc().agents.wait_change,job_id,version,
                                                   0 if version<0 else heartbeat,project)
                if seen!=version:
                    version=seen
                    yield f'id: {version}\nevent: job\ndata: '+json.dumps(item,ensure_ascii=False,default=str)+'\n\n'
                else:
                    yield ': keep-alive\n\n'
                if item['status'] in DONE: return
        return StreamingResponse(events(),media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get('/v1/jobs/{job_id}/insights',tags=['jobs'])
    def insight_previews(job_id: str, x_user_id: User = None):
        return insight_projection(job_id, x_user_id, preview_feed)

    @app.get('/v1/jobs/{job_id}/insights/{result_index}',tags=['jobs'])
    def insight_detail(job_id: str, result_index: int, x_user_id: User = None, insight_id: str | None = None):
        return insight_projection(job_id, x_user_id, lambda job: detail(job, result_index, insight_id))

    @app.get('/v1/jobs/{job_id}/compact',tags=['jobs'],summary='Screen versions of the insights of a finished analysis')
    async def compact(job_id: str, x_user_id: User = None):
        import asyncio
        job = insight_projection(job_id, x_user_id, lambda job: job)
        return await asyncio.to_thread(svc().compact, job_id, job=job,
                                       cache_only=os.environ.get('CABINET_MODE', 'full') == 'server')

    @app.get('/v1/jobs',tags=['jobs'])
    def jobs(x_user_id: User = None, include_results: bool = True):
        project = None if include_results else metadata
        analyses = [svc().agents.get(i, project=project) for i in list(svc().agents.items)]
        return dict(analyses=[job for job in analyses if job['request'].get('user_id') in (None, x_user_id)],
                    reports=[svc().pdfs.get(i, project=project) for i in list(svc().pdfs.items)])

    @app.get('/v1/calendar',tags=['calendar'],summary='Dated insight events by range, from the small SQLite index')
    def calendar(date_from: str = Query(alias='from'), date_to: str = Query(alias='to'),
                 company_id: str | None = None):
        from backend.calendar_index import events_between
        try:
            return dict(events=events_between(date_from, date_to, company_id))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/v1/reports',tags=['reports'],summary='Atomically published report versions')
    def reports():
        with svc().catalog.lock: return dict(active=dict(svc().catalog.current))

    def cabinet_jobs():
        with svc().agents.lock:
            return tuple(dict(job) for job in svc().agents.items.values())

    @app.get('/v1/cabinet/accounts',tags=['companies'])
    def cabinet_accounts(x_user_id: User = None):
        return cabinet_catalog(cabinet_jobs(), x_user_id, svc().root, extra=app.state.precomputed.accounts)

    @app.get('/v1/cabinet/precomputed',tags=['companies'],summary='Background check of finished runs: which companies are showable from saved data')
    def cabinet_precomputed():
        pre=app.state.precomputed
        return dict(checked_at=pre.checked_at,items=pre.report)

    @app.get('/v1/cabinet/feed',tags=['companies'])
    def reader_feed(company_id: str, role: str, x_user_id: User = None):
        try:
            return cabinet_feed(company_id, role, cabinet_jobs(), x_user_id, svc().root)
        except ValueError:
            raise HTTPException(422, 'Invalid company or role')

    @app.get('/v1/cabinet/companies',tags=['companies'])
    def cabinet_companies(q: str = ''):
        if len(q) > 120:
            raise HTTPException(422, 'Search query is too long')
        try:
            return dict(items=search_companies(q))
        except FileNotFoundError:
            raise HTTPException(503, 'Company case tables are not installed')

    @app.post('/v1/reports/prepare',status_code=202,tags=['reports'],summary='Prepare existing reports in parallel with per-report caches and budgets')
    def prepare(request: PreparationRequest):
        options=PreparationOptions(**request.model_dump())
        return prepare_paths(sorted(PDF_DIR.glob('*.pdf')),options,PDF_DIR)

    @app.post('/v1/reports/upload',status_code=202,tags=['reports'],summary='Upload PDFs; prepare and publish in background without interrupting agents')
    async def upload(files: Annotated[list[UploadFile],File(description='One or several PDFs, maximum 50 MiB each')],
                     options: Annotated[str,Form(description='JSON PreparationRequest: charts, tables, digests and per-report budgets')]='{}'):
        require_inference()
        try: prep=PreparationRequest.model_validate_json(options)
        except ValidationError as exc: raise HTTPException(422,jsonable_encoder(exc.errors(include_context=False))) from exc
        if len(files)>32: raise HTTPException(413,'At most 32 reports per upload')
        pending=[]
        for file in files:
            name=Path((file.filename or '').replace('\\','/')).name
            if not name.lower().endswith('.pdf'): raise HTTPException(422,'Only PDF files are supported')
            content=await file.read(50*1024*1024+1)
            await file.close()
            if len(content)>50*1024*1024: raise HTTPException(413,'PDF exceeds 50 MiB')
            try:
                with pymupdf.open(stream=content,filetype='pdf') as doc:
                    if doc.is_encrypted or not len(doc): raise ValueError('Encrypted or empty PDF')
            except Exception as exc: raise HTTPException(422,'Invalid PDF') from exc
            folder=svc().root/'uploads'/hashlib.sha256(content).hexdigest();folder.mkdir(parents=True,exist_ok=True)
            path=folder/name
            if not path.exists(): path.write_bytes(content)
            pending.append(path)
        return prepare_paths(pending,PreparationOptions(**prep.model_dump()))

    @app.post('/v1/search/news',tags=['search'])
    def news_search(request: SearchRequest):
        require_inference()
        from common.news_snapshot import news_snapshot
        from stage5_insight_detection.news_enrichment.related_news import RelatedNews
        with news_snapshot():
            result=RelatedNews().search(request.query,request.k)
            return jsonable_encoder(dict(items=result.to_dict('records'),diagnostics=result.attrs))

    @app.post('/v1/search/bank',tags=['search'])
    def bank_search(request: SearchRequest):
        require_inference()
        version,library,bank=svc().catalog.snapshot(svc().models_factory())
        return dict(version=version,items=bank.search(request.query,request.k))

    @app.post('/v1/insights/ask', response_class=StreamingResponse, tags=['agents'],
              summary='Вопрос по готовому инсайту: ответ стримится токен за токеном')
    def ask_question(body: AskRequest, x_user_id: User = None):
        require_inference()
        from backend.service import Service  # noqa: F401 - the request reaches the same service state
        async def gen():
            try:
                async for chunk in svc().ask(body.company_id, body.role, body.question, body.insight_title, x_user_id):
                    yield chunk
            except Exception as exc:
                yield f'\n[ошибка] {type(exc).__name__}: {exc}'
        return StreamingResponse(gen(), media_type='text/plain; charset=utf-8')

    @app.get('/v1/companies/{company_id}',tags=['companies'])
    def company(company_id: str):
        from stage2_company_facts.profile.integration import company_inputs
        from stage2_company_facts.profile.load import load_profile
        from backend.company_profile import display_profile
        try:
            profile = load_profile(company_id)
            result = company_inputs(profile)
            return jsonable_encoder(dict(card=result.card, facts=result.facts, profile=display_profile(profile)))
        except (KeyError,ValueError,FileNotFoundError) as exc: raise HTTPException(404,'Company not found') from exc

    def payment_result(method, company_id, *args, **kwargs):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',company_id):
            raise HTTPException(422,'Invalid company_id')
        try:
            return method(company_id,*args,**kwargs)
        except KeyError as exc:
            raise HTTPException(404,'Компания или платёжная связь не найдена') from exc
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(503,'Платёжные данные не установлены на сервере') from exc
        except ValueError as exc:
            raise HTTPException(503,'Не удалось рассчитать аналитику: проверьте платёжные данные') from exc

    @app.get('/v1/companies/{company_id}/counterparties',tags=['companies'])
    def counterparties(company_id: str, x_user_id: User = None, direction: Literal['all','in','out','both'] = 'all',
                       q: str = Query('',max_length=120), group: Literal['all','own'] = 'all',
                       sort: Literal['amount','in','out','count','name'] = 'amount',
                       offset: int = Query(0,ge=0), limit: int = Query(15,ge=1,le=100)):
        return payment_result(payment_reports.counterparties,company_id,direction,q,group,sort,offset,limit)

    @app.get('/v1/companies/{company_id}/counterparties/{counterparty_id}',tags=['companies'])
    def counterparty(company_id: str, counterparty_id: str, x_user_id: User = None):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',counterparty_id):
            raise HTTPException(422,'Invalid counterparty_id')
        return payment_result(payment_reports.counterparty,company_id,counterparty_id)

    @app.get('/v1/companies/{company_id}/payments',tags=['companies'])
    def company_payments(company_id: str, x_user_id: User = None, direction: Literal['in','out'] = 'out',
                         q: str = Query('',max_length=120), sort: Literal['amount','count'] = 'amount',
                         offset: int = Query(0,ge=0), limit: int = Query(15,ge=1,le=100)):
        return payment_result(payment_reports.report,company_id,direction,q,sort,offset,limit)
    return app

app=create_app()
