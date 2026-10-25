"""Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pymupdf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.jobs import QueueFull
from backend.cabinet import cached_job, cabinet_feed, catalog as cabinet_catalog, search_companies
from backend.insight_previews import detail, metadata, preview_feed
from backend.schemas import AnalysisRequest, AskRequest, BatchRequest, PreparationRequest, SearchRequest, JobResponse
from backend.service import Service
from common.roles import roles
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR

def create_app(service=None, cors_origins=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service=service or Service(checkpointing=True, agent_workers=1, auto_compact=True)
        # Warm/import each original report independently in the background; uploaded PDFs never touch that folder.
        if (service is None and os.environ.get('CABINET_MODE', 'full') != 'server'
                and os.environ.get('API_PREPARE_ON_START', '1') == '1'):
            for pdf in sorted(PDF_DIR.glob('*.pdf')):
                try: app.state.service.prepare(pdf,PreparationOptions(),PDF_DIR)
                except QueueFull: break
        yield
        import asyncio
        await asyncio.to_thread(app.state.service.close)

    app=FastAPI(title='AlfaFuture Insight API',version='1.0.0',lifespan=lifespan,
                description='Background agent analyses and PDF preparation. Poll job status/results. Swagger exposes every pipeline switch.')

    origins=cors_origins if cors_origins is not None else [origin.strip() for origin in os.environ.get(
        'API_CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000').split(',') if origin.strip()]
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,
                       allow_methods=['GET','POST','OPTIONS'],allow_headers=['Content-Type','Authorization','X-User-ID'])

    User = Annotated[str | None, Header(min_length=1, max_length=100, pattern=r'^[\w.@-]+$')]

    def with_user(request, user_id):
        if user_id is None:
            return request
        if request.user_id is not None and request.user_id != user_id:
            raise HTTPException(422, 'user_id and X-User-ID must match')
        return request.model_copy(update={'user_id':user_id})

    def svc(): return app.state.service
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

    @app.get('/v1/reports',tags=['reports'],summary='Atomically published report versions')
    def reports():
        with svc().catalog.lock: return dict(active=dict(svc().catalog.current))

    def cabinet_jobs():
        with svc().agents.lock:
            return tuple(dict(job) for job in svc().agents.items.values())

    @app.get('/v1/cabinet/accounts',tags=['companies'])
    def cabinet_accounts(x_user_id: User = None):
        return cabinet_catalog(cabinet_jobs(), x_user_id, svc().root)

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
    return app

app=create_app()
