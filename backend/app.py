"""Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.jobs import QueueFull
from backend.schemas import AnalysisRequest, AskRequest, BatchRequest, PreparationRequest, SearchRequest, JobResponse
from backend.service import Service
from common.roles import roles
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR

def create_app(service=None, cors_origins=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service=service or Service(checkpointing=True)
        # Warm/import each original report independently in the background; uploaded PDFs never touch that folder.
        if service is None:
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
                       allow_methods=['GET','POST','OPTIONS'],allow_headers=['Content-Type','Authorization'])

    def svc(): return app.state.service
    def submit(work):
        try: return work()
        except QueueFull as exc: raise HTTPException(429,str(exc)) from exc

    def prepare_paths(paths,options,seed=None):
        jobs=[]
        for i,path in enumerate(paths):
            try: jobs.append(svc().prepare(path,options,seed))
            except QueueFull:
                return dict(jobs=jobs,remaining_files=[p.name for p in paths[i:]],queue_full=True)
        return dict(jobs=jobs,remaining_files=[],queue_full=False)

    @app.get('/health',tags=['system'])
    def health(): return {'status':'ok'}

    @app.get('/v1/options',tags=['system'])
    def options(): return dict(roles=roles(),engines=['judge','loop'],retrieval=['v2','v3','hybrid','lexical'],sources=['news','analytics','both'])

    @app.post('/v1/analyses',status_code=202,response_model=JobResponse,tags=['agents'],summary='Run one company/role analysis with all CLI flags')
    def analyze(request: AnalysisRequest): return submit(lambda:svc().analyze(request))

    @app.post('/v1/analyses/batch',status_code=202,tags=['agents'],summary='Queue independent companies/roles/scenarios in parallel')
    def batch(request: BatchRequest):
        # Every item has its own durable status; a queue limit returns accepted jobs explicitly.
        jobs=[]
        for item in request.items:
            try: jobs.append(svc().analyze(item))
            except QueueFull: return dict(jobs=jobs,remaining=len(request.items)-len(jobs),queue_full=True)
        return dict(jobs=jobs,remaining=0)

    @app.get('/v1/jobs/{job_id}',response_model=JobResponse,tags=['jobs'],summary='Status, result or error of an analysis/PDF job')
    def job(job_id: str):
        for queue in (svc().agents,svc().pdfs):
            try: return queue.get(job_id)
            except KeyError: pass
        raise HTTPException(404,'Unknown job')

    @app.get('/v1/jobs',tags=['jobs'])
    def jobs():
        return dict(analyses=[svc().agents.get(i) for i in list(svc().agents.items)],
                    reports=[svc().pdfs.get(i) for i in list(svc().pdfs.items)])

    @app.get('/v1/reports',tags=['reports'],summary='Atomically published report versions')
    def reports():
        with svc().catalog.lock: return dict(active=dict(svc().catalog.current))

    @app.post('/v1/reports/prepare',status_code=202,tags=['reports'],summary='Prepare existing reports in parallel with per-report caches and budgets')
    def prepare(request: PreparationRequest):
        options=PreparationOptions(**request.model_dump())
        return prepare_paths(sorted(PDF_DIR.glob('*.pdf')),options,PDF_DIR)

    @app.post('/v1/reports/upload',status_code=202,tags=['reports'],summary='Upload PDFs; prepare and publish in background without interrupting agents')
    async def upload(files: Annotated[list[UploadFile],File(description='One or several PDFs, maximum 50 MiB each')],
                     options: Annotated[str,Form(description='JSON PreparationRequest: charts, tables, digests and per-report budgets')]='{}'):
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
        from common.news_snapshot import news_snapshot
        from stage5_insight_detection.news_enrichment.related_news import RelatedNews
        with news_snapshot():
            result=RelatedNews().search(request.query,request.k)
            return jsonable_encoder(dict(items=result.to_dict('records'),diagnostics=result.attrs))

    @app.post('/v1/search/bank',tags=['search'])
    def bank_search(request: SearchRequest):
        version,library,bank=svc().catalog.snapshot(svc().models_factory())
        return dict(version=version,items=bank.search(request.query,request.k))

    @app.post('/v1/insights/ask', response_class=StreamingResponse, tags=['agents'],
              summary='Вопрос по готовому инсайту: ответ стримится токен за токеном')
    def ask_question(body: AskRequest):
        from backend.service import Service  # noqa: F401 - the request reaches the same service state
        def gen():
            try:
                yield from svc().ask(body.company_id, body.role, body.question, body.insight_title)
            except Exception as exc:
                yield f'\n[ошибка] {type(exc).__name__}: {exc}'
        return StreamingResponse(gen(), media_type='text/plain; charset=utf-8')

    @app.get('/v1/companies/{company_id}',tags=['companies'])
    def company(company_id: str):
        from stage2_company_facts.profile.integration import load_company
        try:
            result=load_company(company_id)
            return jsonable_encoder(dict(card=result.card,facts=result.facts))
        except (KeyError,ValueError,FileNotFoundError) as exc: raise HTTPException(404,'Company not found') from exc
    return app

app=create_app()
