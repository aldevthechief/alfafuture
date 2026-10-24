"""HTTP API of the insight discussion: reactions, questions to an insight, history, and the personalization the next
pipeline run will use. Based on the maxon_dialogue branch (aldev), on insight contract v2.

python -m uvicorn feedback_api.app:app --host 127.0.0.1 --port 8001   (Swagger: /docs)

Identity: X-User-ID from a trusted backend (a development identity, not bank authentication). With FEEDBACK_API_TOKEN set,
every /v1 call also needs `Authorization: Bearer <token>`; the token must never reach a browser.
"""
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.personalization import Personalization, feedback_db
from feedback_api.analyzer import FeedbackAnalyzer, InvalidAnalysis
from feedback_api.importer import register_results
from feedback_api.schemas import FeedbackInput, FeedbackRecord, InsightRecord, Kind, Question, ReactionSummary, RegisterInsight, Scope
from feedback_api.store import Conflict, NotFound, Store

log = logging.getLogger(__name__)
USER_ID = re.compile(r'[\w.@-]{1,100}')
ORIGINS = os.environ.get('FEEDBACK_CORS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000').split(',')


class ImportResults(BaseModel):
    company_id: str = Field(min_length=1, max_length=100)
    registration_key: str | None = Field(default=None, min_length=1, max_length=160)
    results: list[dict] = Field(max_length=50, description='rows of data/insights/<company>_<role>.json')


def create_app(store: Store | None = None, analyzer=None, api_token: str | None = None, news_lookup=None, facts=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store if store is not None else Store(feedback_db())
        app.state.store.initialize()
        yield

    app = FastAPI(title='Alfa Insight Feedback API', version='1.0.0', lifespan=lifespan,
                  description='Reactions and questions on ready insights. User input never overwrites bank facts.')
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST', 'DELETE'],
                       allow_headers=['Content-Type', 'X-User-ID', 'Authorization'])
    processor = analyzer if analyzer is not None else FeedbackAnalyzer()
    token = api_token if api_token is not None else os.environ.get('FEEDBACK_API_TOKEN', '')

    def user(x_user_id: Annotated[str, Header(max_length=100)], authorization: Annotated[str | None, Header()] = None) -> str:
        if token and not hmac.compare_digest((authorization or '').encode(), f'Bearer {token}'.encode()):
            raise HTTPException(401, 'Invalid API token')
        if not USER_ID.fullmatch(x_user_id.strip()):
            raise HTTPException(422, 'X-User-ID: 1-100 letters, digits, . _ @ -')
        return x_user_id.strip()

    User = Annotated[str, Depends(user)]

    @app.exception_handler(NotFound)
    async def not_found(request, exc):
        return JSONResponse(status_code=404, content={'detail': 'Not found'})

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={'detail': str(exc)})

    def analyze(user_id: str, record: FeedbackRecord) -> FeedbackRecord:
        if record.analysis is not None:
            return record
        insight = app.state.store.insight(user_id, record.insight_id)
        history = app.state.store.recent(user_id, record)
        try:
            result = processor.analyze(insight, record, history)
        except RuntimeError as exc:
            if str(exc).startswith('set LLM_API_KEY'):
                raise HTTPException(503, 'LLM_API_KEY is not configured; the message is saved') from exc
            log.warning('analysis failed: %s', type(exc).__name__)  # never log prompts, comments or upstream bodies
            raise HTTPException(502, 'Answer unavailable; the message is saved, retry later') from exc
        except Exception as exc:
            log.warning('analysis failed: %s', type(exc).__name__)
            detail = 'Answer cited unknown sources' if isinstance(exc, InvalidAnalysis) else 'Answer unavailable'
            raise HTTPException(502, detail + '; the message is saved, retry later') from exc
        return app.state.store.save_analysis(user_id, record.id, result)

    @app.get('/health')
    def health():
        with app.state.store.connection() as db:
            db.execute('SELECT 1')
        return {'status': 'ok', 'authentication': 'backend_token' if token else 'local_development'}

    @app.post('/v1/insights', response_model=InsightRecord, status_code=201)
    def register(body: RegisterInsight, user_id: User):
        return app.state.store.register(user_id, body)

    @app.post('/v1/insights/import', response_model=list[InsightRecord], status_code=201)
    def import_results(body: ImportResults, user_id: User):
        """Register the ready insights of a pipeline result; evidence snapshots are rebuilt by code."""
        return register_results(app.state.store, user_id, body.company_id, body.results, news_lookup=news_lookup,
                                facts=facts, output=log.info, registration_key=body.registration_key)

    @app.get('/v1/companies/{company_id}/insights', response_model=list[InsightRecord])
    def insights(company_id: str, user_id: User, role: str = Query(min_length=1, max_length=100),
                 limit: int = Query(50, ge=1, le=100)):
        return app.state.store.insights(user_id, company_id, role, limit)

    @app.get('/v1/insights/{insight_id}', response_model=InsightRecord)
    def insight(insight_id: UUID, user_id: User):
        return app.state.store.insight(user_id, str(insight_id))

    @app.post('/v1/insights/{insight_id}/feedback', response_model=FeedbackRecord, status_code=201)
    def add_feedback(insight_id: UUID, body: FeedbackInput, user_id: User):
        """A reaction (like / dislike / emoji / useful / not_relevant / already_known) is answered by rules, without a model;
        a comment is answered by the model grounded in the insight's evidence."""
        record = app.state.store.add(user_id, str(insight_id), body)
        return analyze(user_id, record)

    @app.post('/v1/insights/{insight_id}/ask', response_model=FeedbackRecord, status_code=201)
    def ask(insight_id: UUID, body: Question, user_id: User):
        """A question to the insight: the answer cites news:/fact:/bank: ids of the insight or explicit lookups."""
        record = app.state.store.add(user_id, str(insight_id), FeedbackInput(
            kind=Kind.question, comment=body.question, request_id=body.request_id,
            scope=Scope.future if body.remember else Scope.insight))
        return analyze(user_id, record)

    @app.get('/v1/insights/{insight_id}/feedback', response_model=list[FeedbackRecord])
    def history(insight_id: UUID, user_id: User, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        return app.state.store.history(user_id, str(insight_id), limit, offset)

    @app.get('/v1/insights/{insight_id}/reactions', response_model=ReactionSummary)
    def reactions(insight_id: UUID, user_id: User):
        return app.state.store.reactions(user_id, str(insight_id))

    @app.delete('/v1/feedback/{feedback_id}', status_code=204)
    def delete(feedback_id: UUID, user_id: User):
        app.state.store.delete(user_id, str(feedback_id))
        return Response(status_code=204)

    @app.get('/v1/companies/{company_id}/personalization', response_model=Personalization)
    def personalization(company_id: str, user_id: User, role: str = Query(min_length=1, max_length=100)):
        """What `pipeline.py --user-id` will apply on the next selection for this (user, company, role)."""
        return app.state.store.personalization(user_id, company_id, role)

    return app


app = create_app()
