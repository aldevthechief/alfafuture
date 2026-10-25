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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from common.personalization import Personalization, ProfileAnswer, feedback_db
from common.roles import roles
from feedback_api.profile import latest_results
from feedback_api.analyzer import FeedbackAnalyzer, InvalidAnalysis
from feedback_api.importer import register_results
from feedback_api.schemas import FeedbackInput, FeedbackRecord, InsightRecord, Kind, Question, ReactionSummary, RegisterInsight, Scope
from feedback_api.store import Conflict, NotFound, Store
from feedback_api.company_chat import analyze_company, load_company_context, public_message
from feedback_api.company_store import CompanyChats
from feedback_api.schemas import CompanyChatContext, CompanyChatQuestion, CompanyConversation
from feedback_api.streaming import AnswerStreams

log = logging.getLogger(__name__)
USER_ID = re.compile(r'[\w.@-]{1,100}')
ORIGINS = os.environ.get('FEEDBACK_CORS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000').split(',')


class ImportResults(BaseModel):
    company_id: str = Field(min_length=1, max_length=100)
    registration_key: str | None = Field(default=None, min_length=1, max_length=160)
    results: list[dict] = Field(max_length=50, description='rows of data/insights/<company>_<role>.json')


class AnswerInput(BaseModel):
    statement: str = Field(min_length=4, max_length=300)
    verdict: str = Field(pattern='^(yes|no|other)$')
    text: str = Field('', max_length=500)


class CompanyChatStart(BaseModel):
    company_id: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)


def create_app(store: Store | None = None, analyzer=None, api_token: str | None = None, news_lookup=None, facts=None,
               results_loader=latest_results, company_context_loader=load_company_context,
               company_research_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store if store is not None else Store(feedback_db())
        app.state.store.initialize()
        app.state.company_chats = CompanyChats(app.state.store)
        app.state.company_chats.initialize()
        yield

    app = FastAPI(title='Alfa Insight Feedback API', version='1.0.0', lifespan=lifespan,
                  description='Reactions and questions on ready insights. User input never overwrites bank facts.')
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST', 'PUT', 'DELETE'],
                       allow_headers=['Content-Type', 'X-User-ID', 'Authorization'])
    processor = analyzer if analyzer is not None else FeedbackAnalyzer()
    streams = AnswerStreams()
    token = api_token if api_token is not None else os.environ.get('FEEDBACK_API_TOKEN', '')

    def user(x_user_id: Annotated[str, Header(max_length=100)], authorization: Annotated[str | None, Header()] = None) -> str:
        if token and not hmac.compare_digest((authorization or '').encode(), f'Bearer {token}'.encode()):
            raise HTTPException(401, 'Invalid API token')
        if not USER_ID.fullmatch(x_user_id.strip()):
            raise HTTPException(422, 'X-User-ID: 1-100 letters, digits, . _ @ -')
        return x_user_id.strip()

    User = Annotated[str, Depends(user)]

    def require_inference():
        if os.environ.get('CABINET_MODE', 'full') == 'server':
            raise HTTPException(403, 'This server only serves prepared demo accounts')

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
        require_inference()
        record = app.state.store.add(user_id, str(insight_id), body)
        return analyze(user_id, record)

    @app.post('/v1/insights/{insight_id}/ask', response_model=FeedbackRecord, status_code=201)
    def ask(insight_id: UUID, body: Question, user_id: User):
        """A question to the insight: the answer cites news:/fact:/bank: ids of the insight or explicit lookups."""
        require_inference()
        record = app.state.store.add(user_id, str(insight_id), FeedbackInput(
            kind=Kind.question, comment=body.question, selected_text=body.selected_text, selected_source=body.selected_source, request_id=body.request_id,
            scope=Scope.future if body.remember else Scope.insight))
        return analyze(user_id, record)

    @app.post('/v1/insights/{insight_id}/ask/stream', response_class=StreamingResponse)
    def ask_stream(insight_id: UUID, body: Question, user_id: User):
        """Real model text chunks, followed by a validated and persisted FeedbackRecord.

        The worker finishes saving even if the browser disconnects; a retry uses the same request_id.
        """
        require_inference()
        record = app.state.store.add(user_id, str(insight_id), FeedbackInput(
            kind=Kind.question, comment=body.question, selected_text=body.selected_text, selected_source=body.selected_source, request_id=body.request_id,
            scope=Scope.future if body.remember else Scope.insight))
        def work(on_response):
            current = app.state.store.feedback(user_id, record.id)
            if current.analysis is None:
                insight = app.state.store.insight(user_id, current.insight_id)
                history = app.state.store.recent(user_id, current)
                result = processor.analyze(insight, current, history, on_response=on_response)
                current = app.state.store.save_analysis(user_id, current.id, result)
            return current.model_dump(mode='json')

        return streams.response(record.id, work)

    def company_context(user_id, company_id, role):
        try:
            context = CompanyChatContext.model_validate(company_context_loader(app.state.store, user_id, company_id, role))
            if (context.evidence.company_id, context.evidence.role) != (company_id, role):
                raise ValueError('Company context does not match conversation')
            return context
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, 'Company context is unavailable') from exc

    @app.post('/v1/company-chats', response_model=CompanyConversation)
    def company_chat(body: CompanyChatStart, user_id: User):
        require_inference()
        profile_id(body.company_id)
        if body.role not in roles():
            raise HTTPException(422, 'Unknown role')
        # Opening reads local context but never invokes a model or registers a synthetic insight.
        company_context(user_id, body.company_id, body.role)
        return app.state.company_chats.open(user_id, body.company_id, body.role)

    @app.get('/v1/company-chats/{conversation_id}/messages')
    def company_chat_history(conversation_id: UUID, user_id: User, limit: int = Query(50, ge=1, le=100), before: UUID | None = None):
        items, more = app.state.company_chats.history(user_id, str(conversation_id), limit, str(before) if before else None)
        return dict(items=[public_message(item) for item in items], has_more=more,
                    before=items[0].id if more and items else None)

    @app.post('/v1/company-chats/{conversation_id}/ask/stream', response_class=StreamingResponse)
    def company_chat_ask(conversation_id: UUID, body: CompanyChatQuestion, user_id: User):
        require_inference()
        chats = app.state.company_chats
        conversation = chats.conversation(user_id, str(conversation_id))
        record = chats.add(user_id, conversation.id, body,
                           lambda: company_context(user_id, conversation.company_id, conversation.role))

        def work(on_response):
            current = chats.message(user_id, record.id)
            if current.analysis is None:
                result = analyze_company(processor, conversation, user_id, current, chats.recent(user_id, current),
                                         on_response=on_response, research_factory=company_research_factory)
                current = chats.save_analysis(user_id, current.id, result)
            return public_message(current)

        return streams.response('company:' + record.id, work)

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

    def profile_id(company_id, question_id=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', company_id):
            raise HTTPException(422, 'Invalid company_id')
        if question_id is not None and not re.fullmatch(r'q_[0-9a-f]{12}', question_id):
            raise HTTPException(422, 'question_id: q_ + 12 hex digits')

    @app.get('/v1/companies/{company_id}/questions')
    def questions(company_id: str, user_id: User, role: str = Query(min_length=1, max_length=100)):
        from stage2_company_facts.questions.questions import questions as make
        profile_id(company_id)
        if role not in roles():
            raise HTTPException(422, 'Unknown role')
        answered = {a.question_id: a for a in app.state.store.answers(user_id, company_id)}
        out = make(results_loader(company_id, role, user_id))
        ids = {q['id'] for q in out}
        # Saved answers remain editable even after a new analysis drops that assumption or the role changes.
        out += [dict(id=a.question_id, text=a.statement, question=f'Верно ли, что {a.statement}?',
                     **{'for': 'ранее вы уточняли это о компании'}, impact='Учитываем в следующих анализах.', insights=[])
                for a in answered.values() if a.question_id not in ids]
        for q in out:
            q['answer'] = answered[q['id']].model_dump() if q['id'] in answered else None
        return out

    @app.put('/v1/companies/{company_id}/answers/{question_id}', response_model=ProfileAnswer)
    def answer(company_id: str, question_id: str, body: AnswerInput, user_id: User):
        require_inference()
        profile_id(company_id, question_id)
        if not body.statement.strip() or (body.verdict == 'other' and not body.text.strip()):
            raise HTTPException(422, 'Statement and text for verdict=other must not be empty')
        return app.state.store.set_answer(user_id, company_id, ProfileAnswer(question_id=question_id, **body.model_dump()))

    @app.delete('/v1/companies/{company_id}/answers/{question_id}', status_code=204)
    def delete_answer(company_id: str, question_id: str, user_id: User):
        require_inference()
        profile_id(company_id, question_id)
        app.state.store.delete_answer(user_id, company_id, question_id)
        return Response(status_code=204)

    return app


app = create_app()
