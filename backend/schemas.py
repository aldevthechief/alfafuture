"""Validated public equivalents of all pipeline CLI switches."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from common.roles import roles

class AnalysisRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    company_id: str=Field(min_length=1,max_length=100,pattern=r'^[A-Za-z0-9_-]+$')
    user_id: str | None=Field(None, min_length=1, max_length=100, pattern=r'^[\w.@-]+$',
                              description='Development user identity, same as feedback_api X-User-ID; not authentication')
    role: str='cfo'
    candidates: int=Field(15,ge=1,le=500)  # quota per candidate generator, see stage3_news_search/retrieval.py
    judge: int=Field(15,ge=1,le=100)
    workers: int=Field(16,ge=1,le=32)  # LLM calls in flight: max_concurrency of the graph; 16 = all 15 judged news at once
    engine: Literal['judge']='judge'
    retrieval: Literal['v2','v3','hybrid','lexical']='v2'
    source: Literal['news','analytics','both']='both'
    analytics_cache_only: bool=False
    analytics_refresh: bool=False
    analytics_share: float=Field(.4,gt=0,le=1)
    ocr: bool=False
    bank_charts: bool=False
    bank_tables: bool=False
    bank_digest: bool=False
    bank_prepare: bool=False
    chart_mode: Literal['all','selective','off'] | None=Field(None, description='Chart selection during PDF preparation; default BANK_CHARTS or all')
    ocr_budget: float | None=Field(None,ge=0)
    charts_budget: float | None=Field(None,ge=0)
    tables_budget: float | None=Field(None,ge=0)
    digest_budget: float | None=Field(None,ge=0)

    @model_validator(mode='after')
    def validate_flags(self):
        if self.role not in roles(): raise ValueError('Unknown role')
        if self.judge>self.candidates: raise ValueError('judge must not exceed candidates')
        prepare=self.bank_prepare or self.bank_charts or self.bank_tables or self.bank_digest
        if self.chart_mode is not None and not (prepare or self.ocr): raise ValueError('chart_mode requires bank preparation or ocr')
        for value,enabled in ((self.ocr_budget,self.ocr or prepare),(self.charts_budget,self.bank_charts or self.bank_prepare),
                              (self.tables_budget,self.bank_tables or self.bank_prepare),(self.digest_budget,self.bank_digest or self.bank_prepare)):
            if value is not None and not enabled: raise ValueError('Stage budget requires corresponding preparation flag')
        if self.analytics_refresh and self.analytics_cache_only: raise ValueError('analytics_refresh and analytics_cache_only conflict')
        return self

class BatchRequest(BaseModel):
    items: list[AnalysisRequest]=Field(min_length=1,max_length=32)

class PreparationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    ocr: bool=True
    charts: bool=True
    tables: bool=True
    digests: bool=True
    chart_mode: Literal['all','selective','off'] | None=Field(None, description='all: every chart; selective: new information only; off: no charts; default BANK_CHARTS or all')
    ocr_budget: float=Field(15,ge=0)
    charts_budget: float=Field(8,ge=0)
    tables_budget: float=Field(7,ge=0)
    digest_budget: float=Field(3,ge=0)

class SearchRequest(BaseModel):
    query: str=Field(min_length=1,max_length=5000)
    k: int=Field(8,ge=1,le=40)


class ExportQuery(BaseModel):
    """Saved cabinet data only: exporting never starts an analysis or a model."""
    model_config=ConfigDict(extra='forbid')
    kind: Literal['insights','payments','counterparties']
    job_id: str | None=Field(None, pattern=r'^(?:[0-9a-f]{32}|demo-[A-Za-z0-9_-]{1,120})$')
    company_id: str | None=Field(None, min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    role: str | None=None
    scope: Literal['company','group']='company'
    side: Literal['both','in','out']='both'
    top: int=Field(10, ge=1, le=200)
    origin: Literal['all','news','bank']='all'
    date_from: str | None=None
    date_to: str | None=None

    @model_validator(mode='after')
    def target(self):
        from datetime import date
        if self.kind=='insights' and not (self.job_id or self.company_id):
            raise ValueError('job_id or company_id is required')
        if self.kind!='insights' and not self.company_id:
            raise ValueError('company_id is required')
        if self.role is not None and self.role not in roles(): raise ValueError('Unknown role')
        for value in (self.date_from, self.date_to):
            if value and date.fromisoformat(value).isoformat()!=value: raise ValueError('Use YYYY-MM-DD dates')
        if self.date_from and self.date_to and self.date_from>self.date_to:
            raise ValueError('date_from must not be after date_to')
        return self


class ExportBatch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    items: list[ExportQuery]=Field(min_length=1, max_length=50)
    format: Literal['xlsx','json','csv']='xlsx'
    decimal: Literal['comma','dot']='comma'


class JobStage(BaseModel):
    code: Literal['queued','preparing_reports','collecting','analyzing','generating','verifying','finalizing',
                  'done','failed','interrupted']
    label: str
    step: int
    steps: int
    done: int | None=None
    total: int | None=None
    branches: dict=Field(default_factory=dict)
    started_at: str
    updated_at: str
    seq: int
    history: list[dict]=Field(default_factory=list)


class JobResponse(BaseModel):
    id: str
    kind: str
    request: dict
    status: Literal['queued','running','succeeded','partial','failed','interrupted']
    created_at: str
    finished_at: str | None=None
    result: dict | list | None=None
    error: str | None=None
    stage: JobStage | None=None
    revision: int=0
    partial_results: list[dict] | None=None

class AskRequest(BaseModel):
    """A reader's question about a finished insight; the answer streams over the saved result rows."""
    company_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    role: str = 'cfo'
    question: str = Field(min_length=1, max_length=2000)
    insight_title: str | None = None
