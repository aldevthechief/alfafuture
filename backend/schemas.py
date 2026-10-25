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
    candidates: int=Field(30,ge=1,le=500)
    judge: int=Field(5,ge=1,le=100)
    workers: int=Field(16,ge=1,le=16)
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


class JobResponse(BaseModel):
    id: str
    kind: str
    request: dict
    status: Literal['queued','running','succeeded','partial','failed','interrupted']
    created_at: str
    finished_at: str | None=None
    result: dict | list | None=None
    error: str | None=None

class AskRequest(BaseModel):
    """A reader's question about a finished insight; the answer streams over the saved result rows."""
    company_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    role: str = 'cfo'
    question: str = Field(min_length=1, max_length=2000)
    insight_title: str | None = None
