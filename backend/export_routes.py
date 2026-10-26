"""Read-only exports and their own file job queue; no agent/PDF routes are changed."""
import hmac
import os
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from backend import exports
from backend.jobs import QueueFull
from backend.schemas import ExportBatch, ExportQuery, JobResponse


def create_export_router(get_service, staff_token=None):
    router=APIRouter(prefix='/v1/exports',tags=['exports'])
    token=os.environ.get('STAFF_API_TOKEN','') if staff_token is None else staff_token
    User=Annotated[str | None,Header(min_length=1,max_length=100,pattern=r'^[\w.@-]+$')]
    Authorization=Annotated[str | None,Header()]
    Format=Annotated[Literal['xlsx','csv','json','jsonl'],Query()]
    Decimal=Annotated[Literal['comma','dot'],Query()]

    def is_staff(authorization):
        # No token configured means no staff override of an access file.
        return bool(token) and hmac.compare_digest((authorization or '').encode(),f'Bearer {token}'.encode())

    def export_one(query,user,authorization,fmt,table,decimal):
        try: q=ExportQuery(**query)
        except ValidationError as exc:
            raise HTTPException(422,jsonable_encoder(exc.errors(include_context=False))) from exc
        try: tables,meta,stem=get_service().export_tables(q,user,is_staff(authorization))
        except KeyError: raise HTTPException(404,'Nothing to export')
        except FileNotFoundError as exc: raise HTTPException(503,'Case tables are not in data/') from exc
        try: data,media,ext=exports.render(tables,exports.meta_for(q.kind,**meta),fmt,table,decimal)
        except KeyError: raise HTTPException(422,'Unknown table; one of: '+', '.join(tables))
        return Response(data,media_type=media,headers={
            'Content-Disposition':f'attachment; filename="{stem}_{date.today().isoformat()}.{ext}"'})

    @router.get('/insights',summary='Saved insight feed of a company/role, or one owned/shared job')
    def insights(job_id: str | None=None,company_id: str | None=None,role: str | None=None,
                 origin: Literal['all','news','bank']='all',date_from: str | None=None,date_to: str | None=None,
                 format: Format='xlsx',table: Literal['insights','claims','sources','actions'] | None=None,
                 decimal: Decimal='comma',x_user_id: User=None,authorization: Authorization=None):
        return export_one(dict(kind='insights',job_id=job_id,company_id=company_id,role=role,origin=origin,
                               date_from=date_from,date_to=date_to),x_user_id,authorization,format,table,decimal)

    @router.get('/payments',summary='Payment aggregates and self-transfers for the selected company')
    def payments(company_id: str,scope: Literal['company','group']='company',side: Literal['both','in','out']='both',
                 format: Format='xlsx',decimal: Decimal='comma',x_user_id: User=None,authorization: Authorization=None):
        return export_one(dict(kind='payments',company_id=company_id,scope=scope,side=side),
                          x_user_id,authorization,format,None,decimal)

    @router.get('/counterparties',summary='Top suppliers/buyers, concentration and company group')
    def counterparties(company_id: str,scope: Literal['company','group']='company',top: int=Query(10,ge=1,le=200),
                       format: Format='xlsx',table: Literal['counterparties','concentration','group'] | None=None,
                       decimal: Decimal='comma',x_user_id: User=None,authorization: Authorization=None):
        return export_one(dict(kind='counterparties',company_id=company_id,scope=scope,top=top),
                          x_user_id,authorization,format,table,decimal)

    @router.post('',status_code=202,response_model=JobResponse,summary='Prepare several exports in a background job')
    def batch(request: ExportBatch,x_user_id: User=None,authorization: Authorization=None):
        try: return get_service().export_batch(request,x_user_id,is_staff(authorization))
        except QueueFull as exc: raise HTTPException(429,str(exc)) from exc

    def owned_job(export_id,user_id):
        try: job=get_service().exports.get(export_id)
        except KeyError: raise HTTPException(404,'Unknown export')
        if job['request'].get('user_id') not in (None,user_id): raise HTTPException(404,'Unknown export')
        return job

    @router.get('/{export_id}',response_model=JobResponse)
    def status(export_id: str,x_user_id: User=None): return owned_job(export_id,x_user_id)

    @router.get('/{export_id}/file')
    def file(export_id: str,x_user_id: User=None):
        job=owned_job(export_id,x_user_id)
        if job['status']!='succeeded': raise HTTPException(409,f'Export is {job["status"]}')
        result=job['result']
        return FileResponse(get_service().root/result['file'],media_type=result['media_type'],filename=result['name'])

    return router
