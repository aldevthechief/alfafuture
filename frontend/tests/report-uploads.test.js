import test from 'node:test';
import assert from 'node:assert/strict';
import { File } from 'node:buffer';
import { ApiError, reportJob, reportJobs, uploadReport, USER_ID } from '../src/api.js';
import { mergeReportJobs, reportStatus, submitReports } from '../src/reportUploads.js';

test('PDF upload sends multipart data and user identity, with a browser-generated boundary',async t=>{
  const file=new File(['%PDF-1.4'],'Обзор.pdf',{type:'application/pdf'});
  t.mock.method(globalThis,'fetch',async (url,init)=>{
    assert.ok(url.endsWith('/v1/reports/upload'));
    assert.equal(init.method,'POST');
    assert.equal(init.headers['X-User-ID'],USER_ID);
    assert.equal(init.headers['Content-Type'],undefined);
    assert.ok(init.body instanceof FormData);
    assert.equal(init.body.get('files').name,file.name);
    assert.equal(await init.body.get('files').text(),'%PDF-1.4');
    assert.deepEqual(JSON.parse(init.body.get('options')),{chart_mode:'selective'});
    return {ok:true,json:async()=>({jobs:[{id:'pdf-job',status:'queued'}]})};
  });
  assert.equal((await uploadReport(file)).id,'pdf-job');
});

test('a full queue or an invalid PDF is never presented as an accepted report',async t=>{
  const file=new File(['bad'],'report.pdf');
  t.mock.method(globalThis,'fetch',async()=>({ok:true,json:async()=>({jobs:[],remaining_files:['report.pdf'],queue_full:true})}));
  await assert.rejects(uploadReport(file),error=>error.status===429);
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:422}));
  await assert.rejects(uploadReport(file),/не защищён паролем/);
});

test('a proxy 413 for a 7.1 MiB PDF does not claim the file exceeds 50 MiB',async t=>{
  const file=new File([new Uint8Array(Math.round(7.1*1024*1024))],'Бонд_Бриф_14092026.pdf',{type:'application/pdf'});
  t.mock.method(globalThis,'fetch',async()=>({ok:false,status:413}));
  await assert.rejects(uploadReport(file),error=>error.status===413 && /Сам файл укладывается/.test(error.message));
});

test('a file larger than 50 MiB is rejected before sending a request',async t=>{
  const fetch=t.mock.method(globalThis,'fetch',async()=>assert.fail('oversized file must stay local'));
  await assert.rejects(uploadReport({name:'oversized.pdf',size:50*1024*1024+1}),/превышает допустимый размер/);
  assert.equal(fetch.mock.callCount(),0);
});

test('only failed files remain for retry after partial submission; accepted files are not submitted twice',async()=>{
  const files=['first.pdf','second.pdf','third.pdf'].map(name=>({name}));
  const calls=[],accepted=[],failed=[];
  await submitReports(files,{onStart:()=>{},upload:async file=>{
    calls.push(file.name);if(file===files[1]) throw new ApiError(429,'Очередь заполнена');return {id:file.name};
  },onAccepted:file=>accepted.push(file),onError:file=>failed.push(file)});
  assert.deepEqual(accepted,[files[0],files[2]]);
  assert.deepEqual(failed,[files[1]]);
  await submitReports(failed,{onStart:()=>{},upload:async file=>{calls.push(file.name);return {id:file.name};},onAccepted:()=>{},onError:()=>assert.fail()});
  assert.deepEqual(calls,['first.pdf','second.pdf','third.pdf','second.pdf']);
});

test('saved report history loads without analysis payloads and status checks can be cancelled',async t=>{
  const controller=new AbortController();
  t.mock.method(globalThis,'fetch',async (url,init)=>{
    assert.equal(init.headers['X-User-ID'],USER_ID);
    assert.equal(init.signal,controller.signal);
    assert.ok(url.endsWith('/v1/jobs?include_results=false') || url.endsWith('/v1/jobs/report%2Fid'));
    return {ok:true,json:async()=>url.includes('?') ? {reports:[{id:'report/id'}]} : {id:'report/id',status:'succeeded'}};
  });
  assert.deepEqual(await reportJobs(controller.signal),[{id:'report/id'}]);
  assert.equal((await reportJob('report/id',controller.signal)).status,'succeeded');
});

test('status updates retain final publication details and distinguish failures, interruption and superseded versions',()=>{
  const completed={id:'done',status:'succeeded',result:{published:false},created_at:'2026-10-25'};
  const merged=mergeReportJobs([completed],[{id:'done',status:'succeeded',result:null},{id:'pending',status:'queued',created_at:'2026-10-24'}]);
  assert.equal(merged[0].id,'pending');
  assert.equal(reportStatus(merged[1]).label,'Есть более новая версия');
  assert.equal(reportStatus({status:'failed'}).tone,'error');
  assert.match(reportStatus({status:'interrupted'}).hint,/перезапущен/);
  assert.equal(reportStatus({status:'running'}).tone,'pending');
});

test('missing OCR on the server does not blame the uploaded PDF or expose setup commands',()=>{
  const status=reportStatus({status:'failed',error:'RuntimeError: Local OCR needs rus/eng language data: run python -m common.ocr_local --download or set TESSDATA_PREFIX'});
  assert.equal(status.label,'На сервере не настроено распознавание');
  assert.match(status.hint,/сохранён на сервере/);
  assert.doesNotMatch(status.hint,/Проверьте файл|TESSDATA_PREFIX|--download/);
});
