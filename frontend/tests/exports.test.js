import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchExport, USER_ID } from '../src/api.js';

test('export passes the selected company and role, user identity and cancellation signal and keeps the server filename',async t=>{
  const controller=new AbortController();
  t.mock.method(globalThis,'fetch',async(url,init)=>{
    const request=new URL(url,'http://localhost');
    assert.equal(request.pathname,'/v1/exports/insights');
    assert.equal(request.searchParams.get('company_id'),'U_14366');
    assert.equal(request.searchParams.get('role'),'ceo');
    assert.equal(request.searchParams.get('format'),'csv');
    assert.equal(init.headers['X-User-ID'],USER_ID);
    assert.equal(init.signal,controller.signal);
    return new Response(new Blob(['csv archive']),{headers:{'Content-Disposition':'attachment; filename="alfa-insights_U_14366_ceo.zip"'}});
  });
  const result=await fetchExport('insights',{company_id:'U_14366',role:'ceo',format:'csv'},controller.signal);
  assert.equal(result.name,'alfa-insights_U_14366_ceo.zip');
  assert.equal(await result.blob.text(),'csv archive');
});

test('HTTP errors are not downloaded as files and a cancelled old company request cannot reach download',async t=>{
  t.mock.method(globalThis,'fetch',async()=>new Response(JSON.stringify({detail:'Нет доступа'}),{status:404}));
  await assert.rejects(fetchExport('payments',{company_id:'U_1',format:'xlsx'}),/Нет данных для выгрузки/);
  const controller=new AbortController();
  t.mock.method(globalThis,'fetch',async()=>{
    controller.abort();
    return new Response('old company data');
  });
  await assert.rejects(fetchExport('payments',{company_id:'U_1',format:'json'},controller.signal),{name:'AbortError'});
});

test('payment and counterparty exports remain company scoped with explicit formats and limits',async t=>{
  const seen=[];
  t.mock.method(globalThis,'fetch',async url=>{seen.push(new URL(url,'http://localhost'));return new Response('{}');});
  await fetchExport('payments',{company_id:'U_2',scope:'company',side:'both',format:'json'});
  await fetchExport('counterparties',{company_id:'U_3',scope:'company',top:200,format:'xlsx'});
  assert.equal(seen[0].searchParams.get('company_id'),'U_2');
  assert.equal(seen[0].searchParams.get('side'),'both');
  assert.equal(seen[1].searchParams.get('company_id'),'U_3');
  assert.equal(seen[1].searchParams.get('scope'),'company');
  assert.equal(seen[1].searchParams.get('top'),'200');
});
