import test from 'node:test';
import assert from 'node:assert/strict';
import { paymentAnalytics, companyCounterparties, counterpartyDetails, USER_ID } from '../src/api.js';

test('payment analytics reads the selected company, encodes filters and supports cancelling stale requests',async t=>{
  const controller=new AbortController();
  const result={company_id:'U_14366',summary:{},counterparties:[]};
  t.mock.method(globalThis,'fetch',async(url,init)=>{
    const parsed=new URL(url,'http://localhost');
    assert.equal(parsed.pathname,'/v1/companies/U_14366/payments');
    assert.equal(parsed.searchParams.get('direction'),'in');
    assert.equal(parsed.searchParams.get('q'),'Москва & ремонт');
    assert.equal(parsed.searchParams.get('offset'),'15');
    assert.equal(parsed.searchParams.get('sort'),'count');
    assert.equal(init.headers['X-User-ID'],USER_ID);
    assert.equal(init.signal,controller.signal);
    assert.ok(!init.method || init.method==='GET');
    return {ok:true,json:async()=>result};
  });
  assert.deepEqual(await paymentAnalytics('U_14366',{direction:'in',q:'Москва & ремонт',sort:'count',offset:15},controller.signal),result);
});

test('counterparty list and card stay scoped to the selected company and forward cancellation',async t=>{
  const controller=new AbortController();
  const requests=[];
  t.mock.method(globalThis,'fetch',async(url,init)=>{
    const parsed=new URL(url,'http://localhost');
    requests.push(parsed);
    assert.equal(init.signal,controller.signal);
    assert.equal(init.headers['X-User-ID'],USER_ID);
    return {ok:true,json:async()=>({ok:true})};
  });
  await companyCounterparties('U_14366',{q:'Москва & лизинг',direction:'both',group:'own',sort:'in',offset:15},controller.signal);
  await counterpartyDetails('U_14366','U_86261',controller.signal);
  assert.equal(requests[0].pathname,'/v1/companies/U_14366/counterparties');
  assert.equal(requests[0].searchParams.get('q'),'Москва & лизинг');
  assert.equal(requests[0].searchParams.get('direction'),'both');
  assert.equal(requests[0].searchParams.get('group'),'own');
  assert.equal(requests[0].searchParams.get('sort'),'in');
  assert.equal(requests[0].searchParams.get('offset'),'15');
  assert.equal(requests[1].pathname,'/v1/companies/U_14366/counterparties/U_86261');
});
