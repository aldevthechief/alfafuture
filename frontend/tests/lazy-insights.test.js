import test from 'node:test';
import assert from 'node:assert/strict';
import { jobPreviews, latestJobs, loadInsight, USER_ID } from '../src/api.js';

test('finding the latest analysis requests metadata without full results', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push(url);
    assert.equal(init.headers['X-User-ID'], USER_ID);
    return { ok: true, json: async () => ({ analyses: [{ id:'job-lazy', status:'succeeded', created_at:'2026-10-25',
      request:{company_id:'U_1',role:'cfo',user_id:USER_ID} }] }) };
  });
  assert.equal((await latestJobs('U_1','cfo')).done.id, 'job-lazy');
  assert.equal(calls.length, 1);
  assert.ok(calls[0].endsWith('/v1/jobs?include_results=false'));
});

test('feed previews do not trigger any insight detail requests', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url);
    return {ok:true,json:async()=>({insights:[{id:'preview',preview:true}]})};
  });
  assert.equal((await jobPreviews('job-previews')).insights[0].preview, true);
  assert.equal(calls.length, 1);
  assert.ok(calls[0].endsWith('/v1/jobs/job-previews/insights'));
});

test('opening a card fetches only that result once and keeps its evidence and identity', async t => {
  const card = {id:'row-7-bank',job_id:'job-detail',result_index:7,preview:true,roleLabel:'Финансовый директор'};
  const row = {status:'ready',insight:{title:'Test',role:'cfo'},
    evidence:{role:'cfo',facts:[{id:'fact:1',text:'Frozen original fact'}]}};
  const calls = [];
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url);
    return {ok:true,json:async()=>({id:card.job_id,request:{role:'cfo'},result:{results:[row]}})};
  });
  const [first, second] = await Promise.all([loadInsight(card,'U_1'),loadInsight(card,'U_1')]);
  assert.equal(first, second);
  assert.equal(first.id, card.id);
  assert.equal(first.pipeline_result, row);
  assert.equal(first.sources['fact:1'].text, 'Frozen original fact');
  assert.deepEqual(calls.length, 1);
  assert.ok(calls[0].endsWith('/v1/jobs/job-detail/insights/7'));
});
