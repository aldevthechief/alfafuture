import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { cabinetFeed, jobPreviews, latestJobs, loadInsight, toFeed, USER_ID } from '../src/api.js';
import { view } from '../src/insightView.js';

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

test('every company and role reads the combined cached and inference feed without starting a run', async t => {
  const calls=[];
  t.mock.method(globalThis,'fetch',async (url,init) => {
    calls.push(url);
    assert.equal(init.headers['X-User-ID'],USER_ID);
    assert.equal(init.method,undefined);
    return {ok:true,json:async () => ({insights:[{id:'news',preview:true},{id:'bank',preview:true}]})};
  });
  for (const [company,role] of [['U_14366','cfo'],['U_177581','ceo'],['U_57706','treasury'],['U_999','marketing']]) {
    assert.equal((await cabinetFeed(company,role)).insights.length,2);
    assert.ok(calls.at(-1).endsWith(`/v1/cabinet/feed?company_id=${company}&role=${role}`));
  }
  assert.equal(calls.length,4);
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
  assert.ok(calls[0].endsWith('/v1/jobs/job-detail/insights/7?insight_id=row-7-bank'));
});

test('saved news cards retain factual claims and their original article URLs', async () => {
  const job = JSON.parse(await readFile(new URL('../../backend/demo_cache/U_14366_ceo.json', import.meta.url), 'utf8'));
  const feed = toFeed(job);
  assert.equal(feed.insights.length, 3);
  assert.deepEqual(feed.insights.map(card => card.pipeline_result.insight.origin), ['news', 'news', 'bank']);
  for (const card of feed.insights) {
    assert.equal(card.role, 'ceo');
    if (card.pipeline_result.insight.origin === 'bank') {
      assert.ok(card.bank_view.length > 0);
      assert.ok(card.pipeline_result.evidence.bank.length > 0);
      continue;
    }
    assert.ok(view(card).facts.length > 0);
    const original = card.pipeline_result.evidence.news;
    const cited = view(card).facts.flatMap(claim => claim.sources).filter(id => id.startsWith('news:'));
    assert.ok(cited.length > 0);
    for (const id of cited) {
      const news = original.find(n => `news:${n.id}` === id);
      assert.ok(news);
      assert.equal(card.sources[id].url, news.url);
      assert.ok(/^https?:\/\//.test(card.sources[id].url));
      assert.equal(card.sources[id].title, news.title);
    }
  }
});
