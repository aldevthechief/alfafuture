import test from 'node:test';
import assert from 'node:assert/strict';
import { startAnalysis, toFeed, USER_ID } from '../src/api.js';
import { askQuestion, discussionHistory, registerInsight, sendReaction } from '../src/feedback.js';

const row = (id, text = 'Original company fact') => ({ status:'ready', news_id:id,
  insight:{ id:`insight-${id}`, role:'cfo', title:'Test', what_happened:[], why_it_matters:[], risks:[], opportunities:[],
    hypotheses:[], bank_view:[] },
  evidence:{ company_id:'U_1', role:'cfo', facts:[{ id:'fact:1', text }], news:[], bank:[] },
});
const response = body => ({ ok:true, json:async () => body });

test('feed keeps the raw result, role ID and the original facts per card', () => {
  const first = row(1), second = row(2, 'A different original fact');
  const feed = toFeed({ id:'job-1', result:{ results:[first, second] } }, [{ text:'Updated company fact' }]);
  assert.equal(feed.insights[0].role, 'cfo');
  assert.equal(feed.insights[0].pipeline_result, first);
  assert.equal(feed.insights[0].sources['fact:1'].text, 'Original company fact');
  assert.equal(feed.insights[1].sources['fact:1'].text, 'A different original fact');
  assert.equal(feed.insights[0].job_id, 'job-1');
});

test('analysis uses the same user identity as discussion and polling', async t => {
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.equal(init.headers['X-User-ID'], USER_ID);
    assert.equal(JSON.parse(init.body).user_id, USER_ID);
    assert.equal(JSON.parse(init.body).role, 'cfo');
    return response({ id:'job-2' });
  });
  await startAnalysis('U_1', 'cfo');
});

test('legacy generated role labels do not replace the role identity from frozen evidence', () => {
  const original = row(40);
  original.insight.role = 'Финансовый директор (CFO)';
  const card = toFeed({ id:'job-40', request:{ role:'cfo' }, result:{ results:[original] } }).insights[0];
  assert.equal(card.role, 'cfo');
  assert.equal(card.pipeline_result, original);
  assert.equal(original.insight.role, 'Финансовый директор (CFO)');
});

test('registration sends the raw v2 result once, then history, reactions and retries use its UUID', async t => {
  const raw = row(3);
  const insight = toFeed({ id:'job-3', result:{ results:[raw] } }).insights[0];
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push(url);
    assert.equal(init.headers['X-User-ID'], USER_ID);
    if (url.endsWith('/import')) {
      const body = JSON.parse(init.body);
      assert.deepEqual(body.results, [raw]);
      assert.equal(body.results[0].insight.role, 'cfo');
      assert.equal(body.registration_key, 'job-3:insight-3');
      return response([{ id:'uuid-3', evidence:raw.evidence }]);
    }
    assert.match(url, /\/insights\/uuid-3\//);
    if (url.endsWith('/ask')) assert.equal(JSON.parse(init.body).request_id, 'question-retry-id');
    return response(url.includes('?limit=') ? [] : { id:'message-3' });
  });
  await Promise.all([registerInsight('U_1', insight), registerInsight('U_1', insight)]);
  await discussionHistory('U_1', insight);
  await sendReaction('U_1', insight, { kind:'already_known' }, 'reaction-id');
  await askQuestion('U_1', insight, 'Question?', 'question-retry-id');
  await askQuestion('U_1', insight, 'Question?', 'question-retry-id');
  assert.equal(calls.filter(url => url.endsWith('/import')).length, 1);
});

test('demo cards never become real feedback records', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw Error('unexpected fetch'); });
  await assert.rejects(registerInsight('U_1', { id:'demo', role:'cfo' }), /после запуска анализа/);
  assert.equal(fetch.mock.callCount(), 0);
});
