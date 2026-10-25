import test from 'node:test';
import assert from 'node:assert/strict';
import { USER_ID, companyProfile } from '../src/api.js';
import { companyQuestions, saveAnswer } from '../src/feedback.js';

const question = { id:'q_0123456789ab', text:'Часть долга компании плавающая' };
const response = (body, status = 200) => ({ ok:status < 400, status, json:async () => body });

test('questions are asked for the company and role of the analysis', async t => {
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.match(url, /\/v1\/companies\/U_1\/questions\?role=cfo$/);
    assert.equal(init.headers['X-User-ID'], USER_ID);
    return response([{ ...question, question:'Верно ли, что …?', answer:null }]);
  });
  assert.equal((await companyQuestions('U_1', 'cfo'))[0].id, question.id);
});

test('an answer is saved with its statement, a skip stays local, a withdrawn answer is deleted', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => { calls.push([init.method, url, init.body]); return response({}, init.method === 'DELETE' ? 204 : 200); });
  await saveAnswer('U_1', question, { verdict:'other', text:'около 35%' });
  await saveAnswer('U_1', question, { verdict:'skip' });
  await saveAnswer('U_1', question, undefined);
  assert.equal(calls.length, 2);
  assert.equal(calls[0][0], 'PUT');
  assert.match(calls[0][1], /\/v1\/companies\/U_1\/answers\/q_0123456789ab$/);
  assert.deepEqual(JSON.parse(calls[0][2]), { statement:question.text, verdict:'other', text:'около 35%' });
  assert.equal(calls[1][0], 'DELETE');
});

test('profile reads use the backend projection and API errors are propagated instead of reporting a saved answer', async t => {
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.equal(init.headers['X-User-ID'], USER_ID);
    if (url.endsWith('/v1/companies/U_1')) return response({card:{},facts:[],profile:{industry:'Лизинг'}});
    return response({detail:'Service unavailable'},503);
  });
  assert.equal((await companyProfile('U_1')).profile.industry,'Лизинг');
  await assert.rejects(saveAnswer('U_1',question,{verdict:'yes'}), /Service unavailable/);
});
