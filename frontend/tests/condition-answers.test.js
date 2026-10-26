import test from 'node:test';
import assert from 'node:assert/strict';
import { USER_ID } from '../src/api.js';
import { conditionId, conditionQuestion, createConditionAnswers } from '../src/conditionAnswers.js';

const statement = 'Часть долга компании обслуживается по плавающей ставке';
const id = conditionId(`insight-1|${statement}`);
const response = (body, status = 200) => ({ ok:status < 400, status, json:async () => body });

test('the same condition reuses the profile question and its saved answer instead of creating a duplicate', () => {
  const question = { id:'q_0123456789ab', text:statement, answer:{ verdict:'yes' } };
  assert.equal(conditionQuestion('insight-1', ` ${statement.toLowerCase()}. `, [question]), question);
  assert.deepEqual(conditionQuestion('insight-1', statement, []), { id, text:statement });
});

test('a failed PUT never confirms an answer; retry confirms only after acknowledgement and duplicate clicks are ignored', async t => {
  const calls = [];
  let reply, saved = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push([url, init]);
    assert.equal(init.headers['X-User-ID'], USER_ID);
    if (!init.method) return response([]);
    return new Promise(resolve => { reply = resolve; });
  });
  const form = createConditionAnswers('U_1', 'cfo', () => saved++);
  await form.load();
  const pending = form.pick(id, statement, 'yes');
  assert.equal(form.getSnapshot().items[id].busy, true);
  assert.equal(form.getSnapshot().items[id].choice, undefined);
  await form.pick(id, statement, 'no');
  assert.equal(calls.length, 2);
  reply(response({ detail:'Сервис временно недоступен' }, 503));
  assert.equal(await pending, false);
  assert.equal(saved, 0);
  assert.equal(form.getSnapshot().items[id].choice, undefined);
  assert.equal(form.getSnapshot().items[id].pending, 'yes');
  assert.match(form.getSnapshot().items[id].error, /временно недоступен/);
  const retry = form.pick(id, statement, form.getSnapshot().items[id].pending);
  assert.equal(form.getSnapshot().items[id].choice, undefined);
  reply(response({ question_id:id, statement, verdict:'yes', text:'' }));
  assert.equal(await retry, true);
  assert.equal(saved, 1);
  assert.equal(form.getSnapshot().items[id].choice, 'yes');
  assert.equal(form.getSnapshot().items[id].error, '');
  assert.equal(calls[1][1].method, 'PUT');
  assert.match(calls[1][0], new RegExp(`/U_1/answers/${id}$`));
  assert.deepEqual(JSON.parse(calls[1][1].body), { statement, verdict:'yes', text:'' });
});

test('reopening reads the server answer; a failed change preserves the previously confirmed choice', async t => {
  t.mock.method(globalThis, 'fetch', async (url, init) => init.method
    ? response({ detail:'Не сохранено' }, 500)
    : response([{ id, answer:{ verdict:'yes', text:'' } }]));
  const form = createConditionAnswers('U_1', 'cfo');
  await form.load();
  assert.equal(form.getSnapshot().items[id].choice, 'yes');
  assert.equal(await form.pick(id, statement, 'no'), false);
  assert.equal(form.getSnapshot().items[id].choice, 'yes');
  assert.equal(form.getSnapshot().items[id].pending, 'no');
});

test('unavailable saved answers offer a load retry and cannot be overwritten before loading', async t => {
  let attempts = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.equal(init.method, undefined);
    assert.match(url, /\/U_2\/questions\?role=ceo$/);
    return ++attempts === 1 ? response({ detail:'Недоступно' }, 503) : response([]);
  });
  const form = createConditionAnswers('U_2', 'ceo');
  await form.load();
  assert.match(form.getSnapshot().error, /Недоступно/);
  await form.pick(id, statement, 'yes');
  assert.equal(attempts, 1);
  await form.load();
  assert.equal(form.getSnapshot().error, '');
  assert.deepEqual(form.getSnapshot().items, {});
  assert.match(id, /^q_[0-9a-f]{12}$/);
});
