import test from 'node:test';
import assert from 'node:assert/strict';
import { compactOf, toFeed, USER_ID } from '../src/api.js';
import { view } from '../src/insightView.js';
import { askQuestion } from '../src/feedback.js';
import { accountSession, initialSession, sessionAccount } from '../src/accountSession.js';

test('fresh entry chooses the first account and saved accounts receive current activity-based names', () => {
  const accounts = [
    {id:'first', company_id:'U_1', sector:'Лизинг', role:'ceo', job_id:'demo-U_1-ceo'},
    {id:'second', company_id:'U_2', sector:'Строительство', role:'cfo', job_id:'demo-U_2-cfo'},
  ];
  assert.equal(initialSession(null, accounts).account_id, 'first');
  const saved = accountSession(accounts[1]); saved.company.name = 'Выдуманный бренд';
  assert.equal(initialSession(saved, accounts).company.name, 'Строительство №2');
  assert.equal(initialSession({...saved, company_id:'U_missing', account_id:'missing'}, accounts).account_id, 'first');
  assert.equal(initialSession(null, []), null);
  const live = {...saved, kind:'live', company_id:'U_3', job_id:'live-job', account_id:undefined,
    company:{name:'Старое название',sector:'Торговля'}};
  assert.equal(initialSession(live,accounts).company.name,'Торговля №3');
  const restored = accountSession(sessionAccount(live));
  assert.equal(restored.kind,'live');
  assert.equal(restored.job_id,'live-job');
  assert.equal(restored.company.name,'Торговля №3');
});

test('compact endpoint supplies screen copy while full claims retain IDs for hover details', async t => {
  const compact = {title:'Кратко',lead:'Суть',facts:[{ids:['e1'],text:'Короткий факт',sources:['news:1']}],
    why:[],important_if:['Есть плавающий долг'],risks:[],opportunities:[],actions:[],bank_view:[]};
  t.mock.method(globalThis,'fetch',async (url, init) => {
    assert.ok(url.endsWith('/v1/jobs/saved/compact'));
    assert.equal(init.headers['X-User-ID'], USER_ID);
    return {ok:true,json:async()=>({'card':compact})};
  });
  const versions = await compactOf('saved');
  const card = toFeed({id:'saved',result:{results:[{status:'ready',insight:{id:'card',title:'Полное название',
    what_happened:[{id:'e1',text:'Полный факт',sources:['news:1']}], hypotheses:[{text:'Условие',sources:[]}]}}]}}).insights[0];
  assert.equal(card.what_happened[0].id,'e1');
  assert.equal(view(card,versions.card).title,'Кратко');
  assert.deepEqual(view(card).important_if,['Условие']);
});

test('streaming chat handles split UTF-8 and SSE frames and returns the persisted answer', async t => {
  const card = {id:'stream-card',job_id:'stream-job',role:'cfo',pipeline_result:{insight:{title:'Тест'}}};
  const answer = {id:'saved-answer',request_id:'request-1',analysis:{response:'Привет, мир!',source_ids:['fact:1']}};
  const payload = 'event: text\ndata: {"text":"Привет"}\n\n' +
    'event: text\ndata: {"text":"Привет, мир!"}\n\n' + `event: complete\ndata: ${JSON.stringify(answer)}\n\n`;
  const bytes = new TextEncoder().encode(payload);
  t.mock.method(globalThis,'fetch',async (url, init) => {
    assert.equal(init.headers['X-User-ID'],USER_ID);
    if (url.endsWith('/import')) return {ok:true,json:async()=>[{id:'stream-uuid'}]};
    assert.ok(url.endsWith('/stream-uuid/ask/stream'));
    assert.equal(JSON.parse(init.body).request_id,'request-1');
    return {ok:true,body:new ReadableStream({start(controller){
      for(let n=0;n<bytes.length;n+=3) controller.enqueue(bytes.slice(n,n+3));
      controller.close();
    }})};
  });
  const texts=[];
  assert.deepEqual(await askQuestion('U_1',card,'Вопрос','request-1',text=>texts.push(text)),answer);
  assert.deepEqual(texts,['Привет','Привет, мир!']);
});

test('an interrupted stream cannot be mistaken for a successfully saved answer', async t => {
  const card={id:'broken-stream',job_id:'broken-job',role:'cfo',pipeline_result:{insight:{title:'Тест'}}};
  t.mock.method(globalThis,'fetch',async url => url.endsWith('/import')
    ? {ok:true,json:async()=>[{id:'broken-uuid'}]}
    : {ok:true,body:new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('event: text\ndata: {"text":"Часть"}\n\n'));controller.close();}})});
  await assert.rejects(askQuestion('U_1',card,'Вопрос','same-retry-id',()=>{}), /до сохранения ответа/);
});
