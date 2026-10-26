import test from 'node:test';
import assert from 'node:assert/strict';
import { suggestedFor } from '../src/api.js';
import { applyTopOrder } from '../src/feedOrder.js';

test('ranking retains every card, puts ranked IDs first and preserves new arrival order',()=>{
  const cards=['new-a','b','a','new-b'].map(id=>({id}));
  const feed={insights:cards,stage:{code:'writing'}};
  const sorted=applyTopOrder(feed,['absent','a','a',null,'b']);
  assert.deepEqual(sorted.insights.map(c=>c.id),['a','b','new-a','new-b']);
  assert.deepEqual(feed.insights,cards);
  assert.equal(sorted.stage,feed.stage);
  assert.equal(applyTopOrder(feed,null),feed);
});

test('question chips share one read request per job and cannot send a question themselves',async t=>{
  const calls=[];
  t.mock.method(globalThis,'fetch',async (url,init)=>{
    calls.push(url);assert.equal(init.method,undefined);
    return {ok:true,json:async()=>({a:['Почему?','Что делать?','Когда?'],b:['Другой вопрос?',null,'нет вопроса']})};
  });
  const [a,b]=await Promise.all([suggestedFor('precomputed-test-job','a'),suggestedFor('precomputed-test-job','b')]);
  assert.deepEqual(a,['Почему?','Что делать?','Когда?']);
  assert.deepEqual(b,['Другой вопрос?']);
  assert.equal(calls.length,1);
});

test('unavailable suggestions can be retried without breaking the insight',async t=>{
  let calls=0;
  t.mock.method(globalThis,'fetch',async()=>{
    if (++calls===1) throw new Error('Offline');
    return {ok:true,json:async()=>({a:['Почему?']})};
  });
  assert.deepEqual(await suggestedFor('precomputed-retry-job','a'),[]);
  assert.deepEqual(await suggestedFor('precomputed-retry-job','a'),['Почему?']);
  assert.equal(calls,2);
});
