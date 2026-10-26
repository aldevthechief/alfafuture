import test from 'node:test';
import assert from 'node:assert/strict';
import { consumeJobEvents, followUpdates, mergeJobFeed, liveState, newInsightTracker } from '../src/jobUpdates.js';
import { insightIdentity } from '../src/cabinetToolsState.js';

const frame=job=>`event: job\r\ndata: ${JSON.stringify(job)}\r\n\r\n`;
const response=text=>new Response(new ReadableStream({start(controller){
  const bytes=new TextEncoder().encode(text);
  // Split UTF-8 characters and CRLF delimiters across arbitrary transport chunks.
  for (let n=0;n<bytes.length;n+=7) controller.enqueue(bytes.slice(n,n+7));
  controller.close();
}}));

test('SSE reads split UTF-8 frames, ignores heartbeats and delivers the final snapshot',async()=>{
  const updates=[];
  const result=await consumeJobEvents(response(': heartbeat\r\n\r\n'+frame({status:'running',stage:{label:'Проверяем инсайт'}})+
    frame({status:'succeeded',feed:{insights:[{id:'one'}]}})),job=>updates.push(job));
  assert.equal(updates.length,2);
  assert.equal(updates[0].stage.label,'Проверяем инсайт');
  assert.equal(result.feed.insights[0].id,'one');
});

test('disconnect polls checked cards then reconnects without dropping them',async()=>{
  let connections=0,polls=0;
  const updates=[];
  await followUpdates({retryMs:0,
    stream:async()=>{connections++;return connections===1 ? response(frame({id:'job',status:'running',feed:{insights:[{id:'one'}]}}))
      : response(frame({id:'job',status:'succeeded',feed:{insights:[{id:'one'},{id:'two'}]}}));},
    poll:async()=>{polls++;return {id:'job',status:'running',feed:{insights:[{id:'one'}]}};},
    onUpdate:job=>updates.push(job),
  });
  assert.equal(polls,1);assert.equal(connections,2);
  assert.deepEqual(updates.at(-1).feed.insights.map(card=>card.id),['one','two']);
});

test('unavailable SSE falls back to polling; abort cancels the old company before any update',async()=>{
  const seen=[];
  await followUpdates({stream:async()=>new Response('',{status:404}),poll:async()=>({status:'failed',feed:{insights:[{id:'kept'}]}}),onUpdate:job=>seen.push(job)});
  assert.equal(seen[0].feed.insights[0].id,'kept');
  const controller=new AbortController();
  await followUpdates({signal:controller.signal,stream:async signal=>{
    controller.abort();assert.equal(signal.aborted,true);throw new Error('cancelled');
  },poll:async()=>assert.fail('old context polled'),onUpdate:()=>assert.fail('old context updated')});
});

test('stalled SSE times out and polls rather than freezing the loading screen',async()=>{
  let polled=false;
  await followUpdates({timeoutMs:5,stream:async()=>new Response(new ReadableStream({start(){}})),
    poll:async()=>{polled=true;return {status:'succeeded'};},onUpdate:()=>{}});
  assert.equal(polled,true);
});

test('live merge preserves saved analyses, replaces rerun duplicates and retains failed ready cards',()=>{
  assert.equal(liveState({insights:[],latest_job:null,running_job:null},undefined).state,'none');
  const feed={insights:[{id:'saved',job_id:'old'},{id:'same',job_id:'old',origin:'news'}],running_job:{id:'new'},errors:[]};
  const job={id:'new',status:'failed',error:'connection lost',feed:{insights:[{id:'same',job_id:'new',origin:'news'}]}};
  const merged=mergeJobFeed(feed,job);
  assert.deepEqual(merged.insights.map(card=>card.job_id),['new','old']);
  assert.equal(merged.running_job,null);
  assert.equal(liveState(merged,'new').state,'ready');
  assert.match(liveState(merged,'new').error,/Готовые инсайты сохранены/);
});

test('notifications baseline saved cards and announce each new card once across reconnect and completion',()=>{
  const track=newInsightTracker(insightIdentity);
  const saved={id:'saved',job_id:'old'},fresh={id:'one',job_id:'new'};
  assert.deepEqual(track(undefined),[]);
  assert.deepEqual(track([saved]),[]);
  assert.deepEqual(track([fresh,saved]),[fresh]);
  assert.deepEqual(track([saved,fresh]),[]);
  assert.deepEqual(track([saved]),[]);
  assert.deepEqual(track([fresh,saved]),[]);
  // Selecting another company/role gets an independent baseline.
  assert.deepEqual(newInsightTracker(insightIdentity)([fresh]),[]);
});
