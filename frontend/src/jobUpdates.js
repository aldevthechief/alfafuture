// Fetch-based SSE keeps X-User-ID on every connection. A lost stream polls before reconnecting.
export const JOB_DONE = ['succeeded','partial','failed','interrupted'];
export const jobRevision=job=>job?.revision ?? job?.stage?.seq ?? -1;

export function pause(ms, signal) {
  return new Promise(resolve => {
    if (signal?.aborted) return resolve();
    const finish=()=>{clearTimeout(timer);signal?.removeEventListener('abort',finish);resolve();};
    const timer=setTimeout(finish,ms);
    signal?.addEventListener('abort',finish,{once:true});
  });
}

export async function consumeJobEvents(response,onUpdate,signal,onChunk=()=>{}) {
  if (!response.ok || !response.body) throw new Error('Поток обновлений недоступен');
  const reader=response.body.getReader();
  const decoder=new TextDecoder();
  let buffer='';
  const abort=()=>{reader.cancel().catch(()=>{});};
  signal?.addEventListener('abort',abort,{once:true});
  try {
    while (!signal?.aborted) {
      const {value,done}=await reader.read();
      if (done) break;
      onChunk();
      buffer=(buffer+decoder.decode(value,{stream:true})).replaceAll('\r\n','\n');
      if (buffer.length>2*1024*1024) throw new Error('Слишком большое обновление');
      let end;
      while ((end=buffer.indexOf('\n\n'))>=0) {
        const frame=buffer.slice(0,end); buffer=buffer.slice(end+2);
        const data=frame.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');
        if (!data || signal?.aborted) continue;
        const job=JSON.parse(data);
        onUpdate(job);
        if (JOB_DONE.includes(job.status)) return job;
      }
    }
    return null;
  } finally {
    signal?.removeEventListener('abort',abort);
    await reader.cancel().catch(()=>{});
    reader.releaseLock();
  }
}

export async function followUpdates({stream,poll,onUpdate,onError=()=>{},signal,retryMs=2000,timeoutMs=25000}) {
  while (!signal?.aborted) {
    const connection=new AbortController();
    const abort=()=>connection.abort();
    signal?.addEventListener('abort',abort,{once:true});
    let timer;
    const renew=()=>{clearTimeout(timer);timer=setTimeout(abort,timeoutMs);};
    renew();
    try {
      const response=await stream(connection.signal);
      const job=await consumeJobEvents(response,onUpdate,connection.signal,renew);
      if (job) return job;
    } catch(error) {
      if (signal?.aborted) return;
    } finally {
      clearTimeout(timer);connection.abort();signal?.removeEventListener('abort',abort);
    }
    if (signal?.aborted) return;
    try {
      const job=await poll(signal);
      if (signal?.aborted) return;
      onUpdate(job);
      if (JOB_DONE.includes(job.status)) return job;
    } catch(error) {if (!signal?.aborted) onError(error);}
    await pause(retryMs,signal);
  }
}

const sourceKey=card=>JSON.stringify([card.origin || (card.channel==='Аналитика Банка' ? 'bank' : 'news'),card.id]);

export function mergeJobFeed(feed,job) {
  const base=feed || {insights:[],errors:[]};
  const cards=job.feed?.insights;
  const keys=new Set((cards || []).map(sourceKey));
  const insights=cards ? [...cards,...base.insights.filter(card=>card.job_id!==job.id && !keys.has(sourceKey(card)))] : base.insights;
  const terminal=JOB_DONE.includes(job.status);
  return {...base,insights,computed_at:job.feed?.computed_at || base.computed_at,
    running_job:terminal ? (base.running_job?.id===job.id ? null : base.running_job) : job,
    latest_job:job,errors:job.feed?.errors || base.errors};
}

export function liveState(feed,requested) {
  const latest=feed.latest_job;
  const failed=!!latest && !!requested && latest.id===requested && ['failed','interrupted'].includes(latest.status);
  return {feed,state:feed.running_job ? 'running' : feed.insights.length ? 'ready' : failed ? 'error' : 'none',
    error:failed ? `${latest.error || 'Анализ прерван'}. Готовые инсайты сохранены.` : null};
}

export function newInsightTracker(identity) {
  let seen=null;
  return cards=>{
    if (!cards) return [];
    const fresh=seen ? cards.filter(card=>!seen.has(identity(card))) : [];
    seen=new Set([...(seen || []),...cards.map(identity)]);
    return fresh;
  };
}
