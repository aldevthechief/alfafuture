import { useCallback, useEffect, useRef, useState } from 'react';
import { API_ENABLED, cabinetFeed, followJob, startAnalysis } from './api.js';
import { jobRevision, liveState, mergeJobFeed } from './jobUpdates.js';

export function useLive(ctx) {
  const [live,setLive]=useState({state:API_ENABLED ? 'loading' : 'off'});
  const [activeJob,setActiveJob]=useState(null);
  const starting=useRef(false);
  useEffect(()=>{
    if (!API_ENABLED) return;
    const controller=new AbortController();
    const requested=activeJob || (ctx.kind==='live' ? ctx.job_id : null);
    let timer,watching=null,watchController=null,snapshot=null;
    const update=job=>{
      if (controller.signal.aborted) return;
      if (snapshot?.id===job.id && jobRevision(snapshot)>jobRevision(job)) return;
      snapshot=job;
      setLive(previous=>({...liveState(mergeJobFeed(previous.feed,job),requested || job.id),initialized:previous.initialized}));
    };
    const watch=id=>{
      if (!id || watching===id) return;
      watchController?.abort();watching=id;
      if (snapshot?.id!==id) snapshot=null;
      watchController=new AbortController();
      followJob(id,update,{signal:watchController.signal,onError:error=>{
        if (!controller.signal.aborted) setLive(previous=>({...previous,error:`Связь прервана, переподключаемся. ${error.message}`}));
      }});
    };
    const refresh=async()=>{
      try {
        let feed=await cabinetFeed(ctx.company,ctx.role,controller.signal);
        if (controller.signal.aborted) return;
        const polled=feed.running_job?.id===snapshot?.id ? feed.running_job : feed.latest_job?.id===snapshot?.id ? feed.latest_job : null;
        // A slower polling response cannot roll back a newer stream snapshot.
        if (snapshot && (!feed.latest_job || feed.latest_job.id===snapshot.id) && jobRevision(snapshot)>jobRevision(polled)) feed=mergeJobFeed(feed,snapshot);
        setLive({...liveState(feed,requested || feed.latest_job?.id),initialized:true});
        watch(feed.running_job?.id);
      } catch(error) {
        if (!controller.signal.aborted) setLive(previous=>({...previous,
          state:previous.feed?.running_job ? 'running' : previous.feed?.insights.length ? 'ready' : 'error',error:error.message}));
      } finally {
        if (!controller.signal.aborted) timer=setTimeout(refresh,5000);
      }
    };
    refresh();
    if (requested) watch(requested);
    return ()=>{controller.abort();watchController?.abort();clearTimeout(timer);};
  },[ctx.company,ctx.role,ctx.job_id,ctx.kind,activeJob]);
  const run=useCallback(async()=>{
    if (starting.current) return;
    starting.current=true;
    setLive(previous=>({...previous,state:'starting',error:null}));
    try {const job=await startAnalysis(ctx.company,ctx.role);setActiveJob(job.id);}
    catch(error) {setLive(previous=>({...previous,state:previous.feed?.insights.length ? 'ready' : 'error',error:error.message}));}
    finally {starting.current=false;}
  },[ctx.company,ctx.role]);
  return [live,run];
}
