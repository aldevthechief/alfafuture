import React, { useEffect, useRef, useState } from 'react';
import { insightIdentity } from './cabinetToolsState.js';
import { newInsightTracker } from './jobUpdates.js';
import './liveInsights.css';

export function AnalysisPlaceholder({ live }) {
  const job=live.feed?.running_job;
  const stage=job?.stage;
  if (!['running','starting'].includes(live.state)) return null;
  const label=stage?.label || (live.state==='starting' ? 'Запускаем анализ' : 'Агент готовит новые инсайты');
  const ready=(live.feed?.insights || []).filter(card=>card.job_id===job?.id).length;
  return <section className="analysis-placeholder card" aria-label="Ход анализа" aria-busy="true">
    <div className="analysis-stage" role="status" aria-live="polite" aria-atomic="true">
      <span className="analysis-orbit" aria-hidden="true"/><div><strong>{label}</strong>
        <p>{stage?.step>0 ? `Этап ${stage.step} из ${stage.steps}` : 'Ожидаем начала работы'}
          {stage?.total>0 && ` · обработано ${stage.done ?? 0} из ${stage.total}`}
          {ready>0 && ` · готовых инсайтов: ${ready}`}</p></div>
    </div>
    {!ready && <div className="analysis-skeleton" aria-hidden="true"><span/><span/><span/></div>}
    <p className="analysis-note">Готовые инсайты появляются в ленте сразу. Можно продолжать работу в кабинете.</p>
  </section>;
}

export function InsightArrivals({ cards,onOpen }) {
  const tracker=useRef(null);
  if (!tracker.current) tracker.current=newInsightTracker(insightIdentity);
  const [queue,setQueue]=useState([]);
  useEffect(()=>{
    const fresh=tracker.current(cards);
    if (fresh.length) setQueue(previous=>[...previous,...fresh].slice(-3));
  },[cards]);
  const first=queue[0];
  useEffect(()=>{
    if (!first) return;
    const timer=setTimeout(()=>setQueue(previous=>previous.filter(card=>insightIdentity(card)!==insightIdentity(first))),12000);
    return ()=>clearTimeout(timer);
  },[first]);
  const dismiss=card=>setQueue(previous=>previous.filter(item=>insightIdentity(item)!==insightIdentity(card)));
  return <aside className="insight-arrivals" aria-label="Новые инсайты">
    <div className="arrival-announcement" role="status" aria-live="polite" aria-atomic="true">
      {first ? `Новый инсайт: ${first.title}` : ''}
    </div>
    {queue.map(card=><div className="insight-arrival" key={insightIdentity(card)}>
      <div><span>Новый инсайт</span><strong>{card.title}</strong>
        <button type="button" className="arrival-open" onClick={()=>{dismiss(card);onOpen(card);}}>Посмотреть инсайт →</button></div>
      <button type="button" className="arrival-close" aria-label={`Скрыть уведомление: ${card.title}`} onClick={()=>dismiss(card)}>×</button>
    </div>)}
  </aside>;
}
