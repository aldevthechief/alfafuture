import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { insightIdentity } from './cabinetToolsState.js';
import { insightSignal } from './insightSignal.js';
import { day } from './insightView.js';
import SiteSearch from './siteSearch.jsx';

function InsightResult({ insight, unread, onOpen }) {
  const signal=insightSignal(insight);
  const title=insight.compact?.title || insight.title;
  const lead=insight.compact?.lead || insight.lead;
  return <button type="button" className={`ct-result${unread ? ' ct-result-unread' : ''}`} onClick={()=>onOpen(insight)}>
    <span className={`ct-result-icon ct-result-${signal.tone}`} aria-hidden="true">{signal.emoji}</span>
    <span className="ct-result-copy"><span className="ct-result-meta">{[day(insight.published),insight.channel].filter(Boolean).join(' · ')}
      {unread && <span className="ct-new-dot" aria-label="Не просмотрено"/>}</span>
      <strong>{title}</strong>{lead && <span className="ct-result-lead">{lead}</span>}</span>
    <span className="ct-arrow" aria-hidden="true">›</span>
  </button>;
}

export default function CabinetTools({ tool, company, companyName, insights, unread, seen, loading, error,
  onClose, onOpen, onNavigate, onReadAll }) {
  const modal=useRef(null);
  const input=useRef(null);
  const titleId=useId();
  const [notifications,setNotifications]=useState(unread.length ? 'new' : 'all');
  useEffect(()=>{
    const previous=document.activeElement;
    const dialog=modal.current;
    dialog.showModal();
    if (tool==='search') input.current?.focus();
    return ()=>{dialog.close();previous?.focus?.({preventScroll:true});};
  },[tool]);
  const titles={search:'Поиск',notifications:'Новые инсайты'};
  const shownNotifications=notifications==='new' ? unread : insights;
  const navigate=view=>{onClose();window.showView(view);};
  return createPortal(<dialog id="cabinet-tool-panel" className={`cabinet-tool ct-${tool}`} ref={modal} aria-labelledby={titleId}
    onCancel={event=>{event.preventDefault();onClose();}} onClick={event=>{if (event.target===event.currentTarget) onClose();}}>
    <div className="ct-body"><header className="ct-header"><div><span className="ct-company">{companyName}</span><h2 id={titleId}>{titles[tool]}</h2></div>
      <button type="button" className="ct-close" aria-label="Закрыть окно" onClick={onClose}>×</button></header>
      {tool==='search' && <SiteSearch company={company} insights={insights} loading={loading} error={error} inputRef={input} onNavigate={onNavigate} onOpen={onOpen} InsightResult={InsightResult}/>}
      {tool==='notifications' && <>
        <div className="ct-notification-actions"><div className="ct-tabs" role="group" aria-label="Показать уведомления">
          <button type="button" aria-pressed={notifications==='new'} onClick={()=>setNotifications('new')}>Новые · {unread.length}</button>
          <button type="button" aria-pressed={notifications==='all'} onClick={()=>setNotifications('all')}>Все · {insights.length}</button></div>
          <button type="button" className="ct-read-all" disabled={!unread.length} onClick={onReadAll}>Прочитать все</button></div>
        {loading && !insights.length ? <p role="status" className="ct-empty">Загружаем инсайты…</p> : error && !insights.length ? <p role="alert" className="ct-empty">Не удалось загрузить ленту. {error}</p>
          : shownNotifications.length ? <div className="ct-results">{shownNotifications.map(insight=><InsightResult key={insightIdentity(insight)} insight={insight}
            unread={!seen.includes(insightIdentity(insight))} onOpen={onOpen}/>)}</div>
          : <div className="ct-empty"><span className="ct-all-read" aria-hidden="true">✓</span><b>{insights.length ? 'Вы всё посмотрели' : 'Инсайтов пока нет'}</b>
            <p>{insights.length ? 'Новые события появятся здесь после обновления ленты.' : 'Готовые события для вашей компании появятся здесь.'}</p></div>}
        <footer className="ct-footer"><span className="ct-note">Просмотренные отмечаются в этом браузере.</span><button type="button" className="ct-text-button" onClick={()=>navigate('insights')}>Вся лента <span aria-hidden="true">→</span></button></footer>
      </>}
    </div>
  </dialog>,document.body);
}
