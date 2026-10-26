import React, { useEffect, useState } from 'react';
import { API_ENABLED, companyProfile, paymentAnalytics, companyCounterparties } from './api.js';
import { findInsights, insightIdentity } from './cabinetToolsState.js';
import { SEARCH_SECTIONS, findSearchItems, companySearchItems, paymentSearchItems, counterpartyPaymentItems, companySearchNumber } from './siteSearch.js';

const scopes=[['all','Везде'],['insights','Инсайты'],['company','Компания'],['counterparties','Контрагенты'],['payments','Платежи'],['sections','Разделы']];

function SearchResult({ item, onNavigate }) {
  return <button type="button" className="ct-site-result" onClick={()=>onNavigate(item.view,item)}>
    {item.icon && <span className="ct-site-icon" aria-hidden="true">{item.icon}</span>}
    <span><strong>{item.title}</strong>{item.text && <small>{item.text}</small>}</span><span className="ct-arrow" aria-hidden="true">›</span>
  </button>;
}

function Results({ title, children, more }) {
  return <section className="ct-site-group" aria-label={title}><div className="ct-list-heading"><h3>{title}</h3></div>
    {children}{more}</section>;
}

export default function SiteSearch({ company, insights, loading, error, inputRef, onNavigate, onOpen, InsightResult }) {
  const [query,setQuery]=useState('');
  const [scope,setScope]=useState('all');
  const [retry,setRetry]=useState(0);
  const [base,setBase]=useState({loading:API_ENABLED,profile:null,payments:null,errors:[]});
  const [remote,setRemote]=useState({query:null,rows:[],total:0,error:'',loading:false});
  const inScope=id=>scope==='all' || scope===id;
  useEffect(()=>{
    if (!API_ENABLED) return;
    const controller=new AbortController();
    setBase({loading:true,profile:null,payments:null,errors:[]});
    Promise.allSettled([companyProfile(company,controller.signal),paymentAnalytics(company,{limit:1},controller.signal)]).then(results=>{
      if (controller.signal.aborted) return;
      setBase({loading:false,profile:results[0].status==='fulfilled' ? results[0].value : null,
        payments:results[1].status==='fulfilled' ? results[1].value : null,
        errors:results.flatMap((result,index)=>result.status==='rejected' ? [index===0 ? 'Компания' : 'Платежи'] : [])});
    });
    return ()=>controller.abort();
  },[company,retry]);
  const needsRelationships=scope==='all' || scope==='counterparties' || scope==='payments';
  useEffect(()=>{
    if (!API_ENABLED || !needsRelationships) return;
    const controller=new AbortController();
    setRemote({query,rows:[],total:0,error:'',loading:true});
    const timer=setTimeout(()=>{
      companyCounterparties(company,{q:query.trim(),limit:5},controller.signal).then(data=>{
        if (!controller.signal.aborted) setRemote({query,rows:data.items,total:data.total,error:'',loading:false});
      }).catch(error=>{if (!controller.signal.aborted) setRemote({query,rows:[],total:0,error:error.message,loading:false});});
    },250);
    return ()=>{clearTimeout(timer);controller.abort();};
  },[company,query,needsRelationships,retry]);
  const results=findInsights(insights,query);
  const sections=findSearchItems(SEARCH_SECTIONS,query);
  const companyItems=findSearchItems(companySearchItems(base.profile),query);
  const relationships=remote.query===query ? remote.rows : [];
  const paymentItems=[...findSearchItems(paymentSearchItems(base.payments),query),...(query.trim() ? counterpartyPaymentItems(relationships) : [])];
  const pending=(base.loading && (inScope('company') || inScope('payments'))) || (API_ENABLED && needsRelationships && (remote.loading || remote.query!==query)) || (inScope('insights') && loading && !insights.length);
  const any=(inScope('sections') && sections.length) || (inScope('insights') && results.length) || (inScope('company') && companyItems.length) || (inScope('counterparties') && relationships.length) || (inScope('payments') && paymentItems.length);
  const renderItems=items=>items.map((item,index)=><SearchResult key={`${item.title}:${index}`} item={item} onNavigate={onNavigate}/>);
  return <div className="ct-site-search">
    <div className="ct-search-field"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="6" stroke="currentColor" strokeWidth="2"/><path d="m15 15 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
      <input ref={inputRef} type="search" maxLength={120} aria-label="Поиск по всему кабинету" placeholder="Инсайт, контрагент, данные компании или раздел" value={query} onChange={event=>setQuery(event.target.value)}/>
    </div>
    <div className="ct-tabs ct-site-tabs" role="group" aria-label="Где искать">{scopes.map(([id,label])=><button key={id} type="button" aria-pressed={scope===id} onClick={()=>setScope(id)}>{label}</button>)}</div>
    <p className="ct-note">По данным выбранной компании, её инсайтам и разделам кабинета.</p>
    <div aria-busy={!!pending}>
      {inScope('sections') && !!sections.length && <Results title="Разделы кабинета">{renderItems(sections)}</Results>}
      {inScope('insights') && !!results.length && <Results title={`Инсайты · ${results.length}`} more={results.length>8 && <button type="button" className="ct-text-button ct-site-more" onClick={()=>onNavigate('insights',{query})}>Все найденные инсайты →</button>}>
        {results.slice(0,8).map(insight=><InsightResult key={insightIdentity(insight)} insight={insight} onOpen={onOpen}/>)}</Results>}
      {inScope('company') && !!companyItems.length && <Results title="Данные компании">{renderItems(companyItems.slice(0,6))}</Results>}
      {inScope('counterparties') && !!relationships.length && <Results title="Контрагенты" more={<button type="button" className="ct-text-button ct-site-more" onClick={()=>onNavigate('counterparties',{query})}>Все найденные контрагенты · {remote.total} →</button>}>
        {renderItems(relationships.map(row=>({title:row.name,text:[row.industry,row.region].filter(Boolean).join(' · '),view:'counterparties',query:companySearchNumber(row.id)})))}</Results>}
      {inScope('payments') && !!paymentItems.length && <Results title="Платёжная аналитика">{renderItems(paymentItems.slice(0,6))}</Results>}
    </div>
    {pending && <p className="ct-note ct-site-status" role="status">Ищем в данных компании…</p>}
    {!pending && !any && <p className="ct-empty" role="status">Ничего не найдено. Попробуйте другое слово, тег или номер компании.</p>}
    {!API_ENABLED && <p className="ct-note ct-site-status">Для поиска в данных компании, платежах и контрагентах подключите API кабинета.</p>}
    {inScope('insights') && error && !insights.length && <p className="ct-site-error" role="alert">Лента инсайтов недоступна: {error}</p>}
    {!!base.errors.filter(category=>scope==='all' || (scope==='company' && category==='Компания') || (scope==='payments' && category==='Платежи')).length && <div className="ct-site-error" role="alert">Часть данных недоступна: {base.errors.join(', ')}. <button type="button" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
    {needsRelationships && remote.query===query && remote.error && <div className="ct-site-error" role="alert">Не удалось загрузить платёжные связи. <button type="button" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
  </div>;
}
