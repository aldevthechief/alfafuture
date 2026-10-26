import React, { useEffect, useState } from 'react';
import { API_ENABLED, companyProfile, paymentAnalytics, companyCounterparties } from './api.js';
import { insightSignal } from './insightSignal.js';
import { day } from './insightView.js';
import { companySearchNumber } from './siteSearch.js';
import './homeDashboard.css';
import { AnalysisPlaceholder } from './liveInsights.jsx';

const integer=new Intl.NumberFormat('ru-RU');
const compact=new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',notation:'compact',maximumFractionDigits:1});
const currency=new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',maximumFractionDigits:2});
const shortMoney=value=>value==null ? '—' : compact.format(value);
const exactMoney=value=>value==null ? undefined : currency.format(value);

function Icon({ name }) {
  const paths={in:'M18 6 6 18M6 7v11h11',out:'M6 18 18 6M7 6h11v11',people:'M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M16 3a4 4 0 0 1 0 8M22 21v-2a4 4 0 0 0-3-3.87',
    insight:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z',document:'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8ZM14 2v6h6M8 13h8M8 17h5',
    calendar:'M8 2v4M16 2v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z',chat:'M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z'};
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/>{name==='people' && <circle cx="9" cy="7" r="4"/>}</svg>;
}

function Metric({ label, value, title, hint, icon, tone, onClick, loading }) {
  return <button type="button" className={`card hd-metric${tone ? ` hd-metric-${tone}` : ''}`} onClick={onClick} aria-busy={loading}>
    <span className="hd-metric-label">{label}<Icon name={icon}/></span><strong title={title}>{loading ? <span className="hd-skeleton"/> : value}</strong><small>{hint}</small>
  </button>;
}

export default function HomeDashboard({ company, name, role, insights, unreadCount, feedDate, feedState, feedError, feedJob, onOpen, onNavigate, onNotifications }) {
  const [data,setData]=useState({profile:null,payments:null,relationships:null,loading:API_ENABLED,errors:[]});
  const [retry,setRetry]=useState(0);
  useEffect(()=>{
    if (!API_ENABLED) return;
    const controller=new AbortController();
    setData({profile:null,payments:null,relationships:null,loading:true,errors:[]});
    const keys=['profile','payments','relationships'];
    Promise.allSettled([companyProfile(company,controller.signal),paymentAnalytics(company,{limit:1},controller.signal),companyCounterparties(company,{limit:3},controller.signal)]).then(results=>{
      if (controller.signal.aborted) return;
      setData({loading:false,...Object.fromEntries(results.map((result,index)=>[keys[index],result.status==='fulfilled' ? result.value : null])),
        errors:results.flatMap((result,index)=>result.status==='rejected' ? [({profile:'сведения о компании',payments:'платежи',relationships:'контрагенты'})[keys[index]]] : [])});
    });
    return ()=>controller.abort();
  },[company,retry]);
  const profile=data.profile?.profile;
  const overview=profile?.overview;
  const summary=data.payments?.summary;
  const top=insights[0];
  const signal=top && insightSignal(top);
  const feedLoading=['loading','running'].includes(feedState) && !insights.length;
  const relationships=data.relationships?.items || [];
  const products=profile?.bank_products || [];
  return <div className="hd-page">
    <header className="hd-header"><div><span className="hd-eyebrow">АЛЬФА-БИЗНЕС · ОБЗОР</span><h1 className="h-sys">Главная</h1><p>{name} <span>· {role}</span></p></div>
      {feedDate && <span className="hd-updated"><i aria-hidden="true"/>Инсайты обновлены {day(feedDate)}</span>}</header>
    <div className="hd-metrics">
      <Metric label="Поступления" icon="in" tone="in" loading={data.loading} value={shortMoney(summary?.in.amount)} title={exactMoney(summary?.in.amount)} hint={summary ? `${integer.format(summary.in.count)} платежей в выборке` : 'Данные платёжной аналитики'} onClick={()=>onNavigate('payments',{direction:'in'})}/>
      <Metric label="Списания" icon="out" loading={data.loading} value={shortMoney(summary?.out.amount)} title={exactMoney(summary?.out.amount)} hint={summary ? `${integer.format(summary.out.count)} платежей в выборке` : 'Данные платёжной аналитики'} onClick={()=>onNavigate('payments',{direction:'out'})}/>
      <Metric label="Контрагенты" icon="people" loading={data.loading} value={data.relationships ? integer.format(data.relationships.summary.total) : '—'} hint="Покупатели и поставщики" onClick={()=>onNavigate('counterparties')}/>
      <Metric label="Инсайты для вас" icon="insight" tone="insights" loading={feedLoading} value={integer.format(insights.length)} hint={unreadCount ? `Не просмотрено: ${unreadCount}` : insights.length ? 'Все события просмотрены' : 'Новости и аналитика'} onClick={()=>onNavigate('insights')}/>
    </div>
    <p className="hd-source">Платежи за всю доступную выборку. Период в исходных данных не указан.</p>
    {!!data.errors.length && <div className="hd-error" role="alert"><p>Не удалось загрузить: {data.errors.join(', ')}.</p><button type="button" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
    {!API_ENABLED && <p className="hd-error">Для сводки компании подключите API кабинета.</p>}
    <AnalysisPlaceholder live={{state:feedState,feed:{running_job:feedJob,insights}}} />
    <div className="hd-main-grid">
      <section className={`card hd-feature${signal ? ` hd-feature-${signal.tone}` : ''}`} aria-labelledby="hd-feature-title">
        <div className="hd-section-label"><span><Icon name="insight"/>В центре внимания</span>{unreadCount>0 && <button type="button" className="hd-unread" onClick={onNotifications}>Новых · {unreadCount}</button>}</div>
        {top ? <><span className="hd-signal">{signal.label}{signal.high && ' · Высокое влияние'}</span><h2 id="hd-feature-title">{top.compact?.title || top.title}</h2>
          <p className="hd-feature-lead">{top.compact?.lead || top.lead || top.why_it_matters?.[0]?.text}</p>
          <div className="hd-feature-meta">{[day(top.published),top.channel].filter(Boolean).join(' · ')}</div>
          <footer><button type="button" className="btn btn-red" onClick={()=>onOpen(top)}>Разобрать с помощником <span aria-hidden="true">↗</span></button><button type="button" className="hd-text-button" onClick={()=>onNavigate('insights')}>Все инсайты <span aria-hidden="true">→</span></button></footer>
        </> : <><h2 id="hd-feature-title">{feedLoading ? 'Собираем обзор для вашей компании' : 'Важные события появятся здесь'}</h2><p className="hd-feature-lead" role="status">{feedLoading ? 'Загружаем готовые инсайты и аналитические материалы.' : feedError || 'Откройте ленту инсайтов, чтобы посмотреть доступные анализы и уточнить профиль компании.'}</p>
          <footer><button type="button" className="btn btn-red" onClick={()=>onNavigate('insights')}>Перейти к инсайтам</button></footer></>}
      </section>
      <section className="card hd-company" aria-labelledby="hd-company-title" aria-busy={data.loading}>
        <div className="hd-section-label"><span>О компании</span><span className="hd-company-id">№{companySearchNumber(company)}</span></div>
        <h2 id="hd-company-title">Профиль бизнеса</h2>
        {data.loading ? <div className="hd-profile-loading" role="status"><span className="hd-skeleton"/><span className="hd-skeleton"/>Загружаем сведения…</div> : <>
          <p className="hd-industry">{profile?.industry || 'Сведения об отрасли недоступны'}</p><dl><div><dt>Регион</dt><dd>{profile?.region || 'Нет данных'}</dd></div><div><dt>Выручка</dt><dd title={exactMoney(overview?.revenue)}>{shortMoney(overview?.revenue)}</dd></div><div><dt>Юрлиц в группе</dt><dd>{overview?.group_size != null ? integer.format(overview.group_size) : 'Нет данных'}</dd></div></dl>
          {products.length>0 && <div className="hd-products">{products.slice(0,3).map(product=><span key={product}>{product}</span>)}{products.length>3 && <span>+{products.length-3}</span>}</div>}
        </>}
        <button type="button" className="hd-text-button" onClick={()=>onNavigate('company')}>Сведения о компании <span aria-hidden="true">→</span></button>
      </section>
      <section className="card hd-relations" aria-labelledby="hd-relations-title" aria-busy={data.loading}>
        <div className="hd-section-heading"><div><h2 id="hd-relations-title">Ключевые платёжные связи</h2><p>По сумме поступлений и списаний</p></div><Icon name="people"/></div>
        {data.loading ? <p className="hd-empty" role="status">Загружаем контрагентов…</p> : relationships.length ? <ul>{relationships.map(row=><li key={row.id}><button type="button" onClick={()=>onNavigate('counterparties',{query:companySearchNumber(row.id)})}>
          <span className="hd-relation-name"><strong>{row.name}</strong><small>{row.region || 'Регион неизвестен'}{row.same_group && ' · Своя группа'}</small></span>
          <span className="hd-relation-flows">{row.in.count>0 && <span title={exactMoney(row.in.amount)} className="hd-in">↙ {shortMoney(row.in.amount)}</span>}{row.out.count>0 && <span title={exactMoney(row.out.amount)}>↗ {shortMoney(row.out.amount)}</span>}</span><span className="hd-relation-arrow" aria-hidden="true">›</span>
        </button></li>)}</ul> : <p className="hd-empty">{data.relationships ? 'В доступной выборке нет платёжных связей.' : 'Сведения о контрагентах пока недоступны.'}</p>}
        <div className="hd-relations-footer"><span>↙ Получили · ↗ Перечислили</span><button type="button" className="hd-text-button" onClick={()=>onNavigate('payments')}>Платёжная аналитика <span aria-hidden="true">→</span></button></div>
      </section>
      <section className="card hd-shortcuts" aria-labelledby="hd-shortcuts-title"><h2 id="hd-shortcuts-title">Под рукой</h2>
        {[{view:'documents',icon:'document',title:'Документы и отчёты',text:'Выбрать PDF с аналитикой'}, {view:'calendar',icon:'calendar',title:'Календарь событий',text:'Даты из новостей и отчётов'}, {view:'chat',icon:'chat',title:'Чат с банком',text:'Задать вопрос о вашей компании'}].map(item=><button key={item.view} type="button" onClick={()=>onNavigate(item.view)}><span className={`hd-shortcut-icon hd-shortcut-${item.icon}`}><Icon name={item.icon}/></span><span><strong>{item.title}</strong><small>{item.text}</small></span><span className="hd-relation-arrow" aria-hidden="true">›</span></button>)}
      </section>
    </div>
  </div>;
}
