import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { API_ENABLED, companyCounterparties, counterpartyDetails } from './api.js';
import { ExportControl } from './exportControl.jsx';
import './paymentAnalytics.css';
import './counterparties.css';

const number = new Intl.NumberFormat('ru-RU',{maximumFractionDigits:0});
const currency = new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',maximumFractionDigits:2});
const percent = new Intl.NumberFormat('ru-RU',{style:'percent',maximumFractionDigits:1});
const money = value => value == null ? '—' : currency.format(value);
const share = value => value > 0 && value < .001 ? '<0,1 %' : percent.format(value);
const relations = {in:'Платит нам',out:'Платим ему',both:'В обе стороны'};
const PAGE = 15;

function Group({ row }) {
  return <span className={`cp-group${row.same_group ? ' cp-group-own' : ''}`}>{row.same_group ? 'Своя группа' : row.group_known ? 'Другая группа' : 'Группа неизвестна'}</span>;
}

function Flow({ title, value, direction }) {
  return <section className={`cp-flow cp-flow-${direction}`}><h3>{title}</h3><strong>{money(value.amount)}</strong>
    <dl><div><dt>Количество платежей</dt><dd>{number.format(value.count)}</dd></div>
      <div><dt>Средний платёж</dt><dd>{money(value.average)}</dd></div>
      <div><dt>Доля {direction==='in' ? 'в поступлениях' : 'в списаниях'}</dt><dd>{share(value.share)}</dd></div></dl>
  </section>;
}

function CounterpartyCard({ company, row, onClose }) {
  const dialog = useRef(null);
  const title = useId();
  const [data,setData] = useState(null);
  const [error,setError] = useState('');
  const [retry,setRetry] = useState(0);
  useEffect(() => {
    const previous = document.activeElement;
    const modal = dialog.current;
    modal.showModal();
    return () => { modal.close(); previous?.focus?.({preventScroll:true}); };
  },[]);
  useEffect(() => {
    const controller = new AbortController();
    setError('');
    counterpartyDetails(company,row.id,controller.signal).then(result=>{
      if (!controller.signal.aborted) setData(result);
    }).catch(err=>{ if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  },[company,row.id,retry]);
  const selected = data?.counterparty || row;
  const profile = data?.profile;
  return createPortal(<dialog className="cp-dialog" ref={dialog} aria-labelledby={title}
    onCancel={event=>{event.preventDefault();onClose();}}
    onClick={event=>{if (event.target===event.currentTarget) onClose();}}>
    <div className="cp-dialog-body"><header><div><span className="cp-eyebrow">Карточка контрагента</span>
      <h2 id={title}>{selected.name}</h2><Group row={selected}/></div>
      <button type="button" autoFocus className="cp-close" aria-label="Закрыть карточку контрагента" onClick={onClose}>×</button></header>
      <p className="payments-caption">Платежи между вашей компанией и этим контрагентом за всю доступную выборку.</p>
      <div className="cp-flows"><Flow title="Получили от контрагента" value={selected.in} direction="in"/>
        <Flow title="Перечислили контрагенту" value={selected.out} direction="out"/></div>
      <section className="cp-profile"><h3>О компании</h3>
        <dl><div><dt>Отрасль</dt><dd>{selected.industry || 'Нет данных'}</dd></div>
          <div><dt>ОКВЭД</dt><dd>{selected.industry_code || 'Нет данных'}</dd></div>
          <div><dt>Регион</dt><dd>{selected.region || 'Нет данных'}</dd></div>
          {profile && <><div><dt>Город</dt><dd>{profile.city || 'Нет данных'}</dd></div>
            <div><dt>Идентификатор группы</dt><dd>{profile.group_id || 'Нет данных'}</dd></div>
            <div><dt>Выручка из справочника</dt><dd>{money(profile.revenue)}</dd></div>
            <div><dt>Возраст, лет</dt><dd>{profile.age || 'Нет данных'}</dd></div>
            <div><dt>Численность по ФНС</dt><dd>{profile.staff || 'Нет данных'}</dd></div></>}
        </dl>
        {!data && !error && <p role="status" className="payments-caption">Загружаем справочные данные…</p>}
        {profile && !profile.available && <p className="payments-caption">В справочнике нет этой компании. Показаны доступные платёжные связи.</p>}
        {error && <div role="alert" className="cp-detail-error">{error} <button type="button" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
      </section>
      {!!data?.data_notes.length && <details className="payments-data-note"><summary>Особенности исходных данных</summary>
        <ul>{data.data_notes.map(note=><li key={note}>{note}</li>)}</ul></details>}
      <p className="payments-caption cp-card-note">Доли рассчитаны от всех поступлений и списаний вашей компании соответственно. Переводы между своими счетами исключены. Дат отдельных операций в выборке нет.</p>
    </div>
  </dialog>,document.body);
}

export default function CounterpartiesPage({ company, name, searchRequest }) {
  const [direction,setDirection] = useState('all');
  const [search,setSearch] = useState(searchRequest?.query || '');
  const [query,setQuery] = useState(searchRequest?.query || '');
  const [group,setGroup] = useState('all');
  const [sort,setSort] = useState('amount');
  const [offset,setOffset] = useState(0);
  const [data,setData] = useState(null);
  const [loading,setLoading] = useState(true);
  const [error,setError] = useState('');
  const [retry,setRetry] = useState(0);
  const [selected,setSelected] = useState(null);
  useEffect(()=>{
    if (searchRequest) {
      setSearch(searchRequest.query);setQuery(searchRequest.query);setOffset(0);
      setDirection('all');setGroup('all');setSort('amount');setSelected(null);
    }
  },[searchRequest]);
  useEffect(()=>{
    const timer=setTimeout(()=>{setQuery(search.trim());setOffset(0);},250);
    return ()=>clearTimeout(timer);
  },[search]);
  useEffect(()=>{
    if (!API_ENABLED) {setLoading(false);setError('Для списка контрагентов подключите API кабинета.');return;}
    const controller=new AbortController();
    setLoading(true);setError('');
    companyCounterparties(company,{direction,q:query,group,sort,offset,limit:PAGE},controller.signal)
      .then(result=>{if (!controller.signal.aborted) setData(result);})
      .catch(err=>{if (!controller.signal.aborted) setError(err.message);})
      .finally(()=>{if (!controller.signal.aborted) setLoading(false);});
    return ()=>controller.abort();
  },[company,direction,query,group,sort,offset,retry]);
  const summary=data?.summary;
  return <div className="payments-page cp-page">
    <div className="export-page-heading"><h1 className="h-sys">Контрагенты</h1>
      <ExportControl kind="counterparties" params={{company_id:company,scope:'company',top:200}}
        description="Топ-200 поставщиков и покупателей по сумме платежей, концентрация и группа компании. Фильтры таблицы не ограничивают выгрузку." /></div>
    <p className="h-sub">{name || data?.name} · компании, с которыми связаны ваши платежи</p>
    <p className="payments-caption payments-source-note">Одна компания — одна карточка, даже если платежи идут в обе стороны. Откройте компанию, чтобы посмотреть её профиль и взаимные расчёты.</p>
    {error && <div role="alert" className="payments-error"><div><b>Не удалось загрузить контрагентов</b><p>{error}</p></div>
      <button type="button" className="btn btn-gray" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
    {!data && loading && <div role="status" className="card payments-initial-loading">Загружаем контрагентов…</div>}
    {summary && <div className="payments-metrics">{[['total','Всего контрагентов','Уникальные компании'],['incoming','Платят нам','Есть входящие платежи'],
      ['outgoing','Платим им','Есть исходящие платежи'],['both','В обе стороны','Взаимные расчёты']].map(([key,label,hint])=>
      <div key={key} className="card payments-metric"><span>{label}</span><strong>{number.format(summary[key])}</strong><small>{hint}</small></div>)}</div>}
    <div className="payments-direction cp-directions" role="group" aria-label="Тип платёжной связи">
      {[['all','Все'],['in','Платят нам'],['out','Платим им'],['both','В обе стороны']].map(([id,label])=>
        <button key={id} type="button" aria-pressed={direction===id} onClick={()=>{setDirection(id);setOffset(0);}}>{label}</button>)}
    </div>
    <section className="card payments-counterparties" aria-busy={loading}>
      <div className="payments-table-tools cp-tools"><input type="search" maxLength={120} aria-label="Поиск контрагентов"
        placeholder="Компания, отрасль, регион или номер" value={search} onChange={event=>setSearch(event.target.value)}/>
        <label>Группа<select aria-label="Фильтр по группе" value={group} onChange={event=>{setGroup(event.target.value);setOffset(0);}}>
          <option value="all">Все компании</option><option value="own">Своя группа</option></select></label>
        <label>Сортировка<select aria-label="Сортировка списка контрагентов" value={sort} onChange={event=>{setSort(event.target.value);setOffset(0);}}>
          <option value="amount">По общему обороту</option><option value="in">По поступлениям</option><option value="out">По списаниям</option>
          <option value="count">По числу платежей</option><option value="name">По названию</option></select></label></div>
      <div className="payments-table-scroll"><table><caption className="payments-sr">Платёжные связи компании</caption>
        <thead><tr><th scope="col">Компания</th><th scope="col">Получили от неё</th><th scope="col">Перечислили ей</th><th scope="col">Связь</th></tr></thead>
        <tbody>{!error && data?.items.map(row=><tr key={row.id}>
          <td><button type="button" className="cp-company" onClick={()=>setSelected(row)}>{row.name}<span aria-hidden="true">↗</span></button>
            <span>{[row.industry_code && `ОКВЭД ${row.industry_code}`,row.region || 'Регион неизвестен'].filter(Boolean).join(' · ')}</span>
            <Group row={row}/></td>
          {['in','out'].map(flow=><td key={flow}><b className={flow==='in' ? 'cp-incoming' : ''}>{money(row[flow].amount)}</b>
            <span>Платежей: {number.format(row[flow].count)} · {share(row[flow].share)}</span></td>)}
          <td><span className={`cp-relation cp-relation-${row.relationship}`}>{relations[row.relationship]}</span></td>
        </tr>)}</tbody>
      </table></div>
      {loading && <p role="status" className="payments-empty">Обновляем список…</p>}
      {data && !loading && !error && !data.items.length && <p className="payments-empty">{data.summary.total ? 'По выбранным условиям контрагенты не найдены.' : 'В доступной выборке нет платёжных связей этой компании.'}</p>}
      {data?.total>0 && !error && <div className="payments-pagination"><span>{data.offset+1}–{Math.min(data.offset+PAGE,data.total)} из {number.format(data.total)}</span><div>
        <button type="button" className="btn btn-gray" disabled={loading || offset===0} onClick={()=>setOffset(n=>Math.max(0,n-PAGE))}>Назад</button>
        <button type="button" className="btn btn-gray" disabled={loading || offset+PAGE>=data.total} onClick={()=>setOffset(n=>n+PAGE)}>Далее</button>
      </div></div>}
    </section>
    <p className="payments-data-note">Суммы и количество — по всей доступной выборке CSV. Доли относятся ко всем поступлениям или списаниям компании и не меняются при поиске. Переводы себе исключены.</p>
    {selected && <CounterpartyCard key={selected.id} company={company} row={selected} onClose={()=>setSelected(null)}/>}
  </div>;
}
