import React, { useEffect, useState } from 'react';
import { API_ENABLED, paymentAnalytics } from './api.js';
import './paymentAnalytics.css';

const number = new Intl.NumberFormat('ru-RU', {maximumFractionDigits:0});
const currency = new Intl.NumberFormat('ru-RU', {style:'currency',currency:'RUB',maximumFractionDigits:2});
const compact = new Intl.NumberFormat('ru-RU', {style:'currency',currency:'RUB',notation:'compact',maximumFractionDigits:1});
const percent = new Intl.NumberFormat('ru-RU', {style:'percent',maximumFractionDigits:1});
const share = value => value > 0 && value < .001 ? '<0,1 %' : percent.format(value);
const money = value => value == null ? '—' : currency.format(value);
const shortMoney = value => value == null ? '—' : compact.format(value);
const PAGE = 15;

function Metric({ label, value, exact, hint, tone }) {
  return <div className={`card payments-metric${tone ? ` payments-metric-${tone}` : ''}`}>
    <span>{label}</span><strong title={exact}>{value}</strong><small>{hint}</small>
  </div>;
}

function Distribution({ title, items, knownShare }) {
  return <section className="card payments-distribution"><h2>{title}</h2>
    {!items.length ? <p className="payments-empty">Нет платежей для распределения.</p> : <>
      <p className="payments-caption">Данные известны для {percent.format(knownShare)} суммы потока</p>
      <ul>{items.map((item,index) => <li key={`${index}:${item.label}`}>
        <div className="payments-bar-label"><span title={item.label}>{item.label}</span><b>{share(item.share)}</b></div>
        <div className={`payments-bar${item.unknown ? ' payments-bar-unknown' : ''}`} aria-hidden="true"><i style={{width:`${Math.min(100,Math.max(0,item.share*100))}%`}} /></div>
        <small>{money(item.amount)} · контрагентов: {number.format(item.counterparties_count)}</small>
      </li>)}</ul>
    </>}
  </section>;
}

export default function PaymentAnalyticsPage({ company, name, searchRequest }) {
  const [direction,setDirection] = useState(searchRequest?.direction || 'out');
  const [search,setSearch] = useState(searchRequest?.query || '');
  const [query,setQuery] = useState(searchRequest?.query || '');
  const [sort,setSort] = useState('amount');
  const [offset,setOffset] = useState(0);
  const [data,setData] = useState(null);
  const [loading,setLoading] = useState(true);
  const [error,setError] = useState('');
  const [retry,setRetry] = useState(0);
  useEffect(()=>{
    if (searchRequest) {setSearch(searchRequest.query);setQuery(searchRequest.query);setDirection(searchRequest.direction);setOffset(0);setSort('amount');}
  },[searchRequest]);
  useEffect(() => {
    const timer = setTimeout(() => { setQuery(search.trim()); setOffset(0); },250);
    return () => clearTimeout(timer);
  },[search]);
  useEffect(() => {
    if (!API_ENABLED) { setLoading(false); setError('Для платёжной аналитики подключите API кабинета.'); return; }
    const controller = new AbortController();
    setLoading(true); setError('');
    paymentAnalytics(company,{direction,q:query,sort,offset,limit:PAGE},controller.signal)
      .then(result => { if (!controller.signal.aborted) setData(result); })
      .catch(err => { if (!controller.signal.aborted) setError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  },[company,direction,query,sort,offset,retry]);
  const summary = data?.summary;
  const flow = data?.flows[direction];
  const totalCount = summary ? summary.in.count + summary.out.count : 0;
  const average = summary && totalCount ? (summary.in.amount+summary.out.amount)/totalCount : null;
  const shown = data && data.direction === direction && !error;
  return <div className="payments-page">
    <h1 className="h-sys">Платёжная аналитика</h1>
    <p className="h-sub">{name || data?.name} · структура поступлений и списаний</p>
    <p className="payments-caption payments-source-note">Агрегированные платежи из доступной выборки. Период в исходных данных не указан. Переводы между своими счетами считаются отдельно.</p>
    {error && <div className="payments-error" role="alert"><div><b>Не удалось загрузить платёжную аналитику</b><p>{error}</p></div>
      <button type="button" className="btn btn-gray" onClick={() => setRetry(n=>n+1)}>Повторить</button></div>}
    {!data && loading && <div className="card payments-initial-loading" role="status">Загружаем платёжную аналитику…</div>}
    {summary && <div className="payments-metrics">
      <Metric label="Поступления" value={shortMoney(summary.in.amount)} exact={money(summary.in.amount)} tone="in" hint={`${number.format(summary.in.count)} платежей`} />
      <Metric label="Списания" value={shortMoney(summary.out.amount)} exact={money(summary.out.amount)} tone="out" hint={`${number.format(summary.out.count)} платежей`} />
      <Metric label="Всего платежей" value={number.format(totalCount)} hint="Входящие и исходящие" />
      <Metric label="Средний платёж" value={shortMoney(average)} exact={money(average)} hint="Сумма потоков / число платежей" />
    </div>}
    <details className="card payments-export">
      <summary><span className="payments-export-icon" aria-hidden="true">CSV</span><span><strong>Сводка платежей</strong><small>Показатели компании в одном файле · CSV, UTF-8</small></span><span className="payments-export-chevron" aria-hidden="true">⌄</span></summary>
      <div className="payments-export-body">
        <ul><li>Поступления, списания и средний платёж</li><li>Доли крупнейшего контрагента, топ-5 и своей группы</li><li>Распределение по отраслям и регионам</li><li>Переводы между своими счетами отдельной строкой</li></ul>
        <p className="payments-caption">За всю доступную выборку, суммы в рублях. Дат и реквизитов отдельных операций в исходных данных нет.</p>
        <div className="payments-export-action"><p id="payments-export-status">Скачивание пока недоступно. Показатели доступны в аналитике ниже.</p><button type="button" className="btn btn-red" disabled aria-describedby="payments-export-status">Скачать CSV</button></div>
      </div>
    </details>
    <div className="payments-direction" role="group" aria-label="Направление платежей">
      {[['out','Исходящие'],['in','Входящие']].map(([id,label]) => <button key={id} type="button" aria-pressed={direction===id}
        onClick={() => { setDirection(id); setOffset(0); }}>{label}</button>)}
    </div>
    {flow && <>
      <div className="card payments-concentration">
        <div><span>Контрагентов</span><strong>{number.format(flow.counterparties_count)}</strong></div>
        <div><span>Доля крупнейшего</span><strong>{share(flow.top1_share)}</strong></div>
        <div><span>Доля топ-5</span><strong>{share(flow.top5_share)}</strong></div>
        <div><span>Внутри группы</span><strong>{share(flow.within_group_share)}</strong></div>
      </div>
      <div className="payments-distributions">
        <Distribution title="По отраслям контрагентов" items={flow.industries} knownShare={flow.known_industry_share} />
        <Distribution title="По регионам контрагентов" items={flow.regions} knownShare={flow.known_region_share} />
      </div>
    </>}
    <section className="card payments-counterparties" aria-busy={loading}>
      <div className="payments-table-heading"><h2>Контрагенты</h2><p className="payments-caption">Доля рассчитана от всей суммы выбранного направления.</p></div>
      <div className="payments-table-tools">
        <input type="search" maxLength={120} aria-label="Поиск по контрагентам" placeholder="Компания, отрасль, регион или номер" value={search}
          onChange={event => setSearch(event.target.value)} />
        <label>Сортировка<select aria-label="Сортировка контрагентов" value={sort} onChange={event => { setSort(event.target.value); setOffset(0); }}>
          <option value="amount">По сумме</option><option value="count">По числу платежей</option></select></label>
      </div>
      <div className="payments-table-scroll"><table><caption className="payments-sr">Контрагенты: {direction==='out' ? 'исходящие' : 'входящие'} платежи</caption>
        <thead><tr><th scope="col">Компания</th><th scope="col">Сумма</th><th scope="col">Доля</th><th scope="col">Платежей</th><th scope="col">Средний платёж</th></tr></thead>
        <tbody>{shown && data.counterparties.map(row => <tr key={row.id}>
          <td><strong>{row.name}</strong><span>{[row.industry_code && `ОКВЭД ${row.industry_code}`,row.region || 'Регион неизвестен'].filter(Boolean).join(' · ')}</span>
            {row.same_group && <small className="payments-group-label">Своя группа</small>}</td>
          <td>{money(row.amount)}</td><td>{share(row.share)}</td><td>{number.format(row.count)}</td><td>{money(row.average)}</td>
        </tr>)}</tbody>
      </table></div>
      {loading && <p role="status" className="payments-empty">Загружаем данные…</p>}
      {shown && !loading && !data.counterparties.length && <p className="payments-empty">{query ? 'Контрагенты не найдены. Попробуйте другой запрос.' : 'В выборке нет платежей этого направления.'}</p>}
      {shown && data.total>0 && <div className="payments-pagination"><span>{offset+1}–{Math.min(offset+PAGE,data.total)} из {number.format(data.total)}</span>
        <div><button type="button" className="btn btn-gray" disabled={loading || offset===0} onClick={() => setOffset(n=>Math.max(0,n-PAGE))}>Назад</button>
          <button type="button" className="btn btn-gray" disabled={loading || offset+PAGE>=data.total} onClick={() => setOffset(n=>n+PAGE)}>Далее</button></div></div>}
    </section>
    {data && <section className="payments-data-note">
      <p><b>Переводы между своими счетами:</b> {money(data.self_transfers.amount)} · {number.format(data.self_transfers.count)} платежей. Не включены в показатели выше.</p>
      {!!data.data_notes.length && <details><summary>Особенности данных компании · {data.data_notes.length}</summary><ul>{data.data_notes.map(note=><li key={note}>{note}</li>)}</ul></details>}
    </section>}
  </div>;
}
