import React, { useEffect, useRef, useState } from 'react';
import { API_ENABLED, companyProfile } from './api.js';
import './companyPage.css';

const integer = new Intl.NumberFormat('ru-RU');
const currency = new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',maximumFractionDigits:0});
const compact = new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',notation:'compact',maximumFractionDigits:1});
const money = value => value == null ? 'Нет данных' : currency.format(value);

function Facts({ items, empty='В доступных данных сведений нет.' }) {
  return items.length ? <ul className="co-facts">{items.map((fact,index)=><li key={fact.id || index}>
    <span>{fact.text}</span>{fact.source && <details className="co-fact-source"><summary>Источник</summary><p>{fact.source}</p></details>}
  </li>)}</ul> : <p className="co-note">{empty}</p>;
}

function Metric({ label, value, title, hint }) {
  return <div className="card co-metric"><span>{label}</span><strong title={title}>{value}</strong><small>{hint}</small></div>;
}

export default function CompanyPage({ company, name, searchRequest }) {
  const page=useRef(null);
  const [data,setData] = useState(null);
  const [error,setError] = useState('');
  const [loading,setLoading] = useState(true);
  const [retry,setRetry] = useState(0);
  useEffect(()=>{
    if (!API_ENABLED) {setLoading(false);setError('Для раздела компании подключите API кабинета.');return;}
    const controller=new AbortController();
    setLoading(true);setError('');
    companyProfile(company,controller.signal).then(result=>{
      if (!result.profile.overview) throw new Error('Обновите API кабинета: отсутствуют данные раздела компании.');
      if (!controller.signal.aborted) setData(result);
    }).catch(err=>{if (!controller.signal.aborted) setError(err.message);})
      .finally(()=>{if (!controller.signal.aborted) setLoading(false);});
    return ()=>controller.abort();
  },[company,retry]);
  const profile=data?.profile;
  const overview=profile?.overview;
  const facts=overview?.facts || [];
  const inCategory=(...categories)=>facts.filter(f=>categories.includes(f.category));
  const groupFacts=facts.filter(f=>f.category.startsWith('group'));
  useEffect(()=>{
    if (!data || !['overview','metrics','team','products','trade','group'].includes(searchRequest?.target)) return;
    const section=page.current?.querySelector(`[data-company-section="${searchRequest.target}"]`);
    section?.querySelectorAll('details.co-more').forEach(details=>{details.open=true;});
    section?.scrollIntoView({block:'center',behavior:'instant'});
  },[data,searchRequest]);
  return <div className="co-page" ref={page} data-search-target={searchRequest?.target}>
    <h1 className="h-sys">Компания</h1>
    <p className="h-sub">{name || profile?.name} · сведения о вашем бизнесе</p>
    <p className="co-note co-intro">Профиль из данных компании и расчётных фактов. Эти же сведения используются для персонализации инсайтов.</p>
    {error && <div className="co-error" role="alert"><div><b>Не удалось загрузить компанию</b><p>{error}</p></div>
      <button className="btn btn-gray" type="button" onClick={()=>setRetry(n=>n+1)}>Повторить</button></div>}
    {!data && loading && <div className="card co-loading" role="status">Загружаем сведения о компании…</div>}
    {overview && <>
      <div className="co-metrics" data-company-section="metrics">
        <Metric label="Выручка компании" value={overview.revenue == null ? 'Нет данных' : compact.format(overview.revenue)} title={money(overview.revenue)} hint="По доступным данным о компании"/>
        <Metric label="Масштаб бизнеса" value={overview.size_class || 'Нет данных'} hint="По выручке и численности"/>
        <Metric label="Юрлиц в группе" value={overview.group_size == null ? 'Нет данных' : integer.format(overview.group_size)} hint="В доступной выборке"/>
        <Metric label="Регион" value={profile.region || 'Нет данных'} hint="Регион регистрации"/>
      </div>
      <div className="co-grid">
        <section className="card co-section" data-company-section="overview"><h2>Основные сведения</h2><dl className="co-fields">
          <div><dt>Название</dt><dd>{name || profile.name}</dd></div>
          <div><dt>Отрасль</dt><dd>{profile.industry}</dd></div>
          <div><dt>ОКВЭД</dt><dd>{overview.industry_code || 'Нет данных'}</dd></div>
        </dl><Facts items={inCategory('geo','ownership')}/></section>
        <section className="card co-section" data-company-section="team"><h2>Команда и развитие</h2><Facts items={inCategory('profile')}/>
          <p className="co-note">Численность указана градацией из источника, без оценки точного числа сотрудников.</p></section>
        <section className="card co-section" data-company-section="products"><h2>Продукты Альфа-Банка</h2>
          {profile.bank_products.length ? <ul className="co-chips">{profile.bank_products.map(product=><li key={product}>{product}</li>)}</ul>
            : <p className="co-note">Нет отметок об использовании банковских продуктов.</p>}
          <p className="co-note">Отметки из доступного профиля компании. Состав действующих договоров может отличаться.</p>
          {!!inCategory('bank').length && <details className="co-more"><summary>Сведения об обслуживании</summary><Facts items={inCategory('bank')}/></details>}
        </section>
        <section className="card co-section" data-company-section="trade"><h2>Внешнеторговые связи</h2>
          <span className={`co-trade-status${overview.trade.in_customs_database ? ' co-trade-active' : ''}`}>
            {overview.trade.in_customs_database === true ? 'Есть отметка ФТС' : overview.trade.in_customs_database === false ? 'Нет отметки ФТС' : 'Статус ФТС неизвестен'}</span>
          {overview.trade.countries.length ? <><p className="co-note">Страны контрагентов в доступных данных:</p>
            <ul className="co-chips">{overview.trade.countries.map(country=><li key={country}>{country}</li>)}</ul></>
            : <p className="co-note">Страны контрагентов не указаны. Это не подтверждает отсутствие внешнеторговых операций.</p>}
          <p className="co-note">Направление операций не указано: связи не разделяем на импорт и экспорт. Список стран может быть неполным.</p>
          {!!inCategory('trade').length && <details className="co-more"><summary>Подтверждения из профиля</summary><Facts items={inCategory('trade')}/></details>}
        </section>
        <section className="card co-section co-wide" data-company-section="group"><div className="co-section-heading"><h2>Группа компаний</h2>
          {overview.group_revenue != null && <span className="co-group-revenue">Совокупная выручка: <b>{money(overview.group_revenue)}</b></span>}</div>
          <Facts items={groupFacts.filter(f=>f.category==='group')} empty={overview.group_size===1 ? 'В выборке у этой группы одно юридическое лицо. Дополнительные сведения о группе отсутствуют.' : 'Сведения о группе отсутствуют.'}/>
          {!!groupFacts.filter(f=>f.category!=='group').length && <details className="co-more"><summary>Отрасли, география и другие связи группы</summary>
            <Facts items={groupFacts.filter(f=>f.category!=='group')}/></details>}
        </section>
      </div>
      {!!overview.data_notes.length && <details className="card co-section co-quality"><summary>Особенности исходных данных · {overview.data_notes.length}</summary>
        <ul className="co-facts">{overview.data_notes.map(note=><li key={note}>{note}</li>)}</ul></details>}
      <div className="co-footer"><p className="co-note">Уточнения для подбора новостей можно оставить в «Инсайтах» → «Профиль компании».</p>
        <button className="btn btn-gray" type="button" onClick={()=>window.showView('insights')}>Открыть инсайты</button></div>
    </>}
  </div>;
}
