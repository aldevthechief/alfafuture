import React, { useEffect, useState } from 'react';
import { ALLOW_INFERENCE } from './cabinetMode.js';
import { cabinetAccounts, searchCompanies, startAnalysis } from './api.js';
import './accountChooser.css';

const MAIN_ROLES = [['ceo', 'Генеральный директор', 'CEO'], ['cfo', 'Финансовый директор', 'CFO'],
  ['marketing', 'Директор по маркетингу', 'Маркетинг']];
const Arrow = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>;
const Check = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m5 12 4 4 10-10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>;

export default function AccountChooser({ onEnter, previous }) {
  const [catalog, setCatalog] = useState(null);
  const [selected, setSelected] = useState(previous?.account_id || '');
  const [roleFilter, setRoleFilter] = useState('all');
  const [custom, setCustom] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState([]);
  const [searching, setSearching] = useState(false);
  const [company, setCompany] = useState(null);
  const [name, setName] = useState('');
  const [role, setRole] = useState('cfo');

  useEffect(() => {
    let active = true;
    setError('');
    cabinetAccounts().then(data => {
      if (!active) return;
      setCatalog(data);
      setSelected(id => data.accounts.some(account => account.id === id) ? id : data.accounts[0]?.id || '');
    }).catch(err => active && setError(err.message));
    return () => { active = false; };
  }, [retry]);

  useEffect(() => {
    if (!custom || company || query.trim().length < 2) { setMatches([]); setSearching(false); return; }
    let active = true;
    const timer = setTimeout(() => {
      setSearching(true);
      searchCompanies(query).then(data => { if (active) setMatches(data.items); })
        .catch(err => active && setError(err.message)).finally(() => active && setSearching(false));
    }, 250);
    return () => { active = false; clearTimeout(timer); };
  }, [custom, query, company]);

  const accounts = catalog?.accounts || [];
  const filtered = roleFilter === 'all' ? accounts : accounts.filter(account => account.role === roleFilter);
  const active = accounts.find(account => account.id === selected);
  const cachedRoles = [...new Map(accounts.map(account => [account.role, account.role_title])).entries()];

  function enterCached() {
    if (!active || busy) return;
    onEnter({ kind:'cached', account_id:active.id, company_id:active.company_id, role:active.role,
      role_title:active.role_title, job_id:active.job_id, company:active });
  }

  async function enterCustom(event) {
    event.preventDefault();
    if (!ALLOW_INFERENCE || !company || busy) return;
    setBusy(true); setError('');
    try {
      const job = await startAnalysis(company.id, role, { engine:'judge', source:'both',
        analytics_cache_only:false, analytics_refresh:true });
      onEnter({ kind:'live', company_id:company.id, role,
        role_title:catalog.roles.find(item => item.id === role)?.title || role, job_id:job.id,
        company:{ id:company.id, name:name.trim() || company.name, sector:company.industry, city:company.city,
          initials:(name.trim() || 'Компания').slice(0, 2).toUpperCase(), color:'blue' } });
    } catch (err) { setError(err.message); setBusy(false); }
  }

  return <main className="entry">
    <header className="entry-header">
      <div className="entry-brand"><span className="entry-logo" aria-hidden="true">А<span /></span>Альфа-Бизнес<span className="entry-brand-divider" /><span className="entry-brand-product">Инсайты</span></div>
      <span className="entry-demo-label"><span />Демо-кабинет</span>
    </header>
    <div className="entry-layout">
      <section className="entry-story" aria-labelledby="entry-story-title">
        <div className="entry-eyebrow"><span />Персональная аналитика для бизнеса</div>
        <h1 id="entry-story-title">Важное для бизнеса.<br /><span>С учётом вашей роли.</span></h1>
        <p>Новости становятся решениями, когда вы видите,<br className="entry-desktop-break" /> что они меняют именно для вашей компании.</p>
        <div className="entry-art" aria-hidden="true">
          <div className="entry-orbit entry-orbit-one" /><div className="entry-orbit entry-orbit-two" />
          <div className="entry-paper">
            <div className="entry-paper-top"><span className="entry-art-icon">↗</span><span>Пульс вашего бизнеса</span><span className="entry-art-dot" /></div>
            <div className="entry-art-pill">Есть позиция Банка</div>
            <h2>Одна новость.<br />Ваш следующий шаг.</h2>
            <div className="entry-chart"><i /><i /><i /><i /><i /><i /><i /></div>
            <div className="entry-art-line" /><div className="entry-art-line entry-art-short" />
          </div>
          <div className="entry-floating entry-floating-role"><span>◎</span><div><small>В фокусе</small><strong>Ваша компания и роль</strong></div></div>
          <div className="entry-floating entry-floating-source"><span>✓</span>С опорой на источники</div>
        </div>
        <div className="entry-benefits"><span>01 <b>Контекст компании</b></span><span>02 <b>Выводы для вашей роли</b></span><span>03 <b>Понятные действия</b></span></div>
      </section>

      <section className="entry-panel" aria-labelledby="entry-title">
        {custom ? <>
          <button className="entry-back" onClick={() => { setCustom(false); setError(''); }} disabled={busy}>← К готовым кабинетам</button>
          <h2 id="entry-title">Новая компания.<br />Ваши приоритеты.</h2>
          <p className="entry-intro">Выберите компанию из данных кейса и вашу должность. Подготовим персональный анализ.</p>
          <form onSubmit={enterCustom} className="entry-form">
            <label className="entry-field">Компания<input value={query} placeholder="ID компании или отрасль, например U_3263"
              disabled={busy} autoComplete="off" onChange={event => { setQuery(event.target.value); setCompany(null); setError(''); }} /></label>
            {searching && <p className="entry-search-note" role="status">Ищем компанию…</p>}
            {!company && matches.length > 0 && <div className="entry-matches" aria-label="Найденные компании">{matches.map(item =>
              <button key={item.id} type="button" onClick={() => { setCompany(item); setQuery(`${item.name} · ${item.city}`); setMatches([]); }}>
                <strong>{item.name}</strong><small>{item.industry} · {item.city}</small>
              </button>)}</div>}
            {!company && query.trim().length >= 2 && !searching && !matches.length && <p className="entry-search-note">Выберите компанию из результатов поиска. Доступны компании из данных кейса.</p>}
            {company && <div className="entry-company-selected"><Check /><span>{company.industry}</span></div>}
            <label className="entry-field">Название в кабинете <span className="entry-optional">необязательно</span><input value={name} maxLength={80} disabled={busy}
              placeholder="Например, Сибирская транспортная группа" onChange={event => setName(event.target.value)} /></label>
            <fieldset className="entry-roles"><legend>Ваша должность</legend>{MAIN_ROLES.map(([id, title, short]) =>
              <button type="button" key={id} className={role === id ? 'is-selected' : ''} aria-pressed={role === id}
                disabled={busy} title={title} onClick={() => setRole(id)}>{short}</button>)}</fieldset>
            <label className="entry-field entry-other-role">Или другая должность<select value={role} disabled={busy} onChange={event => setRole(event.target.value)}>
              {(catalog?.roles || []).map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
            </select></label>
            <div className="entry-analysis-note"><span>↗</span><p><strong>Запустим новый анализ</strong>Он займёт несколько минут. Вы сразу попадёте в кабинет и сможете следить за результатом.</p></div>
            {error && <p role="alert" className="entry-error">{error}</p>}
            <button className="entry-primary" type="submit" disabled={!company || busy}>{busy ? 'Создаём анализ…' : 'Войти и запустить анализ'}{!busy && <Arrow />}</button>
          </form>
        </> : <>
          <div className="entry-step">ВАШ РАБОЧИЙ КОНТЕКСТ</div>
          <h2 id="entry-title">Выберите кабинет</h2>
          <p className="entry-intro">Уточните компанию и должность —<br />остальное мы уже подготовили.</p>
          {catalog && <label className="entry-role-filter">Должность сотрудника<select value={roleFilter} onChange={event => {
            const value = event.target.value; setRoleFilter(value);
            const next = accounts.find(account => value === 'all' || account.role === value); if (next) setSelected(next.id);
          }}><option value="all">Все подготовленные должности</option>{cachedRoles.map(([id, title]) => <option key={id} value={id}>{title}</option>)}</select></label>}
          <div className="entry-accounts" role="group" aria-label="Готовые кабинеты">
            {!catalog && !error && <div className="entry-loading" role="status">Загружаем кабинеты…</div>}
            {filtered.map(account => <button key={account.id} className={`entry-account ${selected === account.id ? 'is-selected' : ''}`}
              aria-pressed={selected === account.id} onClick={() => setSelected(account.id)}>
              <span className={`entry-avatar entry-avatar-${account.color}`}>{account.initials}</span>
              <span className="entry-account-content"><strong>{account.name}</strong><span>{account.role_title}</span><small>{account.sector}<i />{account.insight_count} инсайтов</small></span>
              <span className="entry-account-check">{selected === account.id ? <Check /> : <Arrow />}</span>
            </button>)}
          </div>
          {error && <div className="entry-error" role="alert">Не удалось загрузить кабинеты. {error}<button onClick={() => setRetry(n => n + 1)}>Попробовать ещё раз</button></div>}
          {catalog && !accounts.length && <p className="entry-error">Готовые кабинеты пока недоступны. Проверьте наличие сохранённых анализов на сервере.</p>}
          <button className="entry-primary" onClick={enterCached} disabled={!active}>Продолжить<Arrow /></button>
          {ALLOW_INFERENCE && <button className="entry-custom" disabled={!catalog} onClick={() => { setCustom(true); setError(''); }}><span>＋</span><div><strong>Другая компания?</strong><small>Подготовим анализ под вашу задачу</small></div><Arrow /></button>}
          <p className="entry-footnote">Готовые кабинеты открываются без ожидания.<br />Названия компаний в демо условные.</p>
        </>}
      </section>
    </div>
    <footer className="entry-footer"><span>Аналитика Альфа-Банка + контекст вашего бизнеса</span><span>Сформировано AI. Решение — за вами.</span></footer>
  </main>;
}
