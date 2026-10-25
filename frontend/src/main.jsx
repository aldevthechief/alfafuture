// Our widgets inside the cabinet shell (index.html), client's point of view: one-line home block, «Инсайты» page
// (compact feed, company profile with HITL questions) and the large insight window.
// The entry chooses company/role; the feed merges saved and newly completed analyses for that reader.
// API feed previews and individual detail requests keep unopened cards out of the initial render.
import React, { lazy, memo, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { API_ENABLED, cabinetAccounts, cabinetFeed, startAnalysis } from './api';
import './insights.css';
import AccountChooser from './accountChooser.jsx';
import { SESSION_KEY, validSession } from './cabinetMode.js';
import { insightSignal } from './insightSignal.js';
import { applyTheme } from './theme.js';
import { ThemeToggle } from './themeToggle.jsx';
import './theme.css';
import { compactKey, useCompactFeed } from './compactFeed.js';
import { initialSession, sessionAccount } from './accountSession.js';
import { CompanySwitcher } from './companySwitcher.jsx';
import { useCompanyProfile } from './companyProfile.js';

let insightWindowModule;
const prepareInsightWindow = () => insightWindowModule ??= import('./insightWindow.jsx')
  .then(module => ({ default: module.InsightWindow }));
const InsightWindow = lazy(prepareInsightWindow);
const InsightPage = lazy(() => import('./insightPage.jsx'));
const CompanyChat = lazy(() => import('./companyChat.jsx'));

function useView() {
  const [view, setView] = useState(() => document.querySelector('.view.active')?.id.replace('view-', '') || 'home');
  useEffect(() => {
    const onView = event => setView(event.detail.view);
    window.addEventListener('demo:view', onView);
    return () => window.removeEventListener('demo:view', onView);
  }, []);
  return view;
}
const EMPTY_ANSWERS = {};

function useCtx() {
  const [ctx, setCtx] = useState(window.demoContext);
  useEffect(() => {
    const on = e => setCtx({ ...e.detail });
    window.addEventListener('demo:context', on);
    return () => window.removeEventListener('demo:context', on);
  }, []);
  return ctx;
}

// API state for the selected company and role: off | loading | none (no analysis yet) | running | ready | error
function useLive(ctx) {
  const [live, setLive] = useState({ state: API_ENABLED ? 'loading' : 'off' });
  const [activeJob, setActiveJob] = useState(null);
  useEffect(() => {
    if (!API_ENABLED) return;
    let stop = false;
    let timer;
    setLive({ state: 'loading' });
    const refresh = async () => {
      try {
        const feed = await cabinetFeed(ctx.company, ctx.role);
        if (stop) return;
        const latest = feed.latest_job;
        const requested = activeJob || (ctx.kind === 'live' ? ctx.job_id : null);
        const failed = latest?.id === requested && ['failed', 'interrupted'].includes(latest.status);
        setLive({ state:feed.running_job ? 'running' : feed.insights.length ? 'ready' : failed ? 'error' : 'none',
          feed, error:failed ? latest.error || `Анализ: ${latest.status}` : null });
      } catch (error) {
        if (!stop) setLive(l => ({ ...l, state:l.feed?.insights.length ? 'ready' : 'error', error:error.message }));
      } finally {
        if (!stop) timer = setTimeout(refresh, 5000);
      }
    };
    refresh();
    // Refresh only small previews; full evidence is still fetched only when a card opens.
    return () => { stop = true; clearTimeout(timer); };
  }, [ctx.company, ctx.role, ctx.job_id, ctx.kind, activeJob]);
  const run = useCallback(() => startAnalysis(ctx.company, ctx.role).then(job => setActiveJob(job.id))
    .catch(e => setLive(l => ({ ...l, state: 'error', error: e.message }))), [ctx.company, ctx.role]);
  return [live, run];
}

function App() {
  const ctx = useCtx();
  const view = useView();
  const [data, setData] = useState(null);
  const [open, setOpen] = useState(null);
  const [hidden, setHidden] = useState([]);
  const [answers, setAnswers] = useState({}); // company id → hypothesis id → {verdict, text}
  const [bankChat, setBankChat] = useState({ mounted:false, open:false });
  useEffect(() => {
    const toggle = () => setBankChat(value => ({ mounted:true, open:!value.open }));
    window.addEventListener('cabinet:chat', toggle);
    return () => window.removeEventListener('cabinet:chat', toggle);
  }, []);
  useEffect(() => {
    document.getElementById('fab')?.setAttribute('aria-expanded', String(bankChat.open));
    return () => document.getElementById('fab')?.setAttribute('aria-expanded', 'false');
  }, [bankChat.open]);
  const [live, run] = useLive(ctx);
  const [profileState, saveProfileAnswer] = useCompanyProfile(ctx, live.feed, view === 'insights' && API_ENABLED && !!ctx.company_meta);
  useEffect(() => setOpen(null), [ctx.company, ctx.role]);
  useEffect(() => {
    if (ctx.kind) { setData({ roles:[], sources:{} }); return; }
    fetch('feed.json').then(r => r.json()).then(setData);
  }, [ctx.kind]);

  const company = useMemo(() => ctx.company_meta ? { profile:{ industry:ctx.company_meta.sector,
    ...profileState.profile, name:ctx.company_meta.name,
    bank_products:profileState.profile?.bank_products || [], hypotheses:profileState.questions,
    channels:profileState.profile?.channels || [] } } : data?.companies?.[ctx.company], [ctx.company_meta, ctx.company, data, profileState.profile, profileState.questions]);
  const feed = live.feed || (ctx.kind ? undefined : company?.feeds[ctx.role]);
  const sources = live.feed ? live.feed.sources : data?.sources;
  const role = ctx.role_title || data?.roles.find(r => r.id === ctx.role)?.name;
  const compactVersions = useCompactFeed(feed?.insights || []);
  // API insights carry the role id
  const visible = useMemo(() => (feed?.insights || []).filter(i => !hidden.includes(i.id))
    .map(i => ({ ...i, compact: compactVersions[compactKey(i)], roleLabel: i.role === ctx.role ? role : data?.roles.find(r => r.id === i.role)?.name || i.role })), [feed, hidden, data, ctx.role, role, compactVersions]);
  const hide = useCallback(id => { setHidden(h => [...h, id]); setOpen(null); }, []);
  const close = useCallback(() => setOpen(null), []);
  const restore = useCallback(() => setHidden([]), []);
  const answer = useCallback((hid, a) => {
    if (ctx.company_meta) return saveProfileAnswer(hid, a);
    setAnswers(all => ({ ...all, [ctx.company]: { ...all[ctx.company], [hid]: a } }));
    return Promise.resolve(true);
  }, [ctx.company, ctx.company_meta, saveProfileAnswer]);
  if (!data) return null;
  const slot = id => document.getElementById(id);

  return <>
    {bankChat.mounted && <Suspense fallback={<div role="status" className="insight-opening">Открываем чат…</div>}>
      <CompanyChat company={ctx.company} role={ctx.role} name={ctx.company_meta?.name || company?.profile.name}
        roleTitle={role} open={bankChat.open} onClose={() => setBankChat(value => ({ ...value,open:false }))} />
    </Suspense>}
    {createPortal(<HomeWidget insights={visible} state={live.state} error={live.error} onOpen={setOpen} />, slot('insights-widget'))}
    {view === 'insights' && createPortal(<Suspense fallback={<p role="status">Загружаем ленту…</p>}>
      <InsightPage {...{ company, feed, role, live, run, visible, hidden, profileState:ctx.company_meta ? profileState : undefined,
        answers:ctx.company_meta ? profileState.answers : answers[ctx.company] || EMPTY_ANSWERS, answer, onOpen: setOpen, onRestore: restore }} />
    </Suspense>, slot('insights-page'))}
    {open && <Suspense fallback={<div className="insight-opening" role="status">Открываем инсайт…</div>}>
      <InsightWindow key={`${ctx.company}:${ctx.role}:${open.job_id || 'demo'}:${open.id}`}
        insight={open} compact={compactVersions[compactKey(open)]} sources={open.sources || sources} company={ctx.company} onClose={close} onHide={hide} />
    </Suspense>}
  </>;
}

// home: one line with the top-1 insight, like the payment-statuses row next to it (criterion 08)
const HomeWidget = memo(function HomeWidget({ insights, state, error, onOpen }) {
  const top = insights[0];
  const signal = top && insightSignal(top);
  const title = top?.compact?.title || top?.title;
  return <div className={`card iw${signal ? ` insight-tone-${signal.tone}` : ''}`}>
    <b className="iw-label">Инсайты</b>
    {top
      ? <button className="iw-top" onClick={() => onOpen(top)} title={`${signal.label}. ${signal.reason}${signal.high ? ' Высокое влияние.' : ''} ${title}`}>
        <span role="img" aria-label={signal.label}>{signal.emoji}</span>{' '}
        {signal.high && <><span role="img" aria-label="Высокое влияние">🔥</span>{' '}</>}{title}
      </button>
      : <span className="iw-top iw-none" role="status">{state === 'running' ? 'Готовим персональные инсайты…'
        : state === 'loading' ? 'Загружаем инсайты…' : error || 'Сейчас нет важных для вас событий'}</span>}
    <Button size={32} view="secondary" onClick={() => window.showView('insights')}>
      Больше инсайтов{insights.length > 1 ? ` · ${insights.length}` : ''}</Button>
  </div>;
});

function readSession() {
  try { const session = JSON.parse(sessionStorage.getItem(SESSION_KEY)); return validSession(session) ? session : null; }
  catch { return null; }
}

function ActiveCabinet({ session }) {
  const [ready, setReady] = useState(false);
  useLayoutEffect(() => {
    window.setCabinetContext(session);
    setReady(true);
  }, [session]);
  return ready ? <App /> : null;
}

function Root() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  useLayoutEffect(() => applyTheme(theme), [theme]);
  const [session, setSession] = useState(null);
  const [recent, setRecent] = useState(() => {
    try { const items = JSON.parse(sessionStorage.getItem(`${SESSION_KEY}:accounts`) || '[]'); return Array.isArray(items) ? items.filter(validSession) : []; }
    catch { return []; }
  });
  const [previous, setPrevious] = useState(readSession);
  const [catalog, setCatalog] = useState(null);
  const [choosing, setChoosing] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [bootError, setBootError] = useState('');
  const [retry, setRetry] = useState(0);
  const closeMenu = useCallback(() => setMenuOpen(false), []);
  const enter = useCallback(next => {
    setPrevious(next); setSession(next); setChoosing(false); setMenuOpen(false);
    setRecent(items => [next, ...items.filter(item => item.company_id !== next.company_id || item.role !== next.role)].slice(0, 12));
    try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(next)); } catch {}
  }, []);
  useEffect(() => { try { sessionStorage.setItem(`${SESSION_KEY}:accounts`, JSON.stringify(recent)); } catch {} }, [recent]);
  const accountCatalog = useMemo(() => {
    if (!catalog) return null;
    const accounts = [...catalog.accounts];
    for (const saved of recent) if (!accounts.some(account => account.company_id === saved.company_id && account.role === saved.role)) accounts.push(sessionAccount(saved));
    return { ...catalog, accounts };
  }, [catalog, recent]);
  useEffect(() => {
    let active = true;
    setBootError('');
    cabinetAccounts().then(data => {
      if (!active) return;
      setCatalog(data);
      const next = initialSession(readSession(), data.accounts);
      if (next) enter(next); else setChoosing(true);
    }).catch(error => { if (active) setBootError(error.message); });
    return () => { active = false; };
  }, [retry, enter]);
  useEffect(() => {
    const choose = () => {
      setChoosing(true); setMenuOpen(false);
    };
    const logout = () => { choose(); setSession(null); try { sessionStorage.removeItem(SESSION_KEY); } catch {} };
    const toggle = () => setMenuOpen(value => !value);
    window.addEventListener('cabinet:choose', choose);
    window.addEventListener('cabinet:logout', logout);
    window.addEventListener('cabinet:switcher', toggle);
    return () => { window.removeEventListener('cabinet:choose', choose); window.removeEventListener('cabinet:logout', logout); window.removeEventListener('cabinet:switcher', toggle); };
  }, []);
  useEffect(() => {
    document.body.classList.toggle('cabinet-locked', !session || choosing);
  }, [session, choosing]);
  const themeToggle = <ThemeToggle theme={theme} onToggle={() => setTheme(value => value === 'dark' ? 'light' : 'dark')} />;
  if (!catalog && !session) return <div className="cabinet-bootstrap" role="status">
    {bootError ? <><p>Не удалось открыть кабинет: {bootError}</p><button onClick={() => setRetry(n => n + 1)}>Повторить</button></> : <p>Открываем ваш кабинет…</p>}
  </div>;
  return session && !choosing ? <>
    {createPortal(themeToggle, document.getElementById('theme-toggle'))}
    {menuOpen && createPortal(<CompanySwitcher accounts={accountCatalog.accounts} session={session} onEnter={enter}
      onManage={() => { setChoosing(true); setMenuOpen(false); }} onClose={closeMenu} />, document.getElementById('company-switcher'))}
    <ActiveCabinet session={session} key={`${session.company_id}:${session.role}:${session.job_id}`} />
  </> : <AccountChooser onEnter={enter} previous={previous} themeToggle={themeToggle} initialCatalog={accountCatalog}
    onBack={session ? () => setChoosing(false) : undefined} />;
}

// Reuse the root across Vite updates instead of leaving multiple React trees/listeners alive.
const root = import.meta.hot?.data.root || createRoot(document.getElementById('insights-root'));
if (import.meta.hot) import.meta.hot.data.root = root;
root.render(<Root />);
