// Our widgets inside the cabinet shell (index.html), client's point of view: one-line home block, «Инсайты» page
// (compact feed, company profile with HITL questions) and the large insight window.
// The entry chooses company/role and pins an analysis. The shell exposes window.demoContext;
// API feed previews and individual detail requests keep unopened cards out of the initial render.
import React, { lazy, memo, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { API_ENABLED, jobPreviews, latestJobs, startAnalysis, waitJob } from './api';
import './insights.css';
import AccountChooser from './accountChooser.jsx';
import { SESSION_KEY, validSession } from './cabinetMode.js';

let insightWindowModule;
const prepareInsightWindow = () => insightWindowModule ??= import('./insightWindow.jsx')
  .then(module => ({ default: module.InsightWindow }));
const InsightWindow = lazy(prepareInsightWindow);
const InsightPage = lazy(() => import('./insightPage.jsx'));

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
// Profile answers still use the demo handler; insight feedback is persisted by feedback_api.
const send = (kind, payload) => console.info(kind, payload);

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
    const cancelled = () => stop;
    const show = async job => {
      if (stop) return;
      const feed = await jobPreviews(job.id);
      if (!stop) setLive(l => ({ ...l, state: 'ready', feed, job: job.id, error: null }));
    };
    const follow = async job => {
      setLive(l => ({ ...l, state: 'running', error: null }));
      const done = await waitJob(job.id, cancelled);
      if (stop) return;
      if (['succeeded', 'partial'].includes(done.status)) await show(done);
      else setLive(l => ({ ...l, state: l.feed ? 'ready' : 'error', error: done.error || `Анализ: ${done.status}` }));
    };
    setLive({ state: 'loading' });
    (async () => {
      if (ctx.kind === 'cached' && !activeJob) { await show({ id:ctx.job_id }); return; }
      if (activeJob || ctx.job_id) { await follow({ id:activeJob || ctx.job_id }); return; }
      const { done, running } = await latestJobs(ctx.company, ctx.role);
      if (done) await show(done);
      else if (!stop) setLive({ state: 'none' });
      if (running && !stop) await follow(running);
    })().catch(e => !stop && setLive(l => ({ ...l, state: 'error', error: e.message })));
    // Follow the job created by the entry form or the explicit refresh action.
    return () => { stop = true; };
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
  const [live, run] = useLive(ctx);
  useEffect(() => setOpen(null), [ctx.company, ctx.role]);
  useEffect(() => {
    if (ctx.kind) { setData({ roles:[], sources:{} }); return; }
    fetch('feed.json').then(r => r.json()).then(setData);
  }, [ctx.kind]);

  const company = useMemo(() => ctx.company_meta ? { profile:{ name:ctx.company_meta.name, industry:ctx.company_meta.sector,
    scale:'Данные кейса', group:'—', bank_products:[], hypotheses:[], channels:[] } } : data?.companies?.[ctx.company], [ctx.company_meta, ctx.company, data]);
  const feed = live.feed || (ctx.kind ? undefined : company?.feeds[ctx.role]);
  const sources = live.feed ? live.feed.sources : data?.sources;
  const role = ctx.role_title || data?.roles.find(r => r.id === ctx.role)?.name;
  // API insights carry the role id
  const visible = useMemo(() => (feed?.insights || []).filter(i => !hidden.includes(i.id))
    .map(i => ({ ...i, roleLabel: i.role === ctx.role ? role : data?.roles.find(r => r.id === i.role)?.name || i.role })), [feed, hidden, data, ctx.role, role]);
  const hide = useCallback(id => { setHidden(h => [...h, id]); setOpen(null); }, []);
  const close = useCallback(() => setOpen(null), []);
  const restore = useCallback(() => setHidden([]), []);
  const answer = useCallback((hid, a) => {
    send('profile_answer', { company: ctx.company, hypothesis: hid, ...a });
    setAnswers(all => ({ ...all, [ctx.company]: { ...all[ctx.company], [hid]: a } }));
  }, [ctx.company]);
  if (!data) return null;
  const slot = id => document.getElementById(id);

  return <>
    {createPortal(<HomeWidget insights={visible} state={live.state} error={live.error} onOpen={setOpen} />, slot('insights-widget'))}
    {view === 'insights' && createPortal(<Suspense fallback={<p role="status">Загружаем ленту…</p>}>
      <InsightPage {...{ company, feed, role, live, run, visible, hidden, answers: answers[ctx.company] || EMPTY_ANSWERS, answer, onOpen: setOpen, onRestore: restore }} />
    </Suspense>, slot('insights-page'))}
    {open && <Suspense fallback={<div className="insight-opening" role="status">Открываем инсайт…</div>}>
      <InsightWindow key={`${ctx.company}:${ctx.role}:${open.job_id || 'demo'}:${open.id}`}
        insight={open} sources={open.sources || sources} company={ctx.company} onClose={close} onHide={hide} />
    </Suspense>}
  </>;
}

// home: one line with the top-1 insight, like the payment-statuses row next to it (criterion 08)
const HomeWidget = memo(function HomeWidget({ insights, state, error, onOpen }) {
  const top = insights[0];
  return <div className="card iw">
    <b className="iw-label">Инсайты</b>
    {top
      ? <button className="iw-top" onClick={() => onOpen(top)} title={top.title}>{top.title}</button>
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
  const [session, setSession] = useState(readSession);
  const [previous, setPrevious] = useState(session);
  useEffect(() => {
    const choose = () => {
      setSession(null);
      try { sessionStorage.removeItem(SESSION_KEY); } catch {}
    };
    window.addEventListener('cabinet:choose', choose);
    window.addEventListener('cabinet:logout', choose);
    return () => { window.removeEventListener('cabinet:choose', choose); window.removeEventListener('cabinet:logout', choose); };
  }, []);
  useEffect(() => {
    document.body.classList.toggle('cabinet-locked', !session);
  }, [session]);
  const enter = next => {
    setPrevious(next);
    try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(next)); } catch {}
    setSession(next);
  };
  return session ? <ActiveCabinet session={session} key={`${session.company_id}:${session.role}:${session.job_id}`} />
    : <AccountChooser onEnter={enter} previous={previous} />;
}

// Reuse the root across Vite updates instead of leaving multiple React trees/listeners alive.
const root = import.meta.hot?.data.root || createRoot(document.getElementById('insights-root'));
if (import.meta.hot) import.meta.hot.data.root = root;
root.render(<Root />);
