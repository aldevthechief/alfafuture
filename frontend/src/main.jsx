// Our widgets inside the cabinet shell (index.html), client's point of view: one-line home block, «Инсайты» page
// (compact feed, company profile with HITL questions) and the large insight window.
// The shell owns company/role (window.demoContext, event 'demo:context'). Insights come from the backend API (src/api.js):
// the latest finished analysis of the company and role; until there is one, the demo public/feed.json (also roles and profile).
import React, { lazy, memo, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { API_ENABLED, jobPreviews, latestJobs, startAnalysis, waitJob } from './api';
import './insights.css';

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
  const [tick, setTick] = useState(0);
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
      const { done, running } = await latestJobs(ctx.company, ctx.role);
      if (done) await show(done);
      else if (!stop) setLive({ state: 'none' });
      if (running && !stop) await follow(running);
    })().catch(e => !stop && setLive(l => ({ ...l, state: 'error', error: e.message })));
    // run() bumps tick after POST so this effect follows the new job
    return () => { stop = true; };
  }, [ctx.company, ctx.role, tick]);
  const run = useCallback(() => startAnalysis(ctx.company, ctx.role).then(() => setTick(t => t + 1))
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
  useEffect(() => { fetch('feed.json').then(r => r.json()).then(setData); }, []);

  const company = data?.companies[ctx.company];
  const feed = live.feed || company?.feeds[ctx.role];
  const sources = live.feed ? live.feed.sources : data?.sources;
  const role = data?.roles.find(r => r.id === ctx.role)?.name;
  // API insights carry the role id
  const visible = useMemo(() => (feed?.insights || []).filter(i => !hidden.includes(i.id))
    .map(i => ({ ...i, roleLabel: data?.roles.find(r => r.id === i.role)?.name || i.role })), [feed, hidden, data]);
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
    {createPortal(<HomeWidget insights={visible} onOpen={setOpen} />, slot('insights-widget'))}
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
const HomeWidget = memo(function HomeWidget({ insights, onOpen }) {
  const top = insights[0];
  return <div className="card iw">
    <b className="iw-label">Инсайты</b>
    {top
      ? <button className="iw-top" onClick={() => onOpen(top)} title={top.title}>{top.title}</button>
      : <span className="iw-top iw-none">Сейчас нет важных для вас событий</span>}
    <Button size={32} view="secondary" onClick={() => window.showView('insights')}>
      Больше инсайтов{insights.length > 1 ? ` · ${insights.length}` : ''}</Button>
  </div>;
});

// Reuse the root across Vite updates instead of leaving multiple React trees/listeners alive.
const root = import.meta.hot?.data.root || createRoot(document.getElementById('insights-root'));
if (import.meta.hot) import.meta.hot.data.root = root;
root.render(<App />);
