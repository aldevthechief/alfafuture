// Our widgets inside the cabinet shell (index.html), client's point of view: one-line home block, «Инсайты» page
// (feed, company profile with HITL questions) and the insight panel.
// The shell owns company/role (window.demoContext, event 'demo:context'). Insights come from the backend API (src/api.js):
// the latest finished analysis of the company and role; until there is one, the demo public/feed.json (also roles and profile).
import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { Status } from '@alfalab/core-components/status';
import { Tag } from '@alfalab/core-components/tag';
import { Tabs, Tab } from '@alfalab/core-components/tabs';
import { SidePanel } from '@alfalab/core-components/side-panel';
import { Typography } from '@alfalab/core-components/typography';
import { Input } from '@alfalab/core-components/input';
import { InsightDiscussion } from './discussion.jsx';
import { Plate } from '@alfalab/core-components/plate';
import { API_ENABLED, companyFacts, latestJobs, startAnalysis, toFeed, waitJob } from './api';
import './insights.css';

const CONFIDENCE = { high: ['green', 'Уверенность высокая'], medium: ['orange', 'Уверенность средняя'], low: ['grey', 'Уверенность низкая'] };
const TRACE = { plan: 'План', tool: 'Данные', check: 'Проверка' };
const valid = s => s && !isNaN(new Date(s));
const date = s => valid(s) ? new Date(s).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' }) : '';
const time = s => valid(s) ? new Date(s).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : '';
const Title = props => <Typography.Title font="system" {...props} />;
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
      const feed = toFeed(job, await companyFacts(ctx.company));
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
  const run = () => startAnalysis(ctx.company, ctx.role).then(() => setTick(t => t + 1))
    .catch(e => setLive(l => ({ ...l, state: 'error', error: e.message })));
  return [live, run];
}

function App() {
  const ctx = useCtx();
  const [data, setData] = useState(null);
  const [open, setOpen] = useState(null);
  const [hidden, setHidden] = useState([]);
  const [answers, setAnswers] = useState({}); // company id → hypothesis id → {verdict, text}
  const [live, run] = useLive(ctx);
  useEffect(() => setOpen(null), [ctx.company, ctx.role]);
  useEffect(() => { fetch('feed.json').then(r => r.json()).then(setData); }, []);
  if (!data) return null;

  const company = data.companies[ctx.company];
  const feed = live.feed || company?.feeds[ctx.role];
  const sources = live.feed ? live.feed.sources : data.sources;
  const role = data.roles.find(r => r.id === ctx.role)?.name;
  // API insights carry the role id
  const visible = (feed?.insights || []).filter(i => !hidden.includes(i.id))
    .map(i => ({ ...i, roleLabel: data.roles.find(r => r.id === i.role)?.name || i.role }));
  const hide = id => { setHidden(h => [...h, id]); setOpen(null); };
  const answer = (hid, a) => {
    send('profile_answer', { company: ctx.company, hypothesis: hid, ...a });
    setAnswers(all => ({ ...all, [ctx.company]: { ...all[ctx.company], [hid]: a } }));
  };
  const slot = id => document.getElementById(id);

  return <>
    {createPortal(<HomeWidget insights={visible} onOpen={setOpen} />, slot('insights-widget'))}
    {createPortal(<Page {...{ company, feed, role, live, run, visible, hidden, answers: answers[ctx.company] || {}, answer, onOpen: setOpen, onRestore: () => setHidden([]) }} />, slot('insights-page'))}
    <InsightPanel key={`${ctx.company}:${ctx.role}:${open?.job_id || 'demo'}:${open?.id || 'closed'}`}
      insight={open} sources={open?.sources || sources} company={ctx.company} onClose={() => setOpen(null)} onHide={hide} />
  </>;
}

// home: one line with the top-1 insight, like the payment-statuses row next to it (criterion 08)
function HomeWidget({ insights, onOpen }) {
  const top = insights[0];
  return <div className="card iw">
    <b className="iw-label">Инсайты</b>
    {top
      ? <button className="iw-top" onClick={() => onOpen(top)} title={top.title}>{top.title}</button>
      : <span className="iw-top iw-none">Сейчас нет важных для вас событий</span>}
    <Button size={32} view="secondary" onClick={() => window.showView('insights')}>
      Больше инсайтов{insights.length > 1 ? ` · ${insights.length}` : ''}</Button>
  </div>;
}

const LIVE = { loading: 'Загружаем анализ…', none: 'Анализа ещё нет, показан пример', running: 'Идёт анализ новостей и аналитики Банка…',
               error: 'Не удалось получить анализ, показан пример' };
function LiveBar({ live, run }) {
  if (live.state === 'off') return null;
  const errors = live.feed?.errors || [];
  return <div className="ip-live">
    {live.state !== 'ready' && <Typography.Text view="primary-small" color="secondary">{LIVE[live.state]}</Typography.Text>}
    {live.state === 'ready' && errors.length > 0 && <Status color="orange" size={20}>Часть источников не обработана</Status>}
    {live.error && <Typography.Text view="primary-small" color="negative">{live.error}</Typography.Text>}
    <Button size={32} view="secondary" loading={live.state === 'running'} disabled={live.state === 'loading'} onClick={run}>
      {live.feed ? 'Обновить анализ' : 'Запустить анализ'}</Button>
  </div>;
}

function Page({ company, feed, role, live, run, visible, hidden, answers, answer, onOpen, onRestore }) {
  const [tab, setTab] = useState('feed');
  const [filter, setFilter] = useState('all');
  const channels = [...new Set(visible.map(i => i.channel))];
  const shown = visible.filter(i => filter === 'all' || (filter === 'risks' ? i.risks.length : filter === 'opps' ? i.opportunities.length : i.channel === filter));
  const questions = company ? company.profile.hypotheses.filter(h => !answers[h.id]).length : 0;
  return <div className="ip">
    <h1 className="h-sys">Инсайты</h1>
    <p className="h-sub">Что в новостях важно для {company?.profile.name} · {role}{feed && ` · обновлено ${date(feed.computed_at)} в ${time(feed.computed_at)}`}</p>
    <LiveBar live={live} run={run} />
    <Tabs selectedId={tab} onChange={(_, { selectedId }) => setTab(selectedId)} size="s">
      <Tab id="feed" title="События для вас">
        <div className="ip-filters">
          {[['all', 'Все'], ['risks', 'Риски'], ['opps', 'Возможности'], ...channels.map(c => [c, c])].map(([id, t]) =>
            <Tag key={id} size={32} checked={filter === id} onClick={() => setFilter(id)}>{t}</Tag>)}
        </div>
        {shown.map(i => <InsightCard key={i.id} insight={i} onOpen={onOpen} />)}
        {!shown.length && <p className="ip-empty">Сейчас нет событий, важных для вашей роли.</p>}
        {hidden.length > 0 && <Button view="text" size={32} onClick={onRestore}>Показать скрытые ({hidden.length})</Button>}
      </Tab>
      <Tab id="profile" title="Профиль компании" rightAddons={questions > 0 && <Status color="blue" size={20}>{questions}</Status>}>
        <Profile profile={company?.profile} answers={answers} onAnswer={answer} />
      </Tab>
    </Tabs>
    <p className="ip-disclaimer">Сформировано AI на основе открытых новостей и аналитики Альфа-Банка. Не является инвестиционной рекомендацией.</p>
  </div>;
}

function InsightCard({ insight: i, onOpen }) {
  const [color, conf] = CONFIDENCE[i.confidence];
  return <button className="card ic" onClick={() => onOpen(i)}>
    <div className="ic-tags">
      <Status color="blue" size={20}>{i.channel}</Status>
      <Status color={color} size={20}>{conf}</Status>
      {i.bank_view.length > 0 && <Status color="purple" size={20}>Есть позиция Банка</Status>}
      <span className="ic-date">{date(i.published)}</span>
    </div>
    <Title tag="h3" view="small" weight="bold">{i.title}</Title>
    <Typography.Text view="primary-medium" color="secondary" tag="p">{i.why_it_matters[0]?.text}</Typography.Text>
    {i.why_now && <Typography.Text view="primary-small" weight="medium" tag="p">Почему сейчас: {i.why_now}</Typography.Text>}
  </button>;
}

// layers are visually separate (criterion 02): facts / bank view / AI conclusions / hypotheses
function Layer({ kind, label, claims, sources }) {
  if (!claims?.length) return null;
  return <section className={`layer layer-${kind}`}>
    <Typography.Text view="caps" weight="bold" tag="div">{label}</Typography.Text>
    <ul>{claims.map((c, n) => <li key={n}>{c.text}<SourceRefs ids={c.sources} sources={sources} /></li>)}</ul>
  </section>;
}

function SourceRefs({ ids, sources }) {
  return <span className="refs">{ids.map(id => {
    const s = sources[id];
    const label = s?.kind === 'fact' ? 'данные компании' : s ? `${s.outlet || s.title}${s.page ? `, с. ${s.page}` : ''}` : id;
    return <a key={id} href={`#src-${id}`} className={`ref ref-${s?.kind}`}>{label}</a>;
  })}</span>;
}

function InsightPanel({ insight: i, sources, onClose, onHide, company }) {
  const [called, setCalled] = useState(false);
  const used = i ? [...new Set([i.what_happened, i.why_it_matters, i.risks, i.opportunities, i.hypotheses, i.bank_view].flat().flatMap(c => c.sources))] : [];

  return <SidePanel open={!!i} onClose={onClose} size={500} hasCloser>
    {i && <>
      <SidePanel.Header title={i.roleLabel || i.role} hasCloser />
      <SidePanel.Content>
        <div className="panel">
          <Title tag="h2" view="small" weight="bold">{i.title}</Title>
          {i.why_now && <Typography.Text view="primary-small" color="secondary" tag="p">Почему сейчас: {i.why_now}</Typography.Text>}

          <Layer kind="fact" label="Факты · что произошло" claims={i.what_happened} sources={sources} />
          <Layer kind="ai" label="Вывод AI · почему это важно вам" claims={i.why_it_matters} sources={sources} />
          <Layer kind="ai" label="Риски" claims={i.risks} sources={sources} />
          <Layer kind="ai" label="Возможности" claims={i.opportunities} sources={sources} />
          <Layer kind="hyp" label="Гипотезы · требуют проверки" claims={i.hypotheses} sources={sources} />
          {i.bank_view.length
            ? <Layer kind="bank" label="Позиция Банка" claims={i.bank_view} sources={sources} />
            : <Plate view="attention" title="Аналитики Банка по этой теме нет">Вывод не является официальной позицией Банка.</Plate>}

          <section className="layer">
            <Typography.Text view="caps" weight="bold" tag="div">Что сделать</Typography.Text>
            <ol>{i.actions.map(a => <li key={a}>{a}</li>)}</ol>
            <Typography.Text view="primary-medium" tag="p"><b>Итог AI:</b> {i.ai_conclusion}</Typography.Text>
          </section>

          {i.bank_product && <Plate view="common" title="Чем может помочь Банк"
            buttons={[<Button key="m" size={32} view="secondary" disabled={called} onClick={() => setCalled(true)}>
              {called ? 'Менеджер свяжется с вами' : 'Связаться с менеджером'}</Button>]}>
            {i.bank_product}</Plate>}

          <InsightDiscussion company={company} insight={i} />

          <section className="layer">
            <Typography.Text view="caps" weight="bold" tag="div">Источники</Typography.Text>
            <ul className="src">{used.map(id => {
              const s = sources[id];
              if (!s) return <li key={id} id={`src-${id}`}>{id}</li>;
              return <li key={id} id={`src-${id}`}>
                {s.kind === 'fact' ? <><b>Данные компании:</b> {s.text}</>
                  : <><b>{s.kind === 'bank' ? `Альфа-Банк, «${s.title}»` : s.outlet}</b>{s.date && `, ${date(s.date)}`}{s.page && `, с. ${s.page}`}
                    {s.kind === 'news' && <> — {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}</>}
                    <br /><span className="snip">{s.snippet}</span></>}
              </li>;
            })}</ul>
          </section>

          {i.trace.length > 0 && <details className="trace">
            <summary>Как мы пришли к выводу</summary>
            <ol>{i.trace.map((t, n) => <li key={n} className={`t-${t.status || t.kind}`}><span>{TRACE[t.kind]}</span>{t.text}</li>)}</ol>
          </details>}

          <p className="ip-disclaimer">Не является инвестиционной рекомендацией.</p>
        </div>
      </SidePanel.Content>
      <SidePanel.Footer>
        <div className="fb">
          <div className="fb-btns">
            <Button size={40} view="text" onClick={() => onHide(i.id)}>Скрыть</Button>
          </div>
        </div>
      </SidePanel.Footer>
    </>}
  </SidePanel>;
}

// what the bank knows about the company + HITL: agent hypotheses the data could not settle become optional
// «Верно ли, что …?» questions; answers make the exposure profile more precise
const VERDICT = { yes: 'да', no: 'нет' };
function Profile({ profile: p, answers, onAnswer }) {
  const [later, setLater] = useState(false);
  const [other, setOther] = useState(null); // text while «Другое» is open
  if (!p) return null;
  const rest = p.hypotheses.filter(h => !answers[h.id]);
  const h = rest[0];
  const answered = p.hypotheses.filter(x => answers[x.id] && answers[x.id].verdict !== 'skip');
  const save = a => { onAnswer(h.id, a); setOther(null); };
  const rows = [['Отрасль', p.industry], ['Масштаб', p.scale], ['Группа', p.group], ['Продукты Банка', p.bank_products.join(', ')]];

  return <>
    {h && !later && <div className="card hitl">
      <div className="hitl-head">
        <Typography.Text view="primary-small" color="secondary">
          Помогите точнее подбирать события · вопрос {p.hypotheses.length - rest.length + 1} из {p.hypotheses.length}</Typography.Text>
        <Button view="text" size={32} onClick={() => setLater(true)}>Не сейчас</Button>
      </div>
      <Title tag="h3" view="small" weight="bold">{h.question}</Title>
      <Typography.Text view="primary-small" color="secondary" tag="p">Почему спрашиваем: {h.for}. {h.impact}</Typography.Text>
      {other === null
        ? <div className="iw-btns">
          <Button size={40} view="secondary" onClick={() => save({ verdict: 'yes' })}>Да</Button>
          <Button size={40} view="secondary" onClick={() => save({ verdict: 'no' })}>Нет</Button>
          <Button size={40} view="secondary" onClick={() => setOther('')}>Другое</Button>
          <Button size={40} view="text" onClick={() => save({ verdict: 'skip' })}>Пропустить</Button>
        </div>
        : <div className="iw-btns">
          <Input className="hitl-input" size={48} autoFocus placeholder="Как на самом деле?" value={other}
            onChange={e => setOther(e.target.value)} onKeyDown={e => e.key === 'Enter' && other.trim() && save({ verdict: 'other', text: other.trim() })} />
          <Button size={48} view="primary" disabled={!other.trim()} onClick={() => save({ verdict: 'other', text: other.trim() })}>Сохранить</Button>
          <Button size={48} view="text" onClick={() => setOther(null)}>Отмена</Button>
        </div>}
    </div>}

    <div className="card prof">
      <Title tag="h3" view="xsmall" weight="bold">Как мы видим вашу компанию</Title>
      <dl>{rows.map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>)}</dl>
      <Title tag="h3" view="xsmall" weight="bold">Через что новости влияют на вас</Title>
      <ul>{p.channels.map(c => <li key={c.name}><b>{c.name}</b> — {c.why}</li>)}</ul>
      {answered.length > 0 && <>
        <Title tag="h3" view="xsmall" weight="bold">Ваши уточнения</Title>
        <ul className="answers">{answered.map(x => {
          const a = answers[x.id];
          return <li key={x.id}>{x.text}: <b>{VERDICT[a.verdict] || a.text}</b>
            <Button view="text" size={32} onClick={() => { onAnswer(x.id, undefined); setLater(false); }}>Изменить</Button></li>;
        })}</ul>
      </>}
      {later && rest.length > 0 && <Button view="text" size={32} onClick={() => setLater(false)}>Ответить на вопросы о компании ({rest.length})</Button>}
    </div>
  </>;
}

createRoot(document.getElementById('insights-root')).render(<App />);
