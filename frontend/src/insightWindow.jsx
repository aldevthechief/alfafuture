// A wide insight workspace with separate reading and discussion tabs.
import React, { memo, useEffect, useId, useMemo, useState } from 'react';
import { ModalDesktop as Modal } from '@alfalab/core-components/modal/desktop';
import { Button } from '@alfalab/core-components/button';
import { Status } from '@alfalab/core-components/status';
import { Plate } from '@alfalab/core-components/plate';
import { Typography } from '@alfalab/core-components/typography';
import { InsightDiscussion } from './discussion.jsx';
import { loadInsight } from './api.js';

import { CONFIDENCE, day, view } from './insightView.js';
import { insightSignal } from './insightSignal.js';
import { SignalBadge } from './insightSignal.jsx';
import { InsightIcon } from './insightIcons.jsx';
import { InsightTabs } from './insightTabs.jsx';
import { InsightReader } from './insightReader.jsx';
import { ChatSplit } from './chatSplit.jsx';
import { FEEDBACK_ENABLED } from './feedback.js';
import './insightWorkspace.css';

const NO_TRANSITION = { enter: false, exit: false, timeout: 0 };

// one source inline: outlet and date
function NewsRef({ id, sources }) {
  const s = sources[id];
  if (!s) return null;
  const name = s.kind === 'bank' ? 'Аналитика Альфа-Банка' : s.kind === 'fact' ? 'Данные компании' : [s.outlet, day(s.date)].filter(Boolean).join(', ');
  return <span className="nref">{name}</span>;
}

// «подробнее» on a short fact: the full fact of the insight and the source quotes
function More({ full, quotes }) {
  if (!full && !quotes.length) return null;
  return <span className="insight-more" tabIndex={0}>подробнее<span className="insight-pop">
    {full && <span className="insight-pop-full">{full}</span>}
    {quotes.map(q => <span key={q} className="insight-pop-quote">«{q}»</span>)}
  </span></span>;
}

const refs = (line, sources, kinds) => [...new Map((line.sources || []).filter(id => kinds.includes(sources[id]?.kind))
  .map(id => [sources[id].kind === 'news' ? id : sources[id].kind, id])).values()];

function Section({ kind, label, children }) {
  return <section className={`layer layer-${kind}`}>
    <Typography.Text view="caps" weight="bold" tag="div">{label}</Typography.Text>
    {children}
  </section>;
}

function Labeled({ items }) {
  return <ul className="lab">{items.map((x, n) => <li key={n}>{x.label && <b>{x.label}. </b>}{x.text}</li>)}</ul>;
}

// news by title; company data and the bank's reports merged into one line each
const Sources = memo(function Sources({ ids, sources }) {
  const news = ids.filter(id => sources[id]?.kind === 'news');
  const facts = ids.some(id => sources[id]?.kind === 'fact');
  const bank = ids.some(id => sources[id]?.kind === 'bank');
  return <div className="side-card">
    <Typography.Text view="caps" weight="bold" tag="div">Источники</Typography.Text>
    <ul className="src2">
      {news.map(id => {
        const s = sources[id];
        return <li key={id}>
          {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title || s.outlet}</a> : s.title || s.outlet}
          <small> · {[s.outlet, day(s.date)].filter(Boolean).join(', ')}</small>
          {s.snippet && <span className="insight-more" tabIndex={0}> подробнее<span className="insight-pop">«{s.snippet}»</span></span>}
        </li>;
      })}
      {facts && <li>Данные компании <small>· профиль клиента в Альфа-Банке</small></li>}
      {bank && <li>Внутренние отчёты Альфа-Банка</li>}
    </ul>
  </div>;
});

export function InsightWindow({ insight: i, compact, sources = {}, company, onClose, onHide }) {
  const id = useId();
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState(i.preview ? null : i);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setError('');
    loadInsight(i, company).then(card => { if (active) setDetail(card); })
      .catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [i, company, retry]);
  const title = compact?.title || detail?.title || i.title;
  // Keep one portal, backdrop and focus lock from loading through the complete workspace.
  return <Modal open onClose={onClose} size={1140} transitionProps={NO_TRANSITION}
    className={`insight-workbench${expanded ? ' workbench-expanded' : ''}`} wrapperClassName={`workbench-overlay${expanded ? ' is-expanded' : ''}`}
    contentProps={{ className:'workbench-frame' }} wrapperProps={{ 'aria-modal':true, 'aria-labelledby':`${id}-title` }}>
    <header className="workbench-header">
      <div className="workbench-heading">
        <div className="workbench-eyebrow"><span className="workbench-mark"><InsightIcon name="sparkle" size={14} /></span>
          Персональный инсайт<span>·</span>{i.channel}<span>·</span>{day(i.published)}</div>
        <h2 id={`${id}-title`} title={title}>{title}</h2>
      </div>
      <div className="workbench-controls">
        <button className="workbench-icon-button workbench-expand" type="button" onClick={() => setExpanded(value => !value)}
          aria-label={expanded ? 'Уменьшить окно' : 'Развернуть окно'} title={expanded ? 'Уменьшить окно' : 'Развернуть окно'}>
          <InsightIcon name={expanded ? 'collapse' : 'expand'} /></button>
        <button className="workbench-icon-button" type="button" onClick={onClose} aria-label="Закрыть инсайт" title="Закрыть · Esc">
          <InsightIcon name="close" /></button>
      </div>
    </header>
    {detail ? <InsightDetails id={id} insight={detail} compact={compact} sources={detail.sources || sources}
      company={company} onHide={onHide} /> : <div className="workbench-loading" aria-busy={!error}>
      {error ? <><p role="alert">Не удалось загрузить инсайт: {error}</p>
        <Button size={40} view="secondary" onClick={() => setRetry(n => n + 1)}>Повторить</Button></>
        : <p role="status">Загружаем материалы инсайта…</p>}
    </div>}
  </Modal>;
}

function InsightDetails({ id, insight: i, compact, sources, company, onHide }) {
  const [tab, setTab] = useState('overview');
  const [readerOpen, setReaderOpen] = useState(true);
  const [selectedQuote, setSelectedQuote] = useState(null);
  const changeTab = next => {
    setTab(next);
    requestAnimationFrame(() => document.getElementById(`${id}-${next}-tab`)?.focus());
  };
  const [feedbackTarget, setFeedbackTarget] = useState(null);
  const v = useMemo(() => view(i, compact), [i, compact]);
  const [color, conf] = CONFIDENCE[i.confidence] || CONFIDENCE.low;
  const used = useMemo(() => [...new Set([...v.facts, ...v.why, ...v.risks, ...v.opportunities, ...v.bank_view, ...(i.hypotheses || [])].flatMap(x => x.sources || []))], [v, i]);

  return <>
    <InsightTabs id={id} tab={tab} onChange={changeTab} />
    <div className="workbench-panels">
      <section className="workbench-panel workbench-overview" id={`${id}-overview-panel`} role="tabpanel"
        aria-labelledby={`${id}-overview-tab`} hidden={tab !== 'overview'} tabIndex={0}>
        <div className={`iwin insight-detail-tone-${insightSignal(i).tone}`}>
          <div className="ic-tags">
            <SignalBadge insight={i} />
            <Status color={color} size={20}>{conf}</Status>
            {i.bank_view.length > 0 && <Status color="purple" size={20}>Есть позиция Банка</Status>}
          </div>
          <Typography.Text view="primary-large" color="secondary" tag="p">{v.lead}</Typography.Text>
          <div className="iwin-grid">
            <InsightBody insight={i} v={v} compact={compact} sources={sources} />
            <aside className="iwin-side">
              <div className="agent-invitation">
                <span className="agent-avatar"><InsightIcon name="sparkle" size={24} /></span>
                <span className="agent-eyebrow">ОТ НОВОСТИ К РЕШЕНИЮ</span>
                <h3>Разберём, что это меняет для вас</h3>
                <p>Обсудите сценарии, уточните расчёты и определите следующие шаги вместе с агентом.</p>
                <button type="button" onClick={() => changeTab('chat')}>Обсудить с агентом<InsightIcon name="arrow" size={18} /></button>
              </div>
              <Sources ids={used} sources={sources} />
            </aside>
          </div>
          <p className="iwin-disc">Не является инвестиционной рекомендацией.{!i.bank_view.length && ' Вывод не является официальной позицией Банка.'}</p>
        </div>
      </section>
      <section className="workbench-panel workbench-chat" id={`${id}-chat-panel`} role="tabpanel"
        aria-labelledby={`${id}-chat-tab`} hidden={tab !== 'chat'}>
        {/* Keep mounted: changing tabs must preserve drafts, scroll and an answer in flight. */}
        <ChatSplit reading={readerOpen}>
          <InsightDiscussion company={company} insight={i} variant="workspace" active={tab === 'chat'} feedbackTarget={feedbackTarget}
            readerOpen={readerOpen} onToggleReader={() => setReaderOpen(value => !value)}
            selectedQuote={selectedQuote} onClearQuote={() => setSelectedQuote(null)} onQuote={setSelectedQuote} />
          {readerOpen ? <InsightReader insight={i} compact={compact} canAsk={FEEDBACK_ENABLED && !!i.pipeline_result}
            onQuote={text => { setSelectedQuote({ text }); if (window.matchMedia('(max-width: 900px)').matches) setReaderOpen(false); }} onClose={() => setReaderOpen(false)}>
            <details className="chat-context-sources"><summary>Материалы и источники · {used.length}</summary><Sources ids={used} sources={sources} /></details>
          </InsightReader> : <aside className="chat-context" aria-label="Контекст обсуждения">
            <div className="chat-context-heading"><InsightIcon name="document" size={18} /><span>В фокусе разговора</span></div>
            <div className="ic-tags"><SignalBadge insight={i} /></div>
            <h3>{v.title}</h3>
            <p className="chat-context-lead">{v.lead}</p>
            <button className="chat-context-link" type="button" onClick={() => setReaderOpen(true)}>Читать текст и выделить фрагмент<InsightIcon name="document" size={16} /></button>
            <button className="chat-context-link" type="button" onClick={() => changeTab('overview')}>К обзору инсайта<InsightIcon name="arrow" size={16} /></button>
            <details className="chat-context-sources"><summary>Материалы и источники · {used.length}</summary>
            <Sources ids={used} sources={sources} />
            </details>
            <div className="chat-context-note"><InsightIcon name="sparkle" size={17} /><p>Один инсайт — отдельный разговор. История обсуждения сохраняется здесь.</p></div>
          </aside>}
        </ChatSplit>
      </section>
    </div>
    <footer className="workbench-footer" hidden={tab === 'chat'}>
      <div className="workbench-feedback" ref={setFeedbackTarget} />
      <Button size={32} view="text" onClick={() => onHide(i.id)}>Скрыть инсайт</Button>
    </footer>
  </>;
}

const InsightBody = memo(function InsightBody({ insight: i, v, compact, sources }) {
  return <div className="iwin-main">
            {v.facts.length > 0 && <Section kind="fact" label="Факты · что произошло">
              <ul className="facts">{v.facts.map((x, n) => {
                const ids = refs(x, sources, ['news', 'bank']);
                const full = compact && i.what_happened.filter(c => x.ids?.includes(c.id)).map(c => c.text).join(' ');
                return <li key={n}>{x.text}
                  {ids.map(id => <NewsRef key={id} id={id} sources={sources} />)}
                  <More full={full} quotes={ids.map(id => sources[id]?.snippet).filter(Boolean)} /></li>;
              })}</ul>
            </Section>}
            <Section kind="ai" label="Вывод AI · почему это важно вам"><Labeled items={v.why} /></Section>
            {v.important_if.length > 0 && <Section kind="hyp" label="Особенно важно, если">
              <ul>{v.important_if.map(t => <li key={t}>{t}</li>)}</ul>
            </Section>}
            {(v.risks.length > 0 || v.opportunities.length > 0) && <div className="ro">
              {v.risks.length > 0 && <Section kind="risk" label="🐻 Риски"><Labeled items={v.risks} /></Section>}
              {v.opportunities.length > 0 && <Section kind="opportunity" label="📈 Возможности"><Labeled items={v.opportunities} /></Section>}
            </div>}
            {v.actions.length > 0 && <Section kind="act" label="Что сделать"><ol>{v.actions.map(a => <li key={a}>{a}</li>)}</ol></Section>}
            {v.bank_view.length
              ? <Section kind="bank" label="Позиция Банка"><ul>{v.bank_view.map((x, n) => <li key={n}>{x.text}</li>)}</ul></Section>
              : <Plate view="attention" title="Аналитики Банка по этой теме нет">Вывод не является официальной позицией Банка.</Plate>}
            {!compact && i.ai_conclusion && <Section kind="ai" label="Итог AI"><p>{i.ai_conclusion}</p></Section>}
            {i.bank_product && <Plate view="common" title="Чем может помочь Банк">{i.bank_product}</Plate>}
            {i.trace?.length > 0 && <details className="trace">
              <summary>Как мы пришли к выводу</summary>
              <ol>{i.trace.map((step, n) => <li key={n}>{step.text}</li>)}</ol>
            </details>}
          </div>;
});
