// The insight view (24.10 review): a short card in the feed and a large window per insight — the screen
// version of the insight on the left (the full insight when there is none), the persisted AI discussion and
// merged sources on the right. The side-panel variant was removed on 24.10: the window is the demo view.
import React, { memo, useEffect, useMemo, useState } from 'react';
import { ModalDesktop as Modal } from '@alfalab/core-components/modal/desktop';
import { Button } from '@alfalab/core-components/button';
import { Status } from '@alfalab/core-components/status';
import { Plate } from '@alfalab/core-components/plate';
import { Typography } from '@alfalab/core-components/typography';
import { InsightDiscussion } from './discussion.jsx';
import { loadInsight } from './api.js';

import { CONFIDENCE, day, view } from './insightView.js';

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
  if (!detail) return <Modal open onClose={onClose} size={1140} hasCloser transitionProps={NO_TRANSITION}>
    <Modal.Header hasCloser sticky />
    <Modal.Content><div className="iwin">
      <Typography.Title font="system" tag="h2" view="medium" weight="bold">{i.title}</Typography.Title>
      {error ? <><p role="alert">Не удалось загрузить инсайт: {error}</p>
        <Button size={40} view="secondary" onClick={() => setRetry(n => n + 1)}>Повторить</Button></>
        : <p role="status">Загружаем материалы инсайта…</p>}
    </div></Modal.Content>
  </Modal>;
  return <InsightDetails insight={detail} compact={compact} sources={detail.sources || sources}
    company={company} onClose={onClose} onHide={onHide} />;
}

function InsightDetails({ insight: i, compact, sources, company, onClose, onHide }) {
  const [feedbackTarget, setFeedbackTarget] = useState(null);
  const v = useMemo(() => view(i, compact), [i, compact]);
  const [color, conf] = CONFIDENCE[i.confidence] || CONFIDENCE.low;
  const used = useMemo(() => [...new Set([...v.facts, ...v.why, ...v.risks, ...v.opportunities, ...v.bank_view, ...(i.hypotheses || [])].flatMap(x => x.sources || []))], [v, i]);

  return <Modal open onClose={onClose} size={1140} hasCloser transitionProps={NO_TRANSITION}>
    <Modal.Header hasCloser sticky />
    <Modal.Content>
      <div className="iwin">
        <p className="iwin-disc">Не является инвестиционной рекомендацией.{!i.bank_view.length && ' Вывод не является официальной позицией Банка.'}</p>
        <div className="ic-tags">
          <Status color="blue" size={20}>{i.channel}</Status>
          <Status color={color} size={20}>{conf}</Status>
          {i.bank_view.length > 0 && <Status color="purple" size={20}>Есть позиция Банка</Status>}
          <span className="ic-date">{day(i.published)}</span>
        </div>
        <Typography.Title font="system" tag="h2" view="medium" weight="bold">{v.title}</Typography.Title>
        <Typography.Text view="primary-large" color="secondary" tag="p">{v.lead}</Typography.Text>

        <div className="iwin-grid">
          <InsightBody insight={i} v={v} compact={compact} sources={sources} />
          <aside className="iwin-side">
            <InsightDiscussion company={company} insight={i} variant="window" feedbackTarget={feedbackTarget} />
            <Sources ids={used} sources={sources} />
          </aside>
        </div>
      </div>
    </Modal.Content>
    <Modal.Footer sticky>
      <div className="fb insight-footer">
        <div ref={setFeedbackTarget} />
        <div className="fb-btns">
          <Button size={40} view="text" onClick={() => onHide(i.id)}>Скрыть</Button>
        </div>
      </div>
    </Modal.Footer>
  </Modal>;
}

const InsightBody = memo(function InsightBody({ insight: i, v, compact, sources }) {
  return <div className="iwin-main">
            <Section kind="fact" label="Факты · что произошло">
              <ul className="facts">{v.facts.map((x, n) => {
                const ids = refs(x, sources, ['news', 'bank']);
                const full = compact && i.what_happened.filter(c => x.ids?.includes(c.id)).map(c => c.text).join(' ');
                return <li key={n}>{x.text}
                  {ids.map(id => <NewsRef key={id} id={id} sources={sources} />)}
                  <More full={full} quotes={ids.map(id => sources[id]?.snippet).filter(Boolean)} /></li>;
              })}</ul>
            </Section>
            <Section kind="ai" label="Вывод AI · почему это важно вам"><Labeled items={v.why} /></Section>
            {v.important_if.length > 0 && <Section kind="hyp" label="Особенно важно, если">
              <ul>{v.important_if.map(t => <li key={t}>{t}</li>)}</ul>
            </Section>}
            {i.hypotheses?.length > 0 && <Section kind="hyp" label="Гипотезы · требуют проверки"><Labeled items={i.hypotheses} /></Section>}
            {(v.risks.length > 0 || v.opportunities.length > 0) && <div className="ro">
              {v.risks.length > 0 && <Section kind="ai" label="Риски"><Labeled items={v.risks} /></Section>}
              {v.opportunities.length > 0 && <Section kind="ai" label="Возможности"><Labeled items={v.opportunities} /></Section>}
            </div>}
            {v.actions.length > 0 && <Section kind="act" label="Что сделать"><ol>{v.actions.map(a => <li key={a}>{a}</li>)}</ol></Section>}
            {v.bank_view.length
              ? <Section kind="bank" label="Позиция Банка"><ul>{v.bank_view.map((x, n) => <li key={n}>{x.text}</li>)}</ul></Section>
              : <Plate view="attention" title="Аналитики Банка по этой теме нет">Вывод не является официальной позицией Банка.</Plate>}
            {i.ai_conclusion && <Section kind="ai" label="Итог AI"><p>{i.ai_conclusion}</p></Section>}
            {i.bank_product && <Plate view="common" title="Чем может помочь Банк">{i.bank_product}</Plate>}
            {i.trace?.length > 0 && <details className="trace">
              <summary>Как мы пришли к выводу</summary>
              <ol>{i.trace.map((step, n) => <li key={n}>{step.text}</li>)}</ol>
            </details>}
          </div>;
});
