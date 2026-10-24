import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { Input } from '@alfalab/core-components/input';
import { Spinner } from '@alfalab/core-components/spinner';
import { Typography } from '@alfalab/core-components/typography';
import { Tag } from '@alfalab/core-components/tag';
import { ApiError } from './api.js';
import { answerSources, askQuestion, discussionHistory, FEEDBACK_ENABLED, sendReaction } from './feedback.js';

const REASONS = [['Не про нас', 'not_relevant'], ['Уже знали', 'already_known'],
  ['Старая новость', 'dislike'], ['Ошибка в фактах', 'incorrect']];
const EMOJI = ['👍', '🔥', '🤔', '👎'];
const SUGGEST = ['Что будет, если ЦБ снизит ставку?', 'Сколько это может стоить компании?', 'Что сделать в первую очередь?'];
const LABELS = { like:'Полезно', useful:'Полезно', dislike:'Не полезно', not_relevant:'Не про нас',
  already_known:'Уже знали', incorrect:'Ошибка в фактах', emoji:'Реакция' };
const merge = (items, added) => [...new Map([...items, ...added].map(item => [item.id, item])).values()];
const errorText = error => error instanceof ApiError
  ? error.status === 503 ? 'Ассистент не настроен. Сообщение сохранено — ответ можно запросить повторно.'
    : error.status === 502 ? 'Ответ пока недоступен. Сообщение сохранено — попробуйте ещё раз.'
      : `Сервис обсуждения: ${error.message}`
  : error.message === 'Failed to fetch' ? 'Не удалось связаться с сервисом обсуждения. Попробуйте ещё раз.' : error.message;

const Citations = memo(function Citations({ record, item, insight, history }) {
  const ids = item.analysis?.source_ids || [];
  if (!ids.length) return null;
  const sources = answerSources(record, item, insight.sources);
  history.forEach(message => { sources[`user:${message.id}`] = { title:'Комментарий пользователя', text:message.comment }; });
  return <details className="qa-sources"><summary>Источники ответа · {ids.length}</summary>
    <ul>{ids.map(id => {
      const s = sources[id];
      return <li key={id}><b>{s?.title || (s?.kind === 'fact' ? 'Данные компании / расчёт' : id)}</b>
        {s?.page && `, с. ${s.page}`}<p>{s?.text || s?.snippet || id}</p>
        {s?.url && <a href={s.url} target="_blank" rel="noreferrer">Открыть источник</a>}</li>;
    })}</ul>
  </details>;
});

// Typing a question must not rebuild every saved answer and its evidence map.
const HistoryMessages = memo(function HistoryMessages({ history, record, insight, busy, onAsk }) {
  return history.map(item => <div key={item.id} className="qa">
    <b>{item.comment || item.emoji || LABELS[item.kind] || item.kind}</b>
    {item.analysis ? <>
      <p>{item.analysis.response}</p>
      {item.analysis.needs_clarification && <p>{item.analysis.clarification_question}</p>}
      <Citations {...{ record, item, insight, history }} />
    </> : <><p>Сообщение сохранено, ответ пока не получен.</p>
      {item.kind === 'question' && <Button size={32} view="text" disabled={busy}
        onClick={() => onAsk(item.comment, item.request_id)}>Повторить запрос ответа</Button>}</>}
  </div>);
});

export function InsightDiscussion({ company, insight, variant, feedbackTarget }) {
  const [record, setRecord] = useState(null);
  const [history, setHistory] = useState([]);
  const [offset, setOffset] = useState(0);
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [reacting, setReacting] = useState(false);
  const [question, setQuestion] = useState('');
  const [pending, setPending] = useState(null);
  const [error, setError] = useState('');
  const [reasons, setReasons] = useState(false);
  const alive = useRef(true);
  const body = useRef(null);
  const available = FEEDBACK_ENABLED && !!insight.pipeline_result;
  const inWindow = variant === 'window';

  useEffect(() => {
    if (body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [history, pending, busy]);

  async function load(start = 0) {
    const data = await discussionHistory(company, insight, start);
    if (!alive.current) return;
    setRecord(data.record);
    setHistory(items => start ? merge(items, data.items) : data.items);
    setOffset(start + data.items.length);
    setMore(data.items.length === 50);
  }

  useEffect(() => {
    alive.current = true;
    if (available) load().catch(e => alive.current && setError(errorText(e)))
      .finally(() => alive.current && setLoading(false));
    else setLoading(false);
    return () => { alive.current = false; };
  }, [company, insight.id, insight.job_id, available]);

  const ask = useCallback(async (text, requestId = crypto.randomUUID()) => {
    if (!text || busy || !available) return;
    setBusy(true); setError(''); setQuestion(''); setPending({ text, requestId });
    try {
      const answer = await askQuestion(company, insight, text, requestId);
      if (!alive.current) return;
      setHistory(items => merge(items, [answer]));
      setPending(null);
    } catch (e) {
      if (alive.current) { setError(errorText(e)); setPending({ text, requestId, failed:true }); }
    } finally { if (alive.current) setBusy(false); }
  }, [available, busy, company, insight]);

  async function react(kind, emoji) {
    if (reacting || !available) return;
    setReacting(true); setError('');
    const body = { kind, ...(emoji ? { emoji } : {}) };
    if (kind === 'incorrect') Object.assign(body, { comment:'Пользователь отметил ошибку в фактах карточки', scope:'future' });
    try {
      await sendReaction(company, insight, body, crypto.randomUUID());
      if (alive.current) { setReasons(false); await load(); }
    } catch (e) { if (alive.current) setError(errorText(e)); }
    finally { if (alive.current) setReacting(false); }
  }

  const disabled = !available || loading || reacting;
  const rating = [...history].reverse().find(item => ['like','useful','dislike','not_relevant','already_known'].includes(item.kind));
  const reactions = <>
    <div className="fb-btns">
      <Button size={32} view={['like','useful'].includes(rating?.kind) ? 'primary' : 'secondary'}
        disabled={disabled} onClick={() => react('like')}>Полезно</Button>
      <Button size={32} view="secondary" disabled={disabled} onClick={() => setReasons(v => !v)}>Не полезно</Button>
      {EMOJI.map(emoji => <Button key={emoji} size={32} view="text" disabled={disabled}
        onClick={() => react('emoji', emoji)}>{emoji}</Button>)}
      {reacting && <Spinner preset={16} visible />}
    </div>
    {reasons && <div className="fb-reasons">{REASONS.map(([label, kind]) =>
      <Button key={kind} size={32} view="secondary" disabled={disabled} onClick={() => react(kind)}>{label}</Button>)}</div>}
    {rating && <p className="discussion-saved">Сохранено: {LABELS[rating.kind]}. Учтём при следующем анализе.</p>}
  </>;
  return <section className={inWindow ? 'side-card chat2 discussion' : 'layer discussion'}>
    <Typography.Text view="caps" weight="bold" tag="div">{inWindow ? 'Спросить AI об этом событии' : 'Обсуждение инсайта'}</Typography.Text>
    {!available && <p>Вопросы и реакции доступны для результатов анализа. Запустите анализ в ленте.</p>}
    {loading && <p><Spinner preset={16} visible /> Загружаем историю…</p>}
    {feedbackTarget ? createPortal(reactions, feedbackTarget) : reactions}
    <div className={inWindow ? 'chat2-body' : undefined} ref={body}>
    {inWindow && available && !loading && !history.length && !pending && <div className="insight-suggestions">
      {SUGGEST.map(text => <Tag key={text} size={32} onClick={() => ask(text)}>{text}</Tag>)}
    </div>}
    <HistoryMessages {...{ history, record, insight, busy }} onAsk={ask} />
    {more && <Button size={32} view="text" onClick={() => load(offset).catch(e => setError(errorText(e)))}>Показать ещё</Button>}
    {pending && !history.some(item => item.request_id === pending.requestId) && <div className="qa">
      <b>{pending.text}</b>
      {!pending.failed && <div className="thinking" role="status"><span className="dots"><i /><i /><i /></span>AI думает…</div>}
    </div>}
    </div>
    {error && <p role="alert" className="discussion-error">{error}</p>}
    {pending?.failed && <Button size={32} view="secondary" disabled={busy}
      onClick={() => ask(pending.text, pending.requestId)}>Повторить запрос ответа</Button>}
    {available && !loading && !record && <Button size={32} view="secondary"
      onClick={() => load().then(() => setError('')).catch(e => setError(errorText(e)))}>Повторить подключение</Button>}
    <Input block size={48} placeholder="Например: что будет, если ЦБ снизит ставку?" value={question}
      disabled={!available || loading || busy} onChange={e => setQuestion(e.target.value)}
      onKeyDown={e => e.key === 'Enter' && ask(question.trim())} />
    <Button size={32} view="secondary" disabled={!available || loading || busy || !question.trim()}
      onClick={() => ask(question.trim())}>Задать вопрос</Button>
  </section>;
}
