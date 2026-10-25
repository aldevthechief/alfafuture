import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { Spinner } from '@alfalab/core-components/spinner';
import { ApiError } from './api.js';
import { ALLOW_INFERENCE } from './cabinetMode.js';
import { answerSources, askQuestion, discussionHistory, FEEDBACK_ENABLED, sendReaction } from './feedback.js';
import { InsightIcon } from './insightIcons.jsx';

const REASONS = [['Не про нас', 'not_relevant'], ['Уже знали', 'already_known'],
  ['Старая новость', 'dislike'], ['Ошибка в фактах', 'incorrect']];
const EMOJI = ['👍', '🔥', '🤔', '👎'];
const SUGGEST = [
  ['Оценить влияние', 'Какие риски и возможности здесь важнее всего для нашей компании?', 'document'],
  ['Разобрать сценарий', 'Как изменится вывод, если исходные условия изменятся?', 'sparkle'],
  ['Наметить действия', 'Что сделать в первую очередь?', 'arrow'],
];
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
const isMessage = item => ['question', 'comment', 'incorrect'].includes(item.kind);
const messageTime = value => {
  const date = value && new Date(value);
  return date && !isNaN(date) ? date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : '';
};

export const HistoryMessages = memo(function HistoryMessages({ history, record, insight, busy, onAsk }) {
  return history.filter(isMessage).map(item => <article key={item.id} className="chat-turn" aria-label="Сообщение и ответ агента">
    <div className="chat-message chat-message-user">
      <div className="chat-message-meta"><span>Вы</span><time dateTime={item.created_at}>{messageTime(item.created_at)}</time></div>
      <p>{item.comment || LABELS[item.kind] || item.kind}</p>
    </div>
    <div className="chat-message chat-message-agent">
      <span className="chat-message-avatar"><InsightIcon name="sparkle" size={17} /></span>
      <div className="chat-answer">
        <div className="chat-message-meta"><span>Агент по инсайту</span><span className="chat-ai-label">AI</span></div>
        {item.analysis ? <>
          {item.analysis.response.split(/\n{2,}/).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
          {item.analysis.needs_clarification && <p className="chat-clarification">{item.analysis.clarification_question}</p>}
          <Citations {...{ record, item, insight, history }} />
        </> : <><p className="chat-message-note">Сообщение сохранено, ответ пока не получен.</p>
          {item.kind === 'question' && <Button size={32} view="text" disabled={busy}
            onClick={() => onAsk(item.comment, item.request_id)}>Повторить запрос ответа</Button>}</>}
      </div>
    </div>
  </article>);
});

export function InsightDiscussion({ company, insight, variant, active = true, feedbackTarget }) {
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
  const input = useRef(null);
  const stickToBottom = useRef(true);
  const available = FEEDBACK_ENABLED && !!insight.pipeline_result;
  const workspace = variant === 'workspace';
  const hasMessages = history.some(isMessage);

  useEffect(() => {
    if (active && (hasMessages || pending) && stickToBottom.current && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [history, pending, busy, active, hasMessages]);
  useEffect(() => {
    if (!input.current) return;
    input.current.style.height = 'auto';
    input.current.style.height = `${Math.min(input.current.scrollHeight, 140)}px`;
  }, [question, active]);

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
    stickToBottom.current = true;
    setBusy(true); setError(''); setQuestion(''); setPending({ text, requestId });
    try {
      const answer = await askQuestion(company, insight, text, requestId, response => {
        if (alive.current) setPending(current => current?.requestId === requestId ? { ...current, response } : current);
      });
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
  return <section className={workspace ? 'discussion discussion-workspace' : 'layer discussion'} aria-label="Чат с агентом">
    <header className="discussion-header">
      <span className="agent-avatar"><InsightIcon name="sparkle" size={21} /></span>
      <div><h3>Агент по инсайту</h3><p>Сценарии, расчёты и следующие шаги</p></div>
      <span className={`agent-connection${record ? ' is-connected' : ''}`} role="status"><i />
        {record ? 'Контекст подключён' : loading ? 'Подключаем контекст' : available ? 'Нет подключения' : 'Только просмотр'}
      </span>
    </header>
    {feedbackTarget ? createPortal(reactions, feedbackTarget) : reactions}
    <div className="chat-transcript" ref={body} role="log" aria-label="История обсуждения" aria-busy={loading || busy} tabIndex={0}
      onScroll={() => { const node = body.current; if (node) stickToBottom.current = node.scrollHeight - node.scrollTop - node.clientHeight < 100; }}>
      <div className="chat-transcript-inner">
        {loading && <div className="chat-loading" role="status"><Spinner preset={24} visible />Загружаем историю разговора…</div>}
        {!loading && !hasMessages && !pending && <div className="chat-welcome">
          <div className="chat-welcome-orbit"><span className="agent-avatar"><InsightIcon name="sparkle" size={30} /></span></div>
          <span className="agent-eyebrow">ВАШ КОНТЕКСТ. ВАШИ РЕШЕНИЯ.</span>
          <h3>От важного события<br />к понятному действию</h3>
          <p>{!available ? ALLOW_INFERENCE ? 'Вопросы доступны для результатов анализа. Запустите анализ в ленте.'
            : 'В этой версии доступны готовые материалы. Обсуждение с AI — в расширенной версии.'
            : 'Агент уже знает материалы инсайта. Задайте свой вопрос или выберите, с чего начать.'}</p>
          {available && <div className="chat-starters">{SUGGEST.map(([label, text, icon]) =>
            <button type="button" key={label} onClick={() => { setQuestion(text); input.current?.focus(); }}>
              <InsightIcon name={icon} size={20} /><strong>{label}</strong><span>{text}</span>
            </button>)}</div>}
        </div>}
        <HistoryMessages {...{ history, record, insight, busy }} onAsk={ask} />
        {more && <Button size={32} view="text" disabled={busy} onClick={() => load(offset).catch(e => setError(errorText(e)))}>Загрузить продолжение истории</Button>}
        {pending && !history.some(item => item.request_id === pending.requestId) && <article className="chat-turn">
          <div className="chat-message chat-message-user"><div className="chat-message-meta"><span>Вы</span></div><p>{pending.text}</p></div>
          {!pending.failed && <div className="chat-message chat-message-agent">
            <span className="chat-message-avatar"><InsightIcon name="sparkle" size={17} /></span>
            <div className="chat-answer"><div className="chat-message-meta"><span>Агент по инсайту</span></div>
              {pending.response ? <div className="chat-streaming-answer">{pending.response.split(/\n{2,}/).map((paragraph, index) => <p key={index}>{paragraph}</p>)}<span className="chat-streaming-caret" aria-hidden="true" /></div>
                : <div className="thinking" role="status"><span className="dots"><i /><i /><i /></span>Изучаю вопрос и материалы…</div>}
            </div>
          </div>}
        </article>}
      </div>
    </div>
    <div className="chat-compose-area">
      {error && <div className="chat-error"><p role="alert" className="discussion-error">{error}</p>
        {pending?.failed && <Button size={32} view="secondary" disabled={busy}
          onClick={() => ask(pending.text, pending.requestId)}>Повторить запрос ответа</Button>}
        {available && !loading && !record && <Button size={32} view="secondary"
          onClick={() => load().then(() => setError('')).catch(e => setError(errorText(e)))}>Повторить подключение</Button>}
      </div>}
      <form className="chat-composer" onSubmit={event => { event.preventDefault(); ask(question.trim()); }}>
        <textarea ref={input} rows={2} maxLength={2000} placeholder="Задайте вопрос об этом инсайте…" aria-label="Сообщение агенту"
          value={question} disabled={!available || loading || busy} onChange={event => setQuestion(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault(); ask(question.trim());
            }
          }} />
        <button type="submit" className="chat-send" aria-label="Отправить сообщение" title="Отправить · Enter"
          disabled={!available || loading || busy || !question.trim()}>
          {busy ? <Spinner preset={24} visible /> : <InsightIcon name="send" size={22} />}
        </button>
      </form>
      <div className="chat-compose-hint"><span>Enter — отправить<span className="chat-desktop-hint"> · Shift+Enter — новая строка</span></span><span>{question.length} / 2000</span></div>
    </div>
  </section>;
}
