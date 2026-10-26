import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { Spinner } from '@alfalab/core-components/spinner';
import { ApiError, suggestedFor } from './api.js';
import { ALLOW_INFERENCE } from './cabinetMode.js';
import { answerSources, askQuestion, discussionHistory, FEEDBACK_ENABLED, sendReaction } from './feedback.js';
import { InsightIcon } from './insightIcons.jsx';
import { QuotedExcerpt } from './insightReader.jsx';
import { ExcerptAction, useExcerptSelection } from './excerptSelection.jsx';
import { CitedText } from './sourceReferences.jsx';
import { sourceUrl } from './sourceReferences.js';

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
export const errorText = error => error instanceof ApiError
  ? error.status === 503 ? 'Ассистент не настроен. Сообщение сохранено — ответ можно запросить повторно.'
    : error.status === 502 ? 'Ответ пока недоступен. Сообщение сохранено — попробуйте ещё раз.'
      : error.status === 409 ? 'Не удалось связать обсуждение с карточкой. Попробуйте открыть инсайт заново.'
        : 'Не удалось загрузить обсуждение. Попробуйте ещё раз.'
  : error.message === 'Failed to fetch' ? 'Не удалось связаться с сервисом обсуждения. Попробуйте ещё раз.' : error.message;

function messageSources(record, item, insight, history) {
  const sources = answerSources(record, item, insight?.sources);
  history.forEach(message => { sources[`user:${message.id}`] = { kind:'user', title:'Ваше уточнение',
    text:[message.selected_text && `Фрагмент ${message.selected_source === 'answer' ? 'ответа агента' : 'инсайта'}: ${message.selected_text}`, message.comment].filter(Boolean).join('\n\n') }; });
  return sources;
}

const Citations = memo(function Citations({ item, sources }) {
  const ids = item.analysis?.source_ids || [];
  if (!ids.length) return null;
  return <details className="qa-sources"><summary>Источники ответа · {ids.length}</summary>
    <ul>{ids.map(id => {
      const s = sources[id];
      return <li key={id}><b>{s?.title || (s?.kind === 'fact' ? 'Данные компании / расчёт' : 'Источник недоступен')}</b>
        {s?.page && `, с. ${s.page}`}<p>{s?.text || s?.snippet || 'Материал отсутствует в сохранённом снимке.'}</p>
        {sourceUrl(s?.url) && <a href={sourceUrl(s.url)} target="_blank" rel="noreferrer">Открыть источник</a>}</li>;
    })}</ul>
  </details>;
});

// Typing a question must not rebuild every saved answer and its evidence map.
const isMessage = item => ['question', 'comment', 'incorrect'].includes(item.kind);
const messageTime = value => {
  const date = value && new Date(value);
  return date && !isNaN(date) ? date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : '';
};

export const HistoryMessages = memo(function HistoryMessages({ history, record, insight, busy, onAsk, agentLabel = 'Агент по инсайту' }) {
  return history.filter(isMessage).map(item => {
    const sources = messageSources(record, item, insight, history);
    return <article key={item.id} className="chat-turn" aria-label="Сообщение и ответ агента">
    <div className="chat-message chat-message-user">
      <div className="chat-message-meta"><span>Вы</span><time dateTime={item.created_at}>{messageTime(item.created_at)}</time></div>
      <QuotedExcerpt text={item.selected_text} source={item.selected_source} />
      <p>{item.comment || LABELS[item.kind] || item.kind}</p>
    </div>
    <div className="chat-message chat-message-agent">
      <span className="chat-message-avatar"><InsightIcon name="sparkle" size={17} /></span>
      <div className="chat-answer">
        <div className="chat-message-meta"><span>{agentLabel}</span><span className="chat-ai-label">AI</span></div>
        {item.analysis ? <>
          <div className="chat-answer-copy" data-answer-excerpt="true">
            {item.analysis.response.split(/\n{2,}/).map((paragraph, index) => <p key={index}><CitedText text={paragraph} sources={sources} /></p>)}
            {item.analysis.needs_clarification && <p className="chat-clarification"><CitedText text={item.analysis.clarification_question} sources={sources} /></p>}
          </div>
          <Citations {...{ item, sources }} />
        </> : <><p className="chat-message-note">Сообщение сохранено, ответ пока не получен.</p>
          {item.kind === 'question' && <Button size={32} view="text" disabled={busy}
            onClick={() => onAsk(item.comment, item.request_id, item.selected_text || '', item.selected_source || 'insight')}>Повторить запрос ответа</Button>}</>}
      </div>
    </div>
  </article>;
  });
});

export function InsightDiscussion({ company, insight, variant, active = true, feedbackTarget,
  readerOpen, onToggleReader, selectedQuote, onClearQuote, onQuote, queuedQuestion, onQueuedDone }) {
  const [record, setRecord] = useState(null);
  const [history, setHistory] = useState([]);
  const [offset, setOffset] = useState(0);
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [reacting, setReacting] = useState(false);
  const [question, setQuestion] = useState('');
  const [suggestions,setSuggestions]=useState([]);
  const [pending, setPending] = useState(null);
  const [error, setError] = useState('');
  const [reasons, setReasons] = useState(false);
  const alive = useRef(true);
  const body = useRef(null);
  const panel = useRef(null);
  const input = useRef(null);
  const stickToBottom = useRef(true);
  const available = FEEDBACK_ENABLED && !!insight.pipeline_result;
  const workspace = variant === 'workspace';
  const hasMessages = history.some(isMessage);
  const [answerSelection, clearAnswerSelection] = useExcerptSelection({ root:body, anchor:panel,
    selector:'[data-answer-excerpt]', enabled:available && active && !busy && !loading && !!onQuote });

  useEffect(() => {
    if (active && (hasMessages || pending) && stickToBottom.current && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [history, pending, busy, active, hasMessages]);
  useEffect(() => {
    if (!input.current) return;
    input.current.style.height = 'auto';
    input.current.style.height = `${Math.min(input.current.scrollHeight, 140)}px`;
  }, [question, active]);
  useEffect(() => {
    if (selectedQuote && active && !busy && !loading && available) input.current?.focus();
  }, [selectedQuote, active, busy, loading, available]);

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
    const timer=available ? setTimeout(()=>load().catch(e=>alive.current && setError(errorText(e)))
      .finally(()=>alive.current && setLoading(false)),active ? 0 : 1200) : null;
    if (!available) setLoading(false);
    return () => { alive.current = false; clearTimeout(timer); };
  }, [company, insight.id, insight.job_id, available]);

  useEffect(()=>{
    if (!available) return;
    let cancelled=false;
    suggestedFor(insight.job_id,insight.id).then(list=>{if (!cancelled) setSuggestions(list);});
    return ()=>{cancelled=true;};
  },[available,insight.job_id,insight.id]);

  const ask = useCallback(async (text, requestId = crypto.randomUUID(), selectedText = selectedQuote?.text || '', selectedSource = selectedQuote?.source || 'insight') => {
    if (!text || busy || !available) return;
    stickToBottom.current = true;
    setBusy(true); setError(''); setQuestion(''); setPending({ text, requestId, selectedText, selectedSource }); onClearQuote?.();
    try {
      const answer = await askQuestion(company, insight, text, requestId, response => {
        if (alive.current) setPending(current => current?.requestId === requestId ? { ...current, response } : current);
      }, selectedText, selectedSource);
      if (!alive.current) return;
      setHistory(items => merge(items, [answer]));
      setPending(null);
    } catch (e) {
      if (alive.current) { setError(errorText(e)); setPending({ text, requestId, selectedText, selectedSource, failed:true }); }
    } finally { if (alive.current) setBusy(false); }
  }, [available, busy, company, insight, selectedQuote, onClearQuote]);

  const queuedSent=useRef(null);
  useEffect(()=>{
    if (!queuedQuestion || queuedSent.current===queuedQuestion || !available || loading || busy || !record) return;
    queuedSent.current=queuedQuestion;
    onQueuedDone?.();
    ask(queuedQuestion.text,queuedQuestion.id,'','insight');
  },[queuedQuestion,available,loading,busy,record,ask,onQueuedDone]);

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
  return <section ref={panel} className={workspace ? 'discussion discussion-workspace' : 'layer discussion'} aria-label="Чат с агентом">
    <header className="discussion-header">
      <span className="agent-avatar"><InsightIcon name="sparkle" size={21} /></span>
      <div><h3>Агент по инсайту</h3><p>Сценарии, расчёты и следующие шаги</p></div>
      {onToggleReader && <button className="chat-reader-toggle" type="button" aria-expanded={readerOpen} onClick={onToggleReader}>
        <InsightIcon name="document" size={17} /><span>Текст инсайта</span>
      </button>}
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
          <div className="chat-message chat-message-user"><div className="chat-message-meta"><span>Вы</span></div><QuotedExcerpt text={pending.selectedText} source={pending.selectedSource} /><p>{pending.text}</p></div>
          {!pending.failed && <div className="chat-message chat-message-agent">
            <span className="chat-message-avatar"><InsightIcon name="sparkle" size={17} /></span>
            <div className="chat-answer"><div className="chat-message-meta"><span>Агент по инсайту</span></div>
              {pending.response ? <div className="chat-streaming-answer">{pending.response.split(/\n{2,}/).map((paragraph, index) => <p key={index}><CitedText text={paragraph} sources={answerSources(record, {}, insight.sources)} pending /></p>)}<span className="chat-streaming-caret" aria-hidden="true" /></div>
                : <div className="thinking" role="status"><span className="dots"><i /><i /><i /></span>Изучаю вопрос и материалы…</div>}
            </div>
          </div>}
        </article>}
      </div>
    </div>
    <div className="chat-compose-area">
      {available && !loading && suggestions.length>0 && <div className="chat-suggestions" aria-label="Вопросы по инсайту">
        {suggestions.map(q=><button key={q} type="button" className="chat-suggestion" title={q} disabled={busy || !record} onClick={()=>ask(q,undefined,'','insight')}>{q}</button>)}
      </div>}
      {error && <div className="chat-error"><p role="alert" className="discussion-error">{error}</p>
        {pending?.failed && <Button size={32} view="secondary" disabled={busy}
          onClick={() => ask(pending.text, pending.requestId, pending.selectedText || '', pending.selectedSource || 'insight')}>Повторить запрос ответа</Button>}
        {available && !loading && !record && <Button size={32} view="secondary"
          onClick={() => load().then(() => setError('')).catch(e => setError(errorText(e)))}>Повторить подключение</Button>}
      </div>}
      {selectedQuote && <div className="chat-selected-quote" role="region" aria-label="Прикреплённый фрагмент">
        <QuotedExcerpt text={selectedQuote.text} source={selectedQuote.source} />
        <button type="button" className="workbench-icon-button" onClick={onClearQuote} aria-label="Убрать фрагмент из вопроса"><InsightIcon name="close" size={16} /></button>
      </div>}
      <form className="chat-composer" onSubmit={event => { event.preventDefault(); ask(question.trim()); }}>
        <textarea ref={input} rows={2} maxLength={2000} placeholder={selectedQuote ? 'Что хотите уточнить об этом фрагменте?' : 'Задайте вопрос об этом инсайте…'} aria-label="Сообщение агенту"
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
    <ExcerptAction selection={answerSelection} onQuote={text => { onQuote({ text, source:'answer' }); clearAnswerSelection(); }} />
  </section>;
}
