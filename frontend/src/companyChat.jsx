import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@alfalab/core-components/button';
import { Spinner } from '@alfalab/core-components/spinner';
import { askCompanyChat, companyChatHistory, FEEDBACK_ENABLED, openCompanyChat } from './feedback.js';
import { HistoryMessages } from './discussion.jsx';
import { QuotedExcerpt } from './insightReader.jsx';
import { ExcerptAction, useExcerptSelection } from './excerptSelection.jsx';
import { InsightIcon } from './insightIcons.jsx';
import { CitedText } from './sourceReferences.jsx';
import './insightWorkspace.css';
import './companyChat.css';

const merge = (items, added) => [...new Map([...items, ...added].map(item => [item.id,item])).values()];
const messageError = error => [502,503].includes(error.status)
  ? 'Ответ пока недоступен. Вопрос сохранён — попробуйте повторить запрос.' : error.message;

export default function CompanyChat({ company, role, name, roleTitle, open, onClose }) {
  const [conversation, setConversation] = useState(null);
  const [history, setHistory] = useState([]);
  const [before, setBefore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [question, setQuestion] = useState('');
  const [quote, setQuote] = useState(null);
  const [pending, setPending] = useState(null);
  const alive = useRef(true);
  const connecting = useRef(false);
  const sending = useRef(false);
  const transcript = useRef(null);
  const panel = useRef(null);
  const input = useRef(null);
  const stick = useRef(true);
  const [selection, clearSelection] = useExcerptSelection({ root:transcript, anchor:panel,
    selector:'[data-answer-excerpt]', enabled:open && FEEDBACK_ENABLED && !busy && !!conversation });

  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const connect = useCallback(async () => {
    if (connecting.current || !FEEDBACK_ENABLED) return;
    connecting.current = true;
    setLoading(true); setError('');
    try {
      const chat = await openCompanyChat(company, role);
      const data = await companyChatHistory(chat.id);
      if (alive.current) { setConversation(chat); setHistory(data.items); setBefore(data.before); }
    } catch (err) { if (alive.current) setError(messageError(err)); }
    finally { connecting.current = false; if (alive.current) setLoading(false); }
  }, [company,role]);
  useEffect(() => { if (open && !conversation) connect(); }, [open, conversation, connect]);
  useEffect(() => {
    if (open && !busy && !loading && conversation) input.current?.focus();
  }, [open, quote, conversation, loading, busy]);
  useEffect(() => {
    if (open && stick.current && transcript.current) transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [open, history, pending]);

  const ask = useCallback(async (text, requestId = crypto.randomUUID(), selectedText = quote?.text || '') => {
    if (!text || !conversation || sending.current || !FEEDBACK_ENABLED) return;
    sending.current = true; stick.current = true;
    setBusy(true); setError(''); setQuestion(''); setQuote(null);
    setPending({ text, requestId, selectedText });
    try {
      const result = await askCompanyChat(conversation.id, text, requestId, response => {
        if (alive.current) setPending(previous => previous?.requestId === requestId ? { ...previous,response } : previous);
      }, selectedText);
      if (alive.current) { setHistory(items => merge(items,[result])); setPending(null); }
    } catch (err) {
      if (alive.current) { setError(messageError(err)); setPending({ text,requestId,selectedText,failed:true }); }
    } finally { sending.current = false; if (alive.current) setBusy(false); }
  }, [conversation, quote]);

  const older = async () => {
    if (loading || !before) return;
    setLoading(true); setError(''); stick.current = false;
    const height = transcript.current.scrollHeight;
    try {
      const data = await companyChatHistory(conversation.id, before);
      if (alive.current) {
        setHistory(items => merge(data.items,items)); setBefore(data.before);
        requestAnimationFrame(() => { if (transcript.current) transcript.current.scrollTop += transcript.current.scrollHeight - height; });
      }
    } catch (err) { if (alive.current) setError(messageError(err)); }
    finally { if (alive.current) setLoading(false); }
  };
  const shown = pending && !pending.failed ? history.filter(item => item.request_id !== pending.requestId) : history;
  return createPortal(<section ref={panel} className="company-bank-chat discussion-workspace" role="dialog"
    aria-label="Чат с банком" hidden={!open} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose(); } }}>
    <header className="discussion-header">
      <span className="chat-bank-ava" aria-hidden="true"><svg width="16" height="19" viewBox="0 0 22 26" fill="currentColor"><path d="M11 0 2.6 19.5h4.3l1.6-4.1h5l1.6 4.1h4.3L11 0zm-1.2 11.9L11 8.5l1.2 3.4z"/><rect x="2.6" y="22" width="16.8" height="3.2" rx="1.6"/></svg></span>
      <div><h3>Чат с банком <span className="chat-ai-label">AI</span></h3><p>{name} · {roleTitle}</p></div>
      <button type="button" className="workbench-icon-button" aria-label="Закрыть чат с банком" onClick={onClose}><InsightIcon name="close" size={19} /></button>
    </header>
    <div className="chat-transcript" ref={transcript} role="log" aria-label="Переписка с помощником компании" aria-busy={busy || loading} tabIndex={0}
      onScroll={() => { const node = transcript.current; stick.current = node.scrollHeight - node.scrollTop - node.clientHeight < 100; }}>
      <div className="chat-transcript-inner">
        {before && <Button size={32} view="text" disabled={loading || busy} onClick={older}>Загрузить предыдущие сообщения</Button>}
        {loading && <div className="chat-loading" role="status"><Spinner preset={24} visible />Загружаем переписку…</div>}
        {!loading && !history.length && !pending && <div className="company-chat-welcome">
          <p>{FEEDBACK_ENABLED ? 'Здравствуйте! Чем можем помочь вашей компании?' : 'AI-чат доступен в полной версии кабинета.'}</p>
        </div>}
        <HistoryMessages history={shown} busy={busy} onAsk={ask} agentLabel="Помощник компании" />
        {pending && !shown.some(item => item.request_id === pending.requestId) && <article className="chat-turn">
          <div className="chat-message chat-message-user"><div className="chat-message-meta">Вы</div>
            <QuotedExcerpt text={pending.selectedText} source="answer" /><p>{pending.text}</p></div>
          {!pending.failed && <div className="chat-message chat-message-agent"><span className="chat-message-avatar"><InsightIcon name="sparkle" size={17} /></span>
            <div className="chat-answer"><div className="chat-message-meta">Помощник компании</div>
              {pending.response ? <div className="chat-streaming-answer"><CitedText text={pending.response} sources={{}} pending /><span className="chat-streaming-caret" /></div>
                : <div className="thinking" role="status"><Spinner preset={16} visible />Изучаю вопрос и материалы…</div>}
            </div></div>}
        </article>}
      </div>
    </div>
    <div className="chat-compose-area">
      {error && <div className="chat-error"><p role="alert">{error}</p>
        {!conversation && <Button size={32} view="text" onClick={connect} disabled={loading}>Повторить подключение</Button>}
        {pending?.failed && <Button size={32} view="text" disabled={busy} onClick={() => ask(pending.text,pending.requestId,pending.selectedText)}>Повторить запрос</Button>}</div>}
      {quote && <div className="chat-selected-quote"><QuotedExcerpt text={quote.text} source="answer" />
        <button type="button" className="workbench-icon-button" aria-label="Убрать фрагмент из вопроса" onClick={() => setQuote(null)}><InsightIcon name="close" size={16} /></button></div>}
      <form className="chat-composer" onSubmit={event => { event.preventDefault(); ask(question.trim()); }}>
        <textarea ref={input} rows={1} maxLength={2000} aria-label="Вопрос помощнику компании" placeholder={quote ? 'Что уточнить?' : 'Сообщение'}
          value={question} disabled={!conversation || busy || !FEEDBACK_ENABLED} onChange={event => setQuestion(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); ask(question.trim()); } }} />
        <button type="submit" className="chat-send" aria-label="Отправить вопрос компании" disabled={!conversation || busy || !question.trim() || !FEEDBACK_ENABLED}>
          {busy ? <Spinner preset={24} visible /> : <InsightIcon name="arrow" size={20} />}</button>
      </form>
    </div>
    <ExcerptAction selection={selection} onQuote={text => { setQuote({text}); clearSelection(); }} />
  </section>, document.getElementById('company-chat-root'));
}
