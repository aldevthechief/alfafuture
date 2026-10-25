import React, { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { InsightIcon } from './insightIcons.jsx';
import { referenceParts, sourceLabel, sourceUrl } from './sourceReferences.js';
import './sourceReferences.css';
const useClientLayoutEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

export function SourceReference({ source }) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [position, setPosition] = useState({ left:12, top:12 });
  const button = useRef(null);
  const popup = useRef(null);
  const timer = useRef(null);
  const ignoreFocus = useRef(false);
  const id = useId();
  const label = sourceLabel(source);
  const url = sourceUrl(source.url);
  const cancelClose = () => clearTimeout(timer.current);
  const close = () => { setOpen(false); setPinned(false); };
  const dismiss = () => { close(); ignoreFocus.current = true; button.current?.focus(); ignoreFocus.current = false; };
  const leave = () => { if (!pinned) timer.current = setTimeout(() => setOpen(false), 180); };
  useEffect(() => () => clearTimeout(timer.current), []);
  useClientLayoutEffect(() => {
    if (!open) return;
    const rect = button.current.getBoundingClientRect();
    const height = popup.current.getBoundingClientRect().height;
    const width = Math.min(360, window.innerWidth - 24);
    setPosition({ left:Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)),
      top:Math.max(12, rect.bottom + height + 8 > window.innerHeight - 12 ? rect.top - height - 8 : rect.bottom + 8) });
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!button.current?.contains(event.target) && !popup.current?.contains(event.target)) close(); };
    const escape = event => {
      if (event.key === 'Escape') { event.stopPropagation(); event.preventDefault(); dismiss(); }
    };
    const scroll = event => { if (!popup.current?.contains(event.target)) close(); };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape, true);
    document.addEventListener('scroll', scroll, true);
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape, true);
      document.removeEventListener('scroll', scroll, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);
  return <>
    <button ref={button} type="button" className={`source-reference source-reference-${source.kind || 'fact'}`}
      aria-label={`Источник: ${label}`} aria-expanded={open} aria-controls={open ? id : undefined} aria-haspopup="dialog"
      onMouseEnter={() => { cancelClose(); setOpen(true); }} onMouseLeave={leave}
      onFocus={() => { if (!ignoreFocus.current) { cancelClose(); setOpen(true); } }}
      onBlur={event => { if (!popup.current?.contains(event.relatedTarget)) leave(); }}
      onClick={event => {
        cancelClose();
        if (pinned) close(); else {
          setPinned(true); setOpen(true);
          if (event.detail === 0) requestAnimationFrame(() => popup.current?.querySelector('button, a')?.focus());
        }
      }}>
      <InsightIcon name="document" size={12} /><span>{label}</span>
    </button>
    {open && createPortal(<div className="source-reference-popup" ref={popup} style={position} id={id}
      role="dialog" aria-label={`Источник: ${label}`} onMouseEnter={cancelClose} onMouseLeave={leave}
      onFocusCapture={cancelClose} onBlurCapture={event => {
        if (!popup.current?.contains(event.relatedTarget) && !button.current?.contains(event.relatedTarget)) close();
      }}>
      <div className="source-reference-heading"><strong>{label}</strong>
        <button type="button" aria-label="Закрыть источник" onClick={dismiss}><InsightIcon name="close" size={16} /></button></div>
      {source.date && <small>{source.date}</small>}
      <p>{source.text || source.snippet || 'Текст источника не сохранён.'}</p>
      {url && <a href={url} target="_blank" rel="noreferrer">Открыть оригинал ↗</a>}
    </div>, document.body)}
  </>;
}

export function CitedText({ text, sources, pending = false }) {
  return referenceParts(text || '', sources).map((part, index) => part.id
    ? part.source ? <SourceReference key={`${index}:${part.id}`} source={part.source} />
      : <span key={index} className="source-reference-unavailable" title={pending ? 'Источник появится после проверки ответа' : 'Материал с таким идентификатором отсутствует в сохранённом снимке'}>
        {pending ? 'Источник…' : 'Источник недоступен'}</span>
    : <React.Fragment key={index}>{part.text}</React.Fragment>);
}
