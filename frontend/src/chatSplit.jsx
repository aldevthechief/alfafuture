import React, { Children, useEffect, useRef, useState } from 'react';

const DEFAULT = 58;
const KEY = 'alfa-insight-chat-split:v1';
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const savedShare = () => {
  try {
    const value = Number(localStorage.getItem(KEY));
    return Number.isFinite(value) && value >= 40 && value <= 65 ? value : DEFAULT;
  } catch { return DEFAULT; }
};
const persist = value => { try { localStorage.setItem(KEY, String(value)); } catch {} };

export function ChatSplit({ reading, children }) {
  const root = useRef(null);
  const drag = useRef(null);
  const [preferred, setPreferred] = useState(savedShare);
  const [width, setWidth] = useState(0);
  const [resizing, setResizing] = useState(false);
  const available = width - 12;
  // The chat gets 40–65% of the usable width, with 420px/320px minimums on desktop.
  const min = available >= 740 ? Math.max(40, 420 / available * 100) : 40;
  const max = available >= 740 ? Math.min(65, (1 - 320 / available) * 100) : 65;
  const share = clamp(preferred, min, max);
  const [chat, side] = Children.toArray(children);

  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      const next = entries[0].contentRect.width;
      if (next > 0) setWidth(next);
    });
    observer.observe(root.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (resizing) document.documentElement.classList.add('insight-split-resizing');
    return () => document.documentElement.classList.remove('insight-split-resizing');
  }, [resizing]);

  function change(value) {
    const next = clamp(value, min, max);
    setPreferred(next); persist(next);
  }
  function finish(event, cancel = false) {
    if (!drag.current || drag.current.id !== event.pointerId) return;
    const next = cancel ? drag.current.start : drag.current.last;
    drag.current = null;
    setResizing(false); setPreferred(next); persist(next);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  return <div ref={root} className={`chat-workspace${reading ? ' is-reading' : ''}${resizing ? ' is-resizing' : ''}`}
    style={{ '--chat-share': `${share}fr`, '--reader-share': `${100 - share}fr` }}>
    {chat}
    {reading && <div className="chat-splitter" role="separator" tabIndex={0} aria-orientation="vertical"
      aria-label="Ширина чата и текста инсайта" aria-valuemin={Math.round(min)} aria-valuemax={Math.round(max)}
      aria-valuenow={Math.round(share)} aria-valuetext={`Чат ${Math.round(share)}%, текст ${100 - Math.round(share)}%`}
      title="Потяните для изменения ширины. Стрелки ← → — настроить, двойной клик — сбросить"
      onPointerDown={event => {
        if (event.button !== 0) return;
        event.preventDefault(); event.currentTarget.focus({ preventScroll: true });
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { id: event.pointerId, start: preferred, last: share };
        setResizing(true);
      }}
      onPointerMove={event => {
        if (drag.current?.id !== event.pointerId) return;
        const rect = root.current.getBoundingClientRect();
        const next = clamp((event.clientX - rect.left - 6) / (rect.width - 12) * 100, min, max);
        drag.current.last = next; setPreferred(next);
      }}
      onPointerUp={finish} onPointerCancel={event => finish(event, true)} onLostPointerCapture={finish}
      onDoubleClick={() => change(DEFAULT)}
      onKeyDown={event => {
        const step = event.shiftKey ? 5 : 2;
        const next = { ArrowLeft: share - step, ArrowRight: share + step, Home: min, End: max, Enter: DEFAULT }[event.key];
        if (next === undefined) return;
        event.preventDefault(); change(next);
      }}><span aria-hidden="true" /></div>}
    {side}
  </div>;
}
