import React, { useEffect, useState } from 'react';
import { InsightIcon } from './insightIcons.jsx';

// Only one text surface may own an excerpt: never mix messages, metadata or source lists.
export function useExcerptSelection({ root, anchor = root, selector, enabled }) {
  const [selection, setSelection] = useState(null);
  useEffect(() => {
    if (!enabled) { setSelection(null); return; }
    const clear = () => setSelection(null);
    const capture = () => {
      const selected = window.getSelection();
      if (!selected?.rangeCount || selected.isCollapsed) return clear();
      const range = selected.getRangeAt(0);
      const container = root.current;
      if (!container?.contains(range.startContainer) || !container.contains(range.endContainer)) return clear();
      if (selector) {
        const element = node => node.nodeType === 1 ? node : node.parentElement;
        const start = element(range.startContainer)?.closest(selector);
        if (!start || start !== element(range.endContainer)?.closest(selector)) return clear();
      }
      const text = selected.toString().trim();
      if (!text) return clear();
      const rect = range.getBoundingClientRect();
      const bounds = anchor.current.getBoundingClientRect();
      const visible = container.getBoundingClientRect();
      const bottom = Math.min(visible.bottom, bounds.bottom);
      setSelection({ text, top: Math.max(12, Math.min(rect.bottom + 8, bottom - 48) - bounds.top) });
    };
    const escape = event => { if (event.key === 'Escape') clear(); };
    const container = root.current;
    document.addEventListener('selectionchange', capture);
    document.addEventListener('keydown', escape);
    container?.addEventListener('scroll', clear, true);
    window.addEventListener('resize', clear);
    return () => {
      document.removeEventListener('selectionchange', capture);
      document.removeEventListener('keydown', escape);
      container?.removeEventListener('scroll', clear, true);
      window.removeEventListener('resize', clear);
    };
  }, [root, anchor, selector, enabled]);
  return [selection, () => setSelection(null)];
}

export function ExcerptAction({ selection, onQuote }) {
  if (!selection) return null;
  return <div className="insight-selection-action" style={{ top:selection.top }}>
    <button type="button" onMouseDown={event => event.preventDefault()}
      disabled={selection.text.length > 2000} onClick={() => {
        onQuote(selection.text);
        window.getSelection()?.removeAllRanges();
      }}><InsightIcon name="chat" size={16} />
      {selection.text.length > 2000 ? 'Выделите до 2000 символов' : 'Спросить об этом'}
    </button>
  </div>;
}
