import React, { useRef } from 'react';
import { InsightIcon } from './insightIcons.jsx';

export function InsightTabs({ id, tab, onChange }) {
  const refs = useRef([]);
  const tabs = [['overview', 'Обзор инсайта', 'document'], ['chat', 'Чат с агентом', 'chat']];
  function navigate(event) {
    const current = tabs.findIndex(([key]) => key === tab);
    const next = { ArrowRight: (current + 1) % 2, ArrowLeft: (current + 1) % 2, Home: 0, End: 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    onChange(tabs[next][0]);
    refs.current[next]?.focus();
  }
  return <div className="workbench-tabs" role="tablist" aria-label="Материалы и обсуждение инсайта" onKeyDown={navigate}>
    {tabs.map(([key, label, icon], index) => <button key={key} ref={node => { refs.current[index] = node; }}
      type="button" role="tab" id={`${id}-${key}-tab`} aria-controls={`${id}-${key}-panel`}
      aria-selected={tab === key} tabIndex={tab === key ? 0 : -1} onClick={() => onChange(key)}>
      <InsightIcon name={icon} size={18} />{label}{key === 'chat' && <span className="workbench-ai-tag">AI</span>}
    </button>)}
  </div>;
}
