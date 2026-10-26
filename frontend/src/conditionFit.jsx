import React, { useEffect, useMemo, useSyncExternalStore } from 'react';
import { Button } from '@alfalab/core-components/button';
import { FEEDBACK_ENABLED } from './feedback.js';
import { conditionQuestion, createConditionAnswers } from './conditionAnswers.js';

const FIT = { yes:'Про нас', no:'Не про нас' };

export function ConditionFit({ state = {}, disabled, onPick }) {
  return <div className="fit" role="group" aria-label="Подходит ли условие вашей компании" aria-busy={!!state.busy}>
    {state.choice && <span className="fit-thanks" role="status">Ответ сохранён · {FIT[state.choice] || 'Есть уточнение'}
      {state.choice === 'other' && state.text && `: ${state.text}`}</span>}
    <div className="fit-actions">{Object.entries(FIT).map(([verdict, label]) =>
      <Button key={verdict} size={32} view="secondary" disabled={disabled || state.busy}
        aria-pressed={state.choice === verdict} onClick={() => onPick(verdict)}>{label}</Button>)}</div>
    {state.busy && <span className="fit-note" role="status">Сохраняем ответ…</span>}
    {state.error && <div className="fit-error"><span role="alert">Не удалось сохранить ответ: {state.error}</span>
      <Button size={24} view="text" disabled={disabled || state.busy} onClick={() => onPick(state.pending)}>Повторить</Button></div>}
  </div>;
}

export function InsightConditions({ company, role, insightId, items }) {
  const form = useMemo(() => createConditionAnswers(company, role, () => {
    window.dispatchEvent(new CustomEvent('cabinet:profile-answer', { detail:{ company } }));
  }), [company, role]);
  const state = useSyncExternalStore(form.subscribe, form.getSnapshot, form.getSnapshot);
  const enabled = FEEDBACK_ENABLED && !!company && !!role;
  useEffect(() => { if (enabled) form.load(); }, [form, enabled]);
  return <>
    {enabled && state.loading && <p className="fit-note" role="status">Загружаем ваши ответы…</p>}
    {enabled && state.error && <div className="fit-error"><span role="alert">Не удалось загрузить ответы: {state.error}</span>
      <Button size={24} view="text" onClick={() => form.load()}>Повторить загрузку</Button></div>}
    <ul className="cond">{items.map(text => {
      const question = conditionQuestion(insightId, text, state.questions);
      const id = question.id;
      return <li key={id}>{text}{enabled && <ConditionFit state={state.items[id]} disabled={state.loading || !!state.error}
        onPick={verdict => form.pick(id, question.text, verdict)} />}</li>;
    })}</ul>
  </>;
}
