import { useCallback, useEffect, useRef, useState } from 'react';
import { companyProfile } from './api.js';
import { companyQuestions, FEEDBACK_ENABLED, saveAnswer } from './feedback.js';

export function useCompanyProfile(ctx, feed, enabled) {
  const [state, setState] = useState({ profile:null, questions:[], answers:{}, loading:false, busy:false, error:'' });
  const [retry, setRetry] = useState(0);
  const alive = useRef(true);
  const saving = useRef(false);
  const version = JSON.stringify((feed?.insights || []).map(card => [card.job_id, card.id]).sort());
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setState(previous => ({ ...previous, loading:true, error:'' }));
    Promise.allSettled([companyProfile(ctx.company), companyQuestions(ctx.company, ctx.role)]).then(([profile, questions]) => {
      if (cancelled) return;
      setState(previous => {
        const items = questions.status === 'fulfilled' ? questions.value : previous.questions;
        const given = Object.fromEntries(items.filter(q => q.answer).map(q => [q.id, { verdict:q.answer.verdict, text:q.answer.text }]));
        const skipped = Object.fromEntries(Object.entries(previous.answers).filter(([id, answer]) => answer?.verdict === 'skip' && items.some(q => q.id === id)));
        return { ...previous, loading:false, profile:profile.status === 'fulfilled' ? profile.value.profile : previous.profile,
          questions:items, answers:questions.status === 'fulfilled' ? { ...skipped, ...given } : previous.answers,
          error:[profile.status === 'rejected' && `Профиль: ${profile.reason.message}`,
            questions.status === 'rejected' && `Вопросы профиля: ${questions.reason.message}`].filter(Boolean).join('. ') };
      });
    });
    return () => { cancelled = true; };
  }, [ctx.company, ctx.role, enabled, version, retry]);

  const answer = useCallback(async (id, value) => {
    if (saving.current) return false;
    const question = state.questions.find(q => q.id === id);
    if (!question) return false;
    saving.current = true;
    setState(previous => ({ ...previous, busy:true, error:'' }));
    try {
      await saveAnswer(ctx.company, question, value);
      if (!alive.current) return true;
      setState(previous => {
        const answers = { ...previous.answers };
        if (value) answers[id] = value; else delete answers[id];
        return { ...previous, answers };
      });
      return true;
    } catch (error) {
      if (alive.current) setState(previous => ({ ...previous, error:`Не удалось сохранить уточнение: ${error.message}` }));
      return false;
    } finally {
      saving.current = false;
      if (alive.current) setState(previous => ({ ...previous, busy:false }));
    }
  }, [ctx.company, state.questions]);
  return [{ ...state, writable:FEEDBACK_ENABLED, onRetry:() => setRetry(value => value + 1) }, answer];
}
