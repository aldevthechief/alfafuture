import { companyQuestions, saveAnswer } from './feedback.js';

// Compatible with demo-precomputed's saved condition IDs.
export function conditionId(key) {
  let a = 0x811c9dc5, b = 0x01000193;
  for (let k = 0; k < key.length; k++) {
    a = Math.imul(a ^ key.charCodeAt(k), 16777619) >>> 0;
    b = Math.imul(b + key.charCodeAt(k), 2246822519) >>> 0;
  }
  return 'q_' + (a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0')).slice(0, 12);
}

export function conditionQuestion(insightId, text, questions = []) {
  const normalize = value => value.trim().replace(/\s+/g, ' ').replace(/[.!?]+$/, '').toLocaleLowerCase('ru-RU');
  return questions.find(question => normalize(question.text) === normalize(text))
    || { id:conditionId(`${insightId}|${text}`), text };
}

// One server-backed form per open insight; no local success cache or cross-user state.
export function createConditionAnswers(company, role, onSaved = () => {}) {
  let state = { loading:true, error:'', items:{}, questions:[] }, loading;
  const listeners = new Set();
  const publish = next => { state = next; listeners.forEach(listener => listener()); };
  const update = (id, change) => publish({ ...state, items:{ ...state.items, [id]:{ ...state.items[id], ...change } } });
  return {
    getSnapshot: () => state,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    load() {
      if (loading) return loading;
      publish({ ...state, loading:true, error:'' });
      loading = companyQuestions(company, role).then(questions => {
        const items = {};
        for (const q of questions) if (q.answer) items[q.id] = { choice:q.answer.verdict, text:q.answer.text };
        publish({ loading:false, error:'', items, questions });
      }).catch(error => {
        publish({ ...state, loading:false, error:error.message || 'Не удалось загрузить уточнения.' });
      }).finally(() => { loading = null; });
      return loading;
    },
    async pick(id, text, verdict) {
      if (state.loading || state.error || state.items[id]?.busy) return;
      update(id, { busy:true, error:'', pending:verdict });
      let saved;
      try {
        saved = await saveAnswer(company, { id, text }, { verdict });
      } catch (error) {
        update(id, { busy:false, error:error.message || 'Не удалось сохранить ответ.' });
        return false;
      }
      update(id, { choice:saved?.verdict || verdict, text:saved?.text || '', busy:false, error:'', pending:null });
      onSaved();
      return true;
    },
  };
}
