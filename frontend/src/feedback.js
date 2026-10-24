import { ApiError, requestJson, USER_ID } from './api.js';

const ENV = import.meta.env || {};
const BASE = (ENV.VITE_FEEDBACK_API ?? 'http://127.0.0.1:8001').replace(/\/$/, '');
export const FEEDBACK_ENABLED = BASE !== 'off' && BASE !== '';
const call = (path, body) => requestJson(BASE, path, body === undefined ? {} : { method:'POST', body:JSON.stringify(body) });
const registrations = new Map();
const keyOf = (company, i) => JSON.stringify([USER_ID, company, i.role, i.job_id, i.id]);

export function registerInsight(company, insight) {
  if (!FEEDBACK_ENABLED || !insight.pipeline_result || !insight.job_id) {
    return Promise.reject(new Error('Обсуждение доступно после запуска анализа.'));
  }
  const key = keyOf(company, insight);
  if (!registrations.has(key)) {
    const promise = call('/v1/insights/import', { company_id:company, results:[insight.pipeline_result],
      registration_key:`${insight.job_id}:${insight.id}` }).then(records => {
      if (records.length !== 1) throw new Error('Не удалось подключить исходные материалы инсайта. Повторите подключение.');
      return records[0];
    }).catch(error => { registrations.delete(key); throw error; });
    registrations.set(key, promise);
  }
  return registrations.get(key);
}

async function withInsight(company, insight, action) {
  const record = await registerInsight(company, insight);
  try { return await action(record); }
  catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error;
    registrations.delete(keyOf(company, insight));
    return action(await registerInsight(company, insight));
  }
}

export const discussionHistory = (company, insight, offset = 0) => withInsight(company, insight, async record => ({
  record, items:await call(`/v1/insights/${record.id}/feedback?limit=50&offset=${offset}`),
}));

export const sendReaction = (company, insight, body, requestId) => withInsight(company, insight, record =>
  call(`/v1/insights/${record.id}/feedback`, { ...body, request_id:requestId }));

export const askQuestion = (company, insight, question, requestId) => withInsight(company, insight, record =>
  call(`/v1/insights/${record.id}/ask`, { question, request_id:requestId }));

export function answerSources(record, item, cardSources = {}) {
  const sources = { ...cardSources };
  const evidence = record?.evidence;
  (evidence?.facts || []).forEach(f => { sources[f.id] = { kind:'fact', text:f.text }; });
  (evidence?.calculations || []).forEach(f => { sources[f.id] = { kind:'fact', text:f.text }; });
  (evidence?.news || []).forEach(n => { sources[`news:${n.id}`] = { kind:'news', title:n.title, url:n.url, snippet:n.text }; });
  (evidence?.bank || []).forEach(b => { sources[b.id] = { kind:'bank', title:b.report, page:b.page, snippet:b.text }; });
  (item.analysis?.retrieved_sources || []).forEach(s => { sources[s.id] = { title:s.id, snippet:s.text }; });
  return sources;
}
