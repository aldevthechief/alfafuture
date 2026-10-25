import { ApiError, requestJson, USER_ID } from './api.js';
import { ALLOW_INFERENCE } from './cabinetMode.js';

const ENV = import.meta.env || {};
const BASE = (ENV.VITE_FEEDBACK_API ?? (ENV.DEV ? '/feedback' : 'http://127.0.0.1:8001')).replace(/\/$/, '');
export const FEEDBACK_ENABLED = ALLOW_INFERENCE && BASE !== 'off' && BASE !== '';
const call = (path, body) => requestJson(BASE, path, body === undefined ? {} : { method:'POST', body:JSON.stringify(body) });
const registrations = new Map();
const PROFILE_ENABLED = BASE !== 'off' && BASE !== '';
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

export const askQuestion = (company, insight, question, requestId, onText, selectedText = '', selectedSource = 'insight') => withInsight(company, insight, record =>
  typeof onText === 'function'
    ? streamQuestion(record.id, { question, request_id: requestId, ...(selectedText ? { selected_text: selectedText, ...(selectedSource === 'answer' ? { selected_source:'answer' } : {}) } : {}) }, onText)
    : call(`/v1/insights/${record.id}/ask`, { question, request_id:requestId, ...(selectedText ? { selected_text: selectedText, ...(selectedSource === 'answer' ? { selected_source:'answer' } : {}) } : {}) }));

async function streamQuestion(id, body, onText) {
  return streamAnswer(`/v1/insights/${id}/ask/stream`, body, onText);
}

export async function streamAnswer(path, body, onText) {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-ID': USER_ID, Accept: 'text/event-stream' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(response.status, data.detail || response.statusText);
  }
  if (!response.body) throw new Error('Сервис не вернул поток ответа. Повторите запрос.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', answer;
  const consume = frame => {
    const lines = frame.split('\n');
    const name = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
    const json = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
    if (!json) return;
    const data = JSON.parse(json);
    if (name === 'text') onText(data.text);
    if (name === 'complete') answer = data;
    if (name === 'error') throw new ApiError(data.status || 502, data.message);
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        consume(buffer.slice(0, boundary)); buffer = buffer.slice(boundary + 2);
      }
      if (done) { if (buffer.trim()) consume(buffer); break; }
    }
    if (!answer?.analysis) throw new Error('Соединение прервалось до сохранения ответа. Повторите запрос.');
    return answer;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function answerSources(record, item, cardSources = {}) {
  // A company chat freezes a new snapshot per message; an insight chat uses its registered snapshot.
  const evidence = item.evidence || record?.evidence;
  const sources = evidence ? {} : { ...cardSources };
  (evidence?.facts || []).forEach(f => { sources[f.id] = { kind:'fact', text:f.text }; });
  (evidence?.calculations || []).forEach(f => { sources[f.id] = { kind:'fact', text:f.text }; });
  (evidence?.news || []).forEach(n => { sources[`news:${n.id}`] = { kind:'news', title:n.title, url:n.url, date:n.date, snippet:n.text }; });
  (evidence?.bank || []).forEach(b => { sources[b.id] = { kind:'bank', title:b.report, page:b.page, date:b.date, snippet:b.text }; });
  (item.analysis?.retrieved_sources || []).forEach(s => { sources[s.id] = {
    kind:s.origin === 'news_pool' ? 'news' : s.origin === 'bank_analytics' ? 'bank' : 'fact',
    title:s.title || (s.origin === 'company_data' ? 'Данные компании' : s.text.split('\n')[0]),
    url:s.url, page:s.page, date:s.date, snippet:s.text,
  }; });
  return sources;
}

export const openCompanyChat = (company, role) => call('/v1/company-chats', { company_id:company, role });
export const companyChatHistory = (id, before) => call(`/v1/company-chats/${encodeURIComponent(id)}/messages${before ? `?before=${encodeURIComponent(before)}` : ''}`);
export const askCompanyChat = (id, question, requestId, onText, selectedText = '') =>
  streamAnswer(`/v1/company-chats/${encodeURIComponent(id)}/ask/stream`,
    { question, request_id:requestId, ...(selectedText ? { selected_text:selectedText } : {}) }, onText);

export const companyQuestions = (company, role) => !PROFILE_ENABLED ? Promise.resolve([])
  : call(`/v1/companies/${encodeURIComponent(company)}/questions?role=${encodeURIComponent(role)}`);

export async function saveAnswer(company, question, answer) {
  if (answer?.verdict === 'skip') return;
  if (!FEEDBACK_ENABLED) throw new Error('Сохранение уточнений в этой версии недоступно.');
  const path = `/v1/companies/${encodeURIComponent(company)}/answers/${encodeURIComponent(question.id)}`;
  if (!answer) {
    const response = await fetch(BASE + path, { method: 'DELETE', headers: { 'X-User-ID': USER_ID } });
    if (!response.ok) throw new ApiError(response.status, response.statusText);
    return;
  }
  return requestJson(BASE, path, { method: 'PUT', body: JSON.stringify({
    statement: question.text, verdict: answer.verdict, text: answer.text || '',
  }) });
}
