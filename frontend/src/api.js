// Backend client (docs/API.md): POST /v1/analyses → poll GET /v1/jobs/{id}; the job result is mapped to the feed.json shape
// the widgets already read. VITE_API_URL sets the base, VITE_API_URL=off keeps the demo feed.json only.
import { ALLOW_INFERENCE } from './cabinetMode.js';
const ENV = import.meta.env || {};
const BASE = (ENV.VITE_API_URL ?? (ENV.DEV ? '.' : 'http://127.0.0.1:8000')).replace(/\/$/, '');
export const USER_ID = ENV.VITE_USER_ID || 'demo-user';
export const API_ENABLED = BASE !== 'off' && BASE !== '';
// extra AnalysisRequest fields (JSON), e.g. VITE_ANALYSIS='{"engine":"judge","analytics_cache_only":true}'
const ANALYSIS = JSON.parse(ENV.VITE_ANALYSIS || '{}');
const DONE = ['succeeded', 'partial', 'failed', 'interrupted'];
const sleep = ms => new Promise(r => setTimeout(r, ms));

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
export async function requestJson(base, path, init = {}) {
  const r = await fetch(base + path, { ...init,
    headers: { 'Content-Type': 'application/json', ...init.headers, 'X-User-ID': USER_ID } });
  if (!r.ok) {
    const detail = (await r.json().catch(() => ({}))).detail;
    throw new ApiError(r.status, typeof detail === 'string' ? detail : r.statusText);
  }
  return r.json();
}
const call = (path, init) => requestJson(BASE, path, init);

// latest analysis of this company and role: the finished one to show, and the one still running (if newer)
export async function latestJobs(company, role) {
  const { analyses } = await call('/v1/jobs?include_results=false');
  const mine = analyses.filter(j => j.request.company_id === company && j.request.role === role)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const personal = mine.filter(j => j.request.user_id === USER_ID);
  return { done: personal.find(j => ['succeeded', 'partial'].includes(j.status))
      || mine.find(j => !j.request.user_id && ['succeeded', 'partial'].includes(j.status)),
    running: personal.find(j => !DONE.includes(j.status)) };
}

export const startAnalysis = (company, role, options = {}) => {
  if (!ALLOW_INFERENCE) return Promise.reject(new Error('В этой версии доступны только готовые кабинеты.'));
  return call('/v1/analyses', { method: 'POST', body: JSON.stringify({ ...ANALYSIS, ...options, company_id: company, role, user_id: USER_ID }) });
};

export const cabinetAccounts = () => call('/v1/cabinet/accounts');
export const cabinetFeed = (company, role) => call(`/v1/cabinet/feed?company_id=${encodeURIComponent(company)}&role=${encodeURIComponent(role)}`);
export const searchCompanies = query => call(`/v1/cabinet/companies?q=${encodeURIComponent(query)}`);

export async function waitJob(id, cancelled, every = 3000) {
  for (;;) {
    const job = await call(`/v1/jobs/${id}?include_result=false`);
    if (DONE.includes(job.status) || cancelled()) return job;
    await sleep(every);
  }
}

const facts = {};
export const companyFacts = id => facts[id] ??= call(`/v1/companies/${id}`).then(r => r.facts).catch(() => []);

const BLOCKS = ['what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view'];
const claims = list => (list || []).map(c => ({ id: c.id, text: c.text, sources: c.sources || [] }));

export const jobPreviews = id => call(`/v1/jobs/${id}/insights`);
export const compactOf = id => call(`/v1/jobs/${encodeURIComponent(id)}/compact`);

const openedInsights = new Map();
export function loadInsight(card, company) {
  if (!card.preview) return Promise.resolve(card);
  const key = JSON.stringify([USER_ID, company, card.job_id, card.result_index, card.id]);
  if (!openedInsights.has(key)) {
    const promise = call(`/v1/jobs/${card.job_id}/insights/${card.result_index}?insight_id=${encodeURIComponent(card.id)}`).then(async job => {
      const row = job.result.results[0];
      const legacyFacts = !row.evidence && !row.evidence_facts ? await companyFacts(company) : [];
      return { ...toFeed(job, legacyFacts).insights[0], id: card.id, roleLabel: card.roleLabel };
    }).catch(error => {
      if (openedInsights.get(key) === promise) openedInsights.delete(key);
      throw error;
    });
    openedInsights.set(key, promise);
    // A long browsing session must not retain full evidence for every previously opened card.
    while (openedInsights.size > 12) openedInsights.delete(openedInsights.keys().next().value);
  }
  return openedInsights.get(key);
}

// job.result.results (pipeline rows, contract v2) → { computed_at, insights, sources } as in feed.json
export function toFeed(job, companyFacts = []) {
  const rows = (job.result?.results || []).filter(r => r.status === 'ready' && r.insight);
  const allSources = {};
  const insights = rows.map((r, index) => {
    const sources = {};
    const add = (id, s) => { if (id && !sources[id]) sources[id] = s; };
    if (r.evidence) {
      (r.evidence.facts || []).forEach(f => add(f.id, { kind: 'fact', text: f.text }));
      (r.evidence.calculations || []).forEach(f => add(f.id, { kind: 'fact', text: f.text }));
      (r.evidence.news || []).forEach(n => add(`news:${n.id}`, { kind:'news', outlet:n.source,
        date:n.date, title:n.title, url:n.url, snippet:n.text?.slice(0, 300) }));
      (r.evidence.bank || []).forEach(b => add(b.id, { kind:'bank', title:b.report, date:b.date, page:b.page, snippet:b.text?.slice(0, 300) }));
    } else if (r.evidence_facts) {
      r.evidence_facts.forEach((text, n) => add(`fact:${n + 1}`, { kind:'fact', text }));
    } else {
      companyFacts.forEach((f, n) => {
        add(f.id, { kind: 'fact', text: f.text });
        add(`fact:${n + 1}`, { kind: 'fact', text: f.text });
      });
    }
    const i = r.insight;
    const bank = r.source === 'bank_analytics' || i.origin === 'bank';
    Object.entries(r.news_references || {}).forEach(([id, n]) =>
      add(id, { kind: 'news', outlet: n.source || 'Новость', date: n.date, title: n.title, url: n.url, snippet: '' }));
    (r.news || []).forEach(n =>  // loop engine: the related news it read
      add(`news:${n.id}`, { kind: 'news', outlet: n.source || 'Новость', date: n.date, title: n.title, url: n.url, snippet: '' }));
    // facts and estimates computed for this insight (loop: verdict.brief, judge: calculations)
    const brief = r.verdict?.brief || {};
    [...(brief.facts || []), ...(brief.hypotheses || []), ...(r.calculations || [])].forEach(f => f.id && add(f.id,
      { kind: 'fact', text: [f.label, f.value !== '' && f.value != null && `${f.value}${f.unit ? ' ' + f.unit : ''}`].filter(Boolean).join(': ') }));
    [...(r.judge_bank || []), ...(r.calculation_bank || []), ...(r.bank || [])].forEach(b =>
      add(b.id || `bank:${b.report}:${b.page}`, { kind: 'bank', title: b.report, date: b.date, page: b.page, snippet: b.text?.slice(0, 300) }));
    BLOCKS.flatMap(k => i[k] || []).forEach(c => (c.evidence || []).forEach(e => {
      const kind = e.source.startsWith('bank:') ? 'bank' : 'news';
      add(e.source, { kind, outlet: 'Новость', title: e.title, date: e.date, page: e.page, url: e.url, snippet: e.quote });
      if (sources[e.source] && !sources[e.source].snippet) sources[e.source].snippet = e.quote;
      // the claim may cite the report page while the evidence names the fragment
      c.sources.filter(s => !sources[s]).forEach(s => add(s, { ...sources[e.source] }));
    }));
    BLOCKS.flatMap(k => i[k] || []).flatMap(c => c.sources || []).forEach(s =>
      s.startsWith('fact:') && add(s, { kind: 'fact', text: 'данные компании' }));
    const news = r.news_references?.[`news:${r.news_id}`];
    Object.assign(allSources, sources);
    return {
      id: i.id || `row-${index}-${r.news_id ?? 'bank'}`,
      job_id: job.id, pipeline_result: r, sources,
      published: i.event_date || news?.date || job.finished_at,
      channel: i.channel || (bank ? 'Аналитика Банка' : 'Новости'),
      relevance: r.stage4?.relevance?.score,
      title: i.title, role: r.evidence?.role || job.request?.role || i.role,
      why_now: i.why_now || '',
      ...Object.fromEntries(BLOCKS.map(k => [k, claims(i[k])])),
      actions: i.actions || [], ai_conclusion: i.ai_conclusion || '',
      confidence: i.confidence || 'medium', bank_product: i.bank_product || '',
      impact_level: i.impact?.level ?? null,
      not_bank_position: i.not_bank_position, warnings: i.review?.warnings || [],
      trace: i.trace || [],
    };
  });
  return { computed_at: job.finished_at || job.created_at, insights, sources: allSources, status: job.status,
           errors: job.result?.errors || (job.error ? [{ component: 'job', message: job.error }] : []) };
}
