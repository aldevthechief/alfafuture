// Backend client (docs/API.md): POST /v1/analyses → poll GET /v1/jobs/{id}; the job result is mapped to the feed.json shape
// the widgets already read. VITE_API_URL sets the base, VITE_API_URL=off keeps the demo feed.json only.
const BASE = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
export const API_ENABLED = BASE !== 'off' && BASE !== '';
// extra AnalysisRequest fields (JSON), e.g. VITE_ANALYSIS='{"engine":"loop","analytics_cache_only":true}'
const ANALYSIS = JSON.parse(import.meta.env.VITE_ANALYSIS || '{}');
const DONE = ['succeeded', 'partial', 'failed', 'interrupted'];
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function call(path, init) {
  const r = await fetch(BASE + path, init && { ...init, headers: { 'Content-Type': 'application/json' } });
  if (!r.ok) throw new Error(`${r.status} ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`);
  return r.json();
}

// latest analysis of this company and role: the finished one to show, and the one still running (if newer)
export async function latestJobs(company, role) {
  const { analyses } = await call('/v1/jobs');
  const mine = analyses.filter(j => j.request.company_id === company && j.request.role === role)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  return { done: mine.find(j => ['succeeded', 'partial'].includes(j.status)), running: mine.find(j => !DONE.includes(j.status)) };
}

export const startAnalysis = (company, role) =>
  call('/v1/analyses', { method: 'POST', body: JSON.stringify({ company_id: company, role, ...ANALYSIS }) });

export async function waitJob(id, cancelled, every = 3000) {
  for (;;) {
    const job = await call(`/v1/jobs/${id}`);
    if (DONE.includes(job.status) || cancelled()) return job;
    await sleep(every);
  }
}

const facts = {};
export const companyFacts = id => facts[id] ??= call(`/v1/companies/${id}`).then(r => r.facts).catch(() => []);

const BLOCKS = ['what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view'];
const claims = list => (list || []).map(c => ({ text: c.text, sources: c.sources || [] }));

// job.result.results (pipeline rows, contract v2) → { computed_at, insights, sources } as in feed.json
export function toFeed(job, companyFacts = []) {
  const rows = (job.result?.results || []).filter(r => r.status === 'ready' && r.insight);
  const sources = {};
  const add = (id, s) => { if (id && !sources[id]) sources[id] = s; };
  companyFacts.forEach((f, n) => {
    add(f.id, { kind: 'fact', text: f.text });
    add(`fact:${n + 1}`, { kind: 'fact', text: f.text });
  });
  const insights = rows.map(r => {
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
    return {
      id: i.id || `news-${r.news_id}`,
      published: i.event_date || news?.date || job.finished_at,
      channel: i.channel || (bank ? 'Аналитика Банка' : 'Новости'),
      relevance: r.stage4?.relevance?.score,
      title: i.title, role: i.role,
      why_now: i.why_now || '',
      ...Object.fromEntries(BLOCKS.map(k => [k, claims(i[k])])),
      actions: i.actions || [], ai_conclusion: i.ai_conclusion || '',
      confidence: i.confidence || 'medium', bank_product: i.bank_product || '',
      not_bank_position: i.not_bank_position, warnings: i.review?.warnings || [],
      trace: i.trace || [],
    };
  });
  return { computed_at: job.finished_at || job.created_at, insights, sources, status: job.status,
           errors: job.result?.errors || (job.error ? [{ component: 'job', message: job.error }] : []) };
}
