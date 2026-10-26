export const insightIdentity = insight => JSON.stringify([insight.job_id || '',insight.id]);
export const notificationScope = (user,company,role) => `alfa-insight-views:${JSON.stringify([user,company,role])}`;

export function readInsightViews(key) {
  try {
    const values=JSON.parse(localStorage.getItem(key) || '[]');
    return Array.isArray(values) ? values.filter(value=>typeof value==='string').slice(-400) : [];
  } catch { return []; }
}

export function mergeInsightViews(previous,insights) {
  return [...new Set([...previous,...insights.map(insightIdentity)])].slice(-400);
}

export const normalizeSearch = value => String(value ?? '').toLocaleLowerCase('ru-RU').replaceAll('ё','е');
export function matchesSearch(text, query) {
  const terms=normalizeSearch(query).trim().split(/\s+/).map(term=>term.replace(/^#/, '')).filter(Boolean);
  const haystack=normalizeSearch(text);
  return terms.every(term=>haystack.includes(term));
}

export function insightTags(insight) {
  return [insight.channel, ...(insight.tags || []).map(tag=>typeof tag==='string' ? tag : tag.name),
    (insight.counts?.risks ?? insight.risks?.length) > 0 && 'Риски',
    (insight.counts?.opportunities ?? insight.opportunities?.length) > 0 && 'Возможности'].filter(Boolean);
}

export function toggleInsightTag(selected, tag) {
  if (tag==='all') return [];
  return selected.includes(tag) ? selected.filter(value=>value!==tag) : [...selected,tag];
}

export function findInsights(insights,query,filter=[]) {
  const selected=(Array.isArray(filter) ? filter : [filter]).filter(tag=>tag!=='all');
  const copy=value=>typeof value==='string' ? value : value?.text || value?.title || '';
  return insights.filter(insight=>{
    const tags=insightTags(insight);
    const hasTag=tag=>tags.includes(({risks:'Риски',opps:'Возможности'})[tag] || tag);
    if (selected.length && !selected.some(hasTag)) return false;
    const text=[...tags, ...[insight,insight.compact].filter(Boolean).flatMap(item=>[
      item.title,item.lead,item.why_now,item.ai_conclusion,
      ...['what_happened','why_it_matters','risks','opportunities','bank_view','actions','hypotheses'].flatMap(key=>(item[key] || []).map(copy)),
    ])].filter(Boolean).join(' ');
    return matchesSearch(text,query);
  });
}
