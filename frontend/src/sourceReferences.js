const escape = value => Array.from(value, ch => '.*+?^$()[]{}|'.includes(ch) || ch.charCodeAt(0) === 92
  ? String.fromCharCode(92) + ch : ch).join('');

export function referenceParts(text, sources = {}) {
  const known = Object.keys(sources).filter(id => /^(news|fact|bank|lookup|user):/.test(id))
    .sort((a, b) => b.length - a.length).map(escape);
  const pattern = new RegExp(String.raw`(?<![\p{L}\p{N}_:/])(?:${known.length ? `${known.join('|')}|` : ''}news:\d+|fact:[\p{L}\p{N}_:-]+|(?:bank|lookup|user):[\p{L}\p{N}_:/.-]+)(?![\p{L}\p{N}_:-])`, 'gu');
  const parts = [];
  let previous = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > previous) parts.push({ text:text.slice(previous, match.index) });
    const id = sources[match[0]] ? match[0] : match[0].replace(/[.!?]+$/, '');
    parts.push({ id, source:sources[id] });
    if (id.length < match[0].length) parts.push({ text:match[0].slice(id.length) });
    previous = match.index + match[0].length;
  }
  if (previous < text.length) parts.push({ text:text.slice(previous) });
  return parts;
}

export const sourceLabel = source => source.kind === 'fact' ? 'Данные компании'
  : source.kind === 'user' ? 'Ваше уточнение'
    : `${source.title || (source.kind === 'bank' ? 'Аналитика Банка' : 'Новость')}${source.page ? ` · с. ${source.page}` : ''}`;

export function sourceUrl(value) {
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}
