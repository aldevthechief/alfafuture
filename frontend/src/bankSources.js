export function bankReports(ids, sources) {
  const reports = new Map();
  for (const id of ids) {
    const source = sources[id];
    if (source?.kind !== 'bank') continue;
    const name = (source.title || 'Отчёт Альфа-Банка').replace(/\s*\([^)]*\)\s*$/, '');
    const key = JSON.stringify([name, source.date || '']);
    const report = reports.get(key) || { key, name, date:source.date, pages:new Set() };
    if (source.page != null && source.page !== '') report.pages.add(String(source.page));
    reports.set(key, report);
  }
  return [...reports.values()].map(report => ({ ...report,
    pages:[...report.pages].sort((a, b) => a.localeCompare(b, 'ru', { numeric:true })),
  }));
}
