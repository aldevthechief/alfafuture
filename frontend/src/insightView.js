const CONFIDENCE = { high: ['green', 'Уверенность высокая'], medium: ['orange', 'Уверенность средняя'], low: ['grey', 'Уверенность низкая'] };
const dateFormatter = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'long' });
const day = value => {
  const date = value && new Date(value);
  return date && !isNaN(date) ? dateFormatter.format(date) : '';
};

// the screen version, or the same shape built from the full insight
export function view(i, c) {
  if (c) return c;
  const plain = list => (list || []).map(x => ({ text: x.text, sources: x.sources || [] }));
  return { title: i.title, lead: i.why_it_matters[0]?.text || '', facts: plain(i.what_happened), why: plain(i.why_it_matters),
           important_if: (i.hypotheses || []).map(x => x.text), risks: plain(i.risks), opportunities: plain(i.opportunities), actions: i.actions,
           bank_view: plain(i.bank_view) };
}


export { CONFIDENCE, day };
