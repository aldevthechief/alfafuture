// Presentation only: reuse the writer's risks/opportunities and impact; no inference or requests.
export function insightSignal(insight) {
  const risks = insight.counts?.risks ?? insight.risks?.length ?? 0;
  const opportunities = insight.counts?.opportunities ?? insight.opportunities?.length ?? 0;
  const high = (insight.impact_level ?? insight.impact?.level) === 'high';
  const signal = risks && opportunities
    ? { tone: 'negative', emoji: '⚠️', label: 'Риски и возможности', reason: 'Есть и риски, и возможности для компании. Красный акцент — повод проверить риски.' }
    : risks
      ? { tone: 'negative', emoji: '🐻', label: 'Риски для компании', reason: 'В анализе отмечены риски для компании.' }
      : opportunities
        ? { tone: 'positive', emoji: '📈', label: 'Возможности для компании', reason: 'В анализе отмечены возможности для компании, риски не выделены.' }
        : { tone: 'neutral', emoji: '📰', label: 'Без оценки направления', reason: 'В анализе не выделены риски или возможности; направление влияния не определено.' };
  return { ...signal, high };
}
