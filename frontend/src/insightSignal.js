// Presentation only: reuse the writer's risks/opportunities and impact; no inference or requests.
export function insightSignal(insight) {
  const risks = insight.counts?.risks ?? insight.risks?.length ?? 0;
  const opportunities = insight.counts?.opportunities ?? insight.opportunities?.length ?? 0;
  const high = (insight.impact_level ?? insight.impact?.level) === 'high';
  const signal = risks && opportunities
    ? { tone: 'negative', label: 'Риск и возможность', reason: 'Есть и риски, и возможности для компании. Красный акцент — повод проверить риски.' }
    : risks
      ? { tone: 'negative', label: 'Риск', reason: 'В анализе отмечены риски для компании.' }
      : opportunities
        ? { tone: 'positive', label: 'Возможность', reason: 'В анализе отмечены возможности для компании, риски не выделены.' }
        : { tone: 'neutral', label: 'Без оценки', reason: 'В анализе не выделены риски или возможности; направление влияния не определено.' };
  // Existing search/notification tiles use the icon; the tags and home widget use text only.
  const emoji = risks && opportunities ? '⚠️' : risks ? '🐻' : opportunities ? '📈' : '📰';
  return { ...signal, emoji, high };
}

// Separate, equal tags: a risk and an opportunity are two tags, not one merged label.
export function signalTags(insight) {
  const risks = insight.counts?.risks ?? insight.risks?.length ?? 0;
  const opportunities = insight.counts?.opportunities ?? insight.opportunities?.length ?? 0;
  const tags = [];
  if (risks) tags.push({ label: 'Риск', color: 'red', reason: 'В анализе отмечены риски для компании.' });
  if (opportunities) tags.push({ label: 'Возможность', color: 'green', reason: 'В анализе отмечены возможности для компании.' });
  if (!tags.length) tags.push({ label: 'Без оценки', color: 'grey', reason: 'В анализе не выделены риски или возможности.' });
  if ((insight.impact_level ?? insight.impact?.level) === 'high')
    tags.push({ label: 'Высокое влияние', color: 'teal', reason: 'В существующем анализе указан высокий уровень влияния на компанию.' });
  return tags;
}
