import React from 'react';
import { insightSignal } from './insightSignal.js';

export function SignalBadge({ insight }) {
  const signal = insightSignal(insight);
  return <>
    <span className={`insight-signal signal-${signal.tone}`} title={signal.reason}>
      <span aria-hidden="true">{signal.emoji}</span> {signal.label}
    </span>
    {signal.high && <span className="insight-signal signal-important" title="В существующем анализе указан высокий уровень влияния на компанию.">
      <span aria-hidden="true">🔥</span> Высокое влияние
    </span>}
  </>;
}
