import React, { memo } from 'react';
import { Status } from '@alfalab/core-components/status';
import { Typography } from '@alfalab/core-components/typography';
import { CONFIDENCE, day } from './insightView.js';
import { insightSignal } from './insightSignal.js';
import { SignalBadge } from './insightSignal.jsx';

export const CompactCard = memo(function CompactCard({ insight: i, compact, onOpen }) {
  // A collapsed card only needs a heading and counters, not copies of every claim/source.
  compact = compact || i.compact;
  const v = compact || i;
  const [color, conf, confTitle] = CONFIDENCE[i.confidence] || CONFIDENCE.low;
  const count = [['Риски', i.counts?.risks ?? v.risks.length], ['Возможности', i.counts?.opportunities ?? v.opportunities.length],
    ['Действия', i.counts?.actions ?? v.actions.length]].filter(([, n]) => n);
  return <button className={`card ic ic2 insight-tone-${insightSignal(i).tone}`} onClick={() => onOpen(i)}>
    <div className="ic-tags">
      <SignalBadge insight={i} />
      <Status color="blue" size={20}>{i.channel}</Status>
      <span title={confTitle}><Status color={color} size={20}>{conf}</Status></span>
      {(i.counts?.bank_view ?? i.bank_view.length) > 0 && <Status color="purple" size={20}>Позиция Банка</Status>}
      <span className="ic-date">{day(i.published)}</span>
    </div>
    <Typography.Title font="system" tag="h3" view="small" weight="bold">{v.title}</Typography.Title>
    <Typography.Text view="primary-medium" color="secondary" tag="p">{compact ? compact.lead : i.lead ?? i.why_it_matters[0]?.text}</Typography.Text>
    {count.length > 0 && <div className="ic-count">{count.map(([t, n]) => <span key={t}>{t} · {n}</span>)}</div>}
  </button>;
});
