import React from 'react';
import { Status } from '@alfalab/core-components/status';
import { signalTags } from './insightSignal.js';

// Same component and size as every other tag on the card (channel, confidence, bank position).
export function SignalBadge({ insight }) {
  return <>{signalTags(insight).map(t => <span key={t.label} title={t.reason}><Status color={t.color} size={20}>{t.label}</Status></span>)}</>;
}
