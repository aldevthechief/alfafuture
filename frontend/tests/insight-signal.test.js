import test from 'node:test';
import assert from 'node:assert/strict';
import { insightSignal } from '../src/insightSignal.js';
import { toFeed } from '../src/api.js';

test('risk, opportunity and mixed outcomes have distinct labels without inferring from headlines', () => {
  assert.equal(insightSignal({ title: 'Рынок растёт', risks: [{}] }).emoji, '🐻');
  assert.equal(insightSignal({ opportunities: [{}] }).tone, 'positive');
  const mixed = insightSignal({ risks: [{}], opportunities: [{}, {}, {}] });
  assert.equal(mixed.emoji, '⚠️');
  assert.equal(mixed.tone, 'negative'); // Counting more opportunities cannot cancel a risk.
  assert.equal(insightSignal({}).tone, 'neutral');
});

test('preview counts including explicit zeros agree with full details', () => {
  for (const [risks, opportunities] of [[0, 0], [1, 0], [0, 2], [2, 3]]) {
    const detail = { risks: Array(risks).fill({}), opportunities: Array(opportunities).fill({}), impact: { level: 'high' } };
    assert.deepEqual(insightSignal({ counts: { risks, opportunities }, impact_level: 'high' }), insightSignal(detail));
  }
  assert.equal(insightSignal({ counts: { risks: 0, opportunities: 1 }, risks: [{}] }).tone, 'positive');
});

test('high influence survives API mapping and remains independent of confidence and direction', () => {
  const [card] = toFeed({ result: { results: [{ status: 'ready', evidence: {}, insight: {
    risks: [{ text: 'Риск' }], confidence: 'low', impact: { level: 'high' },
  } }] } }).insights;
  assert.equal(insightSignal(card).high, true);
  assert.equal(insightSignal(card).tone, 'negative');
  assert.equal(insightSignal({ confidence: 'high', impact_level: 'mid' }).high, false);
  assert.equal(insightSignal({ confidence: 'high' }).high, false);
});
