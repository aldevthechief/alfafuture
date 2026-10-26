import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { bankReports } from '../src/bankSources.js';
import { signalTags } from '../src/insightSignal.js';

const require = createRequire(import.meta.url);
const result = await build({
  stdin:{ contents:`export { ConditionFit } from './src/conditionFit.jsx';
    export { SignalBadge } from './src/insightSignal.jsx';`, resolveDir:fileURLToPath(new URL('..', import.meta.url)) },
  bundle:true, write:false, format:'cjs', platform:'node', external:['react', 'react-dom'], loader:{'.css':'empty'},
  define:{'import.meta.env':'{}'},
});
const module = { exports:{} };
new Function('require', 'module', 'exports', result.outputFiles[0].text)(require, module, module.exports);
const { ConditionFit, SignalBadge } = module.exports;
const render = state => renderToStaticMarkup(React.createElement(ConditionFit, { state, onPick:() => {} }));

test('pending and failed saves have honest status, disabled pending controls and an explicit retry', () => {
  const pending = render({ busy:true, pending:'yes' });
  assert.ok(pending.includes('Сохраняем ответ'));
  assert.ok(pending.includes('disabled=""'));
  assert.ok(!pending.includes('Ответ сохранён'));
  const failed = render({ error:'Сервер недоступен', pending:'yes' });
  assert.ok(failed.includes('role="alert"'));
  assert.ok(failed.includes('Повторить'));
  assert.ok(!failed.includes('Ответ сохранён'));
  const confirmed = render({ choice:'no' });
  assert.ok(confirmed.includes('Ответ сохранён · Не про нас'));
  assert.ok(confirmed.includes('aria-pressed="true"'));
});

test('risk and opportunity remain separate tags alongside high impact, with no emoji in the rendered badges', () => {
  const insight = { counts:{ risks:1, opportunities:2 }, impact_level:'high' };
  assert.deepEqual(signalTags(insight).map(t => t.label), ['Риск', 'Возможность', 'Высокое влияние']);
  const html = renderToStaticMarkup(React.createElement(SignalBadge, { insight }));
  assert.ok(html.includes('Риск') && html.includes('Возможность') && html.includes('Высокое влияние'));
  assert.ok(!/🐻|📈|🔥|⚠️/.test(html));
});

test('bank sources group cited pages numerically while keeping different report editions separate', () => {
  const sources = {
    a:{ kind:'bank', title:'Прогноз', date:'2026-10-25', page:10 },
    b:{ kind:'bank', title:'Прогноз', date:'2026-10-25', page:2 },
    c:{ kind:'bank', title:'Прогноз', date:'2026-10-25', page:'2' },
    d:{ kind:'bank', title:'Прогноз', date:'2026-09-25', page:3 },
    e:{ kind:'news', title:'Новость' },
  };
  const reports = bankReports(Object.keys(sources), sources);
  assert.equal(reports.length, 2);
  assert.deepEqual(reports[0].pages, ['2', '10']);
  assert.notEqual(reports[0].key, reports[1].key);
  assert.equal(reports[1].date, '2026-09-25');
});
