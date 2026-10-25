import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const require = createRequire(import.meta.url);
async function variant(mode) {
  const result = await build({
    stdin: { contents:`export { default as Entry } from './src/accountChooser.jsx';
      export * from './src/cabinetMode.js'; export { startAnalysis } from './src/api.js';`,
      resolveDir:fileURLToPath(new URL('..', import.meta.url)) },
    bundle:true, write:false, format:'esm', platform:'node', loader:{'.css':'empty'},
    define:{'import.meta.env':JSON.stringify({VITE_CABINET_MODE:mode})},
    plugins:[{name:'shared-react', setup(api) {
      api.onResolve({filter:/^react$/}, () => ({path:pathToFileURL(require.resolve('react')).href, external:true}));
    }}],
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const session = {kind:'cached', company_id:'U_14366', role:'cfo', job_id:'demo-U_14366-cfo', company:{name:'Север Финанс'}};

test('server build renders only prepared entry and rejects inference before any HTTP request', async t => {
  const server = await variant('server');
  const html = renderToStaticMarkup(React.createElement(server.Entry, {onEnter:() => {}}));
  assert.ok(html.includes('Выберите кабинет'));
  assert.ok(!html.includes('Другая компания?'));
  assert.ok(server.validSession(session));
  assert.equal(server.validSession({...session, kind:'live'}), false);
  const fetch = t.mock.fn();
  t.mock.method(globalThis, 'fetch', fetch);
  await assert.rejects(server.startAnalysis('U_14366', 'cfo'), /только готовые кабинеты/);
  assert.equal(fetch.mock.calls.length, 0);
});

test('full build offers custom entry and posts both agent branches for the selected role', async t => {
  const full = await variant('full');
  const html = renderToStaticMarkup(React.createElement(full.Entry, {onEnter:() => {}}));
  assert.ok(html.includes('Другая компания?'));
  assert.ok(full.validSession({...session, kind:'live'}));
  assert.equal(full.validSession({...session, company:null}), false);
  assert.notEqual(full.SESSION_KEY, (await variant('server')).SESSION_KEY);
  let payload;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.ok(url.endsWith('/v1/analyses'));
    payload = JSON.parse(init.body);
    return {ok:true, json:async () => ({id:'new-job', status:'queued'})};
  });
  const job = await full.startAnalysis('U_777', 'marketing', {engine:'judge', source:'both',
    analytics_cache_only:false, analytics_refresh:true});
  assert.equal(job.id, 'new-job');
  assert.equal(payload.company_id, 'U_777');
  assert.equal(payload.role, 'marketing');
  assert.equal(payload.source, 'both');
  assert.equal(payload.analytics_refresh, true);
  assert.equal(payload.analytics_cache_only, false);
});
