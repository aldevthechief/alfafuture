import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { toFeed } from '../src/api.js';
import { confidenceDescription } from '../src/insightView.js';

const require=createRequire(import.meta.url);
const bundle=await build({stdin:{contents:"export { InsightAssessment } from './src/insightAssessment.jsx';",resolveDir:fileURLToPath(new URL('..',import.meta.url))},
  bundle:true,write:false,format:'cjs',platform:'node',external:['react','react-dom'],loader:{'.css':'empty'}});
const module={exports:{}};
new Function('require','module','exports',bundle.outputFiles[0].text)(require,module,module.exports);
const {InsightAssessment}=module.exports;

test('API mapping carries confidence reasons, source credibility and product reason with the frozen result',()=>{
  const row={status:'ready',insight:{id:'card',title:'Ставка',confidence:'high',bank_product:'Обсудить фиксацию ставки'},
    business_value:{confidence_basis:['Цитаты сверены'],bank_product:{why:'В профиле есть кредит'}},
    credibility:{label:'Публикации в 2 изданиях',n_outlets:2}};
  const card=toFeed({id:'job',request:{role:'cfo'},result:{results:[row]}}).insights[0];
  assert.equal(card.pipeline_result,row);
  assert.equal(card.confidence,'high');assert.equal(card.bank_product_reason,'В профиле есть кредит');
  assert.match(confidenceDescription(card),/Цитаты сверены/);
  const html=renderToStaticMarkup(React.createElement(InsightAssessment,{insight:card}));
  assert.match(html,/Почему такая оценка уверенности/);
  assert.match(html,/Цитаты сверены/);assert.match(html,/Публикации в 2 изданиях/);
});

test('legacy cards remain readable without metadata; repeated source labels are shown once and text is escaped',()=>{
  assert.equal(renderToStaticMarkup(React.createElement(InsightAssessment,{insight:{}})),'');
  const card={confidence:'medium',confidence_basis:['Источник <script>'],credibility:{label:'Источник <script>'}};
  const html=renderToStaticMarkup(React.createElement(InsightAssessment,{insight:card}));
  assert.equal((html.match(/Источник &lt;script&gt;/g)||[]).length,1);
  assert.equal(html.includes('<script>'),false);
});
