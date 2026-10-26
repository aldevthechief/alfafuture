import test from 'node:test';
import assert from 'node:assert/strict';
import { findInsights, insightIdentity, notificationScope, readInsightViews, mergeInsightViews, toggleInsightTag } from '../src/cabinetToolsState.js';

test('local search uses original and compact previews, ignores case and ё, and requires all query words',()=>{
  const cards=[{id:'1',title:'Нефть и топливо',channel:'Инфляция',compact:{title:'Рост цен',lead:'Платёж за топливо вырастет'}},
    {id:'2',title:'Рост ставки',lead:'Изменение цены финансирования'}, {id:'3',title:'Курс рубля'}];
  assert.deepEqual(findInsights(cards,'ПЛАТЕЖ топливо').map(i=>i.id),['1']);
  assert.deepEqual(findInsights(cards,'Инфляция рост').map(i=>i.id),['1']);
  assert.deepEqual(findInsights(cards,'Рост').map(i=>i.id),['1','2']);
  assert.deepEqual(findInsights(cards,'финансирования').map(i=>i.id),['2']);
  assert.deepEqual(findInsights(cards,' '),cards);
  assert.deepEqual(findInsights(cards,'топливо ставка'),[]);
  assert.deepEqual(findInsights([{id:'empty'}],'нефть'),[]);
});

test('viewed state is scoped to user, company and role and distinguishes jobs with the same card id',()=>{
  const one={job_id:'a',id:'1'}, two={job_id:'b',id:'1'};
  assert.notEqual(insightIdentity(one),insightIdentity(two));
  const key=notificationScope('alice','U_1','cfo');
  for (const args of [['bob','U_1','cfo'],['alice','U_2','cfo'],['alice','U_1','ceo']])
    assert.notEqual(notificationScope(...args),key);
  const previous=[insightIdentity(one)];
  const merged=mergeInsightViews(previous,[one,two]);
  assert.deepEqual(merged,[insightIdentity(one),insightIdentity(two)]);
  assert.deepEqual(previous,[insightIdentity(one)]);
  assert.equal(mergeInsightViews([],Array.from({length:500},(_,id)=>({id}))).length,400);
});

test('text and tag filters intersect and search includes available claim text and risk tags',()=>{
  const cards=[{id:'one',channel:'ставка',counts:{risks:2,opportunities:0},compact:{risks:[{text:'Фондирование станет дороже'}]}},
    {id:'two',channel:'ставка',counts:{risks:0,opportunities:1},lead:'Фондирование дешевеет'},
    {id:'three',channel:'Новости',tags:['Лизинг'],counts:{risks:1},why_it_matters:[{text:'Вырастут расходы'}]}];
  assert.deepEqual(findInsights(cards,'фондирование','risks').map(i=>i.id),['one']);
  assert.deepEqual(findInsights(cards,'#ставка фондирование','opps').map(i=>i.id),['two']);
  assert.deepEqual(findInsights(cards,'расходы','Лизинг').map(i=>i.id),['three']);
  assert.deepEqual(findInsights(cards,'#риски').map(i=>i.id),['one','three']);
  assert.deepEqual(findInsights(cards,'фондирование','Новости'),[]);
});

test('malformed or unavailable browser storage leaves notifications usable',t=>{
  const values=new Map();
  const previous=Object.getOwnPropertyDescriptor(globalThis,'localStorage');
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:key=>{
    if (key==='unavailable') throw new Error('Storage unavailable');
    return values.get(key);
  }}});
  t.after(()=>{if (previous) Object.defineProperty(globalThis,'localStorage',previous); else delete globalThis.localStorage;});
  values.set('valid',JSON.stringify(['one',null,7,'two']));
  values.set('bad','broken JSON');
  values.set('object','{}');
  assert.deepEqual(readInsightViews('valid'),['one','two']);
  assert.deepEqual(readInsightViews('bad'),[]);
  assert.deepEqual(readInsightViews('object'),[]);
  assert.deepEqual(readInsightViews('absent'),[]);
  assert.deepEqual(readInsightViews('unavailable'),[]);
});

test('multiple tags always match any selected tag while words narrow the result',()=>{
  const cards=[{id:'both',channel:'Новости',title:'Лизинг',counts:{risks:1,opportunities:1}},
    {id:'risk',channel:'ставка',title:'Лизинг',counts:{risks:1}},
    {id:'opportunity',channel:'Новости',title:'Курс',counts:{opportunities:1}}];
  const ids=(query,tags)=>findInsights(cards,query,tags).map(item=>item.id);
  assert.deepEqual(ids('',['risks','opps']),['both','risk','opportunity']);
  assert.deepEqual(ids('',['Новости','ставка']),['both','risk','opportunity']);
  assert.deepEqual(ids('Лизинг',['Новости','opps']),['both']);
  assert.deepEqual(ids('Лизинг',['risks','ставка']),['both','risk']);
  assert.deepEqual(ids('',[]),['both','risk','opportunity']);
});

test('selected tags toggle independently and All clears the selection without mutation',()=>{
  const original=['risks'];
  const combined=toggleInsightTag(original,'Новости');
  assert.deepEqual(combined,['risks','Новости']);
  assert.deepEqual(original,['risks']);
  assert.deepEqual(toggleInsightTag(combined,'risks'),['Новости']);
  assert.deepEqual(toggleInsightTag(combined,'all'),[]);
});
