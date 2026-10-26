import test from 'node:test';
import assert from 'node:assert/strict';
import { SEARCH_SECTIONS, findSearchItems, companySearchItems, paymentSearchItems, counterpartyPaymentItems } from '../src/siteSearch.js';

test('global navigation covers cabinet sections and synonyms',()=>{
  for (const [query,view] of [['PDF','documents'],['платежная аналитика','payments'],['численность','company'],['поставщики','counterparties'],['помощник','chat'],['календарь','calendar']])
    assert.ok(findSearchItems(SEARCH_SECTIONS,query).some(item=>item.view===view));
});

test('company search preserves facts and targets their actual sections without inventing trade status',()=>{
  const items=companySearchItems({profile:{name:'Лизинг №1',industry:'Лизинг',bank_products:['Гарантии'],overview:{trade:{countries:['Китай']},facts:[
    {category:'profile',text:'Численность 100–500'}, {category:'group_geo',text:'Группа работает в Омске'},
  ]}}});
  assert.equal(findSearchItems(items,'Китай')[0].target,'trade');
  assert.match(findSearchItems(items,'Китай')[0].text,/Статус ФТС неизвестен/);
  assert.equal(findSearchItems(items,'100–500')[0].target,'team');
  assert.equal(findSearchItems(items,'Омске')[0].target,'group');
  assert.equal(findSearchItems(items,'Гарантии')[0].target,'products');
  assert.deepEqual(companySearchItems(null),[]);
});

test('payment search keeps incoming/outgoing navigation separate and carries counterparty IDs',()=>{
  const items=paymentSearchItems({summary:{in:{amount:100,count:2,average:50},out:{amount:20,count:1,average:20}},flows:{in:{regions:[{label:'Москва',amount:100,share:1}]}}});
  assert.equal(findSearchItems(items,'Москва')[0].direction,'in');
  assert.equal(findSearchItems(items,'списания')[0].direction,'out');
  const links=counterpartyPaymentItems([{id:'U_123',name:'Компания №123',in:{amount:100,count:2},out:{amount:0,count:0}}]);
  assert.equal(links.length,1);
  assert.equal(links[0].query,'123');
  assert.equal(links[0].direction,'in');
  assert.deepEqual(paymentSearchItems(null),[]);
});
