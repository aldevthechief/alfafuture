import test from 'node:test';
import assert from 'node:assert/strict';
import { referenceParts, sourceLabel, sourceUrl } from '../src/sourceReferences.js';
import { answerSources, askCompanyChat, companyChatHistory, openCompanyChat } from '../src/feedback.js';

test('inline citations keep punctuation and resolve exact IDs including report titles and number prefixes', () => {
  const sources = { 'news:8472':{title:'Статья'}, 'fact:50':{kind:'fact',text:'68%'}, 'fact:52':{kind:'fact',text:'5%'},
    'bank:Отчёт (июль).pdf:p12.3':{kind:'bank',title:'Отчёт',page:12} };
  const text = '1) рост (news:8472); 2) платежи (fact:50, fact:52). Позиция: bank:Отчёт (июль).pdf:p12.3.';
  const parts = referenceParts(text,sources);
  assert.deepEqual(parts.filter(p=>p.id).map(p=>p.id),Object.keys(sources));
  assert.equal(parts.map(p=>p.text ?? p.id).join(''),text);
  assert.equal(referenceParts('fact:500',sources)[0].source,undefined);
  assert.equal(sourceLabel(sources['fact:50']),'Данные компании');
  assert.equal(sourceLabel(sources['bank:Отчёт (июль).pdf:p12.3']),'Отчёт · с. 12');
  assert.equal(sourceUrl('javascript:alert(1)'),null);
  assert.equal(sourceUrl('https://example.com/article'),'https://example.com/article');
});

test('an answer uses its own frozen facts and lookup URLs instead of a newer card snapshot', () => {
  const saved = {evidence:{facts:[{id:'fact:50',text:'Сохранённые 68%'}]}};
  const sources = answerSources(saved,{analysis:{retrieved_sources:[{id:'lookup:news:9',origin:'news_pool',title:'Найденная статья',
    url:'https://example.com/news/9',text:'Цитата',date:'2026-10-25'}]}}, {'fact:50':{text:'Позднее 10%'},'fact:52':{text:'Чужой факт'}});
  assert.equal(sources['fact:50'].text,'Сохранённые 68%');
  assert.equal(sources['fact:52'],undefined);
  assert.equal(sources['lookup:news:9'].url,'https://example.com/news/9');
  assert.equal(sources['lookup:news:9'].title,'Найденная статья');
  assert.equal(answerSources(saved,{evidence:{facts:[{id:'fact:50',text:'Отдельный снимок ответа'}]}})['fact:50'].text,'Отдельный снимок ответа');
});

test('company chat opens and streams through its own endpoints with no insight registration', async t => {
  const paths=[];
  t.mock.method(globalThis,'fetch',async (url,init)=>{
    paths.push(url);
    if(url.endsWith('/v1/company-chats')) {
      assert.deepEqual(JSON.parse(init.body),{company_id:'U_1',role:'cfo'});
      return {ok:true,json:async()=>({id:'conversation'})};
    }
    if(url.includes('/messages')) return {ok:true,json:async()=>({items:[],has_more:false,before:null})};
    assert.deepEqual(JSON.parse(init.body),{question:'Вопрос',request_id:'request',selected_text:'Цитата ответа'});
    const data='event: text\ndata: {"text":"Ответ"}\n\nevent: complete\ndata: {"id":"answer","analysis":{"response":"Ответ"}}\n\n';
    return {ok:true,body:new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode(data));controller.close();}})};
  });
  const chat=await openCompanyChat('U_1','cfo');
  await companyChatHistory(chat.id);
  const chunks=[];
  const result=await askCompanyChat(chat.id,'Вопрос','request',text=>chunks.push(text),'Цитата ответа');
  assert.equal(result.analysis.response,'Ответ');
  assert.deepEqual(chunks,['Ответ']);
  assert.ok(paths.every(path=>!path.includes('/insights')));
  assert.ok(paths.at(-1).endsWith('/v1/company-chats/conversation/ask/stream'));
});
