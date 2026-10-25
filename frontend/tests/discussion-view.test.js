import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const require = createRequire(import.meta.url);
const result = await build({
  stdin: { contents: "export { HistoryMessages } from './src/discussion.jsx';", resolveDir: fileURLToPath(new URL('..', import.meta.url)) },
  bundle: true, write: false, format: 'cjs', platform: 'node', external: ['react', 'react-dom'], loader: { '.css': 'empty' },
  define: { 'import.meta.env': '{}' },
});
const module = { exports: {} };
new Function('require', 'module', 'exports', result.outputFiles[0].text)(require, module, module.exports);
const { HistoryMessages } = module.exports;
const render = history => renderToStaticMarkup(React.createElement(HistoryMessages, {
  history, record: {}, insight: { sources: { 'news:1': { title: 'Исходная новость', snippet: 'Исходная цитата' } } }, busy: false, onAsk: () => {},
}));

test('the conversation keeps user questions, full answers and citations while reactions stay out of the transcript', () => {
  const html = render([
    { id: 'rating', kind: 'like', analysis: { response: 'Спасибо за оценку!' } },
    { id: 'question', kind: 'question', comment: 'Как изменится бюджет?', selected_text: 'Условия финансирования', analysis: {
      response: 'Первый абзац.\n\nВторой абзац.', source_ids: ['news:1'], needs_clarification: true, clarification_question: 'Какая доля долга плавающая?',
    } },
  ]);
  assert.ok(!html.includes('Спасибо за оценку!'));
  for (const text of ['Как изменится бюджет?', 'Фрагмент инсайта', 'Условия финансирования', '<p>Первый абзац.</p>', '<p>Второй абзац.</p>', 'Какая доля долга плавающая?', 'Исходная новость', 'Исходная цитата']) {
    assert.ok(html.includes(text), text);
  }
  assert.equal((html.match(/class="chat-turn"/g) || []).length, 1);
});

test('a saved question without an answer stays visible with a retry; factual corrections are also retained', () => {
  const html = render([
    { id: 'pending', kind: 'question', comment: 'Сохранённый вопрос', request_id: 'same-on-retry', analysis: null },
    { id: 'correction', kind: 'incorrect', comment: 'Уточнение пользователя', selected_text:'Цитата из ответа', selected_source:'answer', analysis: { response: 'Проверим исходные данные.' } },
  ]);
  for (const text of ['Сохранённый вопрос', 'Повторить запрос ответа', 'Уточнение пользователя', 'Проверим исходные данные.', 'Фрагмент ответа агента', 'Цитата из ответа']) assert.ok(html.includes(text));
  assert.equal((html.match(/class="chat-turn"/g) || []).length, 2);
});

test('internal source IDs in an agent answer become accessible source buttons', () => {
  const html=render([{id:'q',kind:'question',comment:'Почему?',evidence:{facts:[{id:'fact:50',text:'68% платежей'}],
    news:[{id:8472,title:'Название статьи',url:'https://example.com/article',text:'Новость'}]},
    analysis:{response:'Рост (news:8472); платежи (fact:50). Неизвестный факт fact:999.',source_ids:['news:8472','fact:50']}}]);
  assert.ok(html.includes('aria-label="Источник: Название статьи"'));
  assert.ok(html.includes('aria-label="Источник: Данные компании"'));
  assert.ok(html.includes('Источник недоступен'));
  assert.ok(!html.includes('news:8472'));
  assert.ok(!html.includes('fact:50'));
});
