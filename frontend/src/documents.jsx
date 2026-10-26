import React, { useRef, useState } from 'react';
import { pdfKey, selectPdfs } from './pdfSelection.js';
import './documents.css';

const sizeLabel = bytes => `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits:1 }).format(bytes / 1024 / 1024)} МБ`;

export default function DocumentsPage({ name }) {
  const input = useRef(null);
  const dragDepth = useRef(0);
  const [selection, setSelection] = useState({ files:[], errors:[] });
  const [dragging, setDragging] = useState(false);
  const { files, errors } = selection;
  const add = incoming => setSelection(previous => selectPdfs(previous.files, incoming));
  return <div className="documents-page">
    <h1 className="h-sys">Документооборот</h1>
    <p className="h-sub">{name} · аналитические материалы</p>
    <section className="card documents-import" aria-labelledby="documents-title">
      <header><span className="documents-kicker">ОТЧЁТЫ И ОБЗОРЫ</span><h2 id="documents-title">Аналитика в PDF</h2><p>Подготовьте материалы, которые помогут дополнить контекст компании: отраслевые обзоры, исследования и аналитические отчёты.</p></header>
      <div className={`documents-drop${dragging ? ' documents-drop-active' : ''}`}
        onDragEnter={event=>{event.preventDefault();dragDepth.current++;setDragging(true);}}
        onDragOver={event=>{event.preventDefault();event.dataTransfer.dropEffect='copy';}}
        onDragLeave={event=>{event.preventDefault();if (--dragDepth.current<=0) {dragDepth.current=0;setDragging(false);}}}
        onDrop={event=>{event.preventDefault();dragDepth.current=0;setDragging(false);add(Array.from(event.dataTransfer.files));}}>
        <span className="documents-pdf-icon" aria-hidden="true">PDF</span>
        <strong>Перетащите PDF сюда</strong><p>Или выберите файлы на компьютере</p>
        <button type="button" className="btn btn-gray" onClick={()=>input.current?.click()}>Выбрать PDF</button>
        <input ref={input} className="documents-input" type="file" accept="application/pdf,.pdf" multiple aria-label="Выберите PDF с аналитикой"
          onChange={event=>{add(Array.from(event.target.files));event.target.value='';}} />
        <small>До 32 файлов · до 50 МБ каждый</small>
      </div>
      {!!errors.length && <div className="documents-errors" role="alert"><strong>Некоторые файлы не добавлены</strong><ul>{errors.map((error,index)=><li key={index}>{error}</li>)}</ul></div>}
      {!!files.length && <section className="documents-selection" aria-label="Выбранные файлы">
        <div className="documents-list-heading"><h3>Выбрано · {files.length}</h3><button type="button" onClick={()=>setSelection({files:[],errors:[]})}>Очистить список</button></div>
        <ul>{files.map(file=><li key={pdfKey(file)}><span className="documents-file-badge" aria-hidden="true">PDF</span><div><strong title={file.name}>{file.name}</strong><small>{sizeLabel(file.size)} · выбран на устройстве</small></div>
          <button type="button" className="documents-remove" aria-label={`Удалить ${file.name}`} onClick={()=>setSelection(previous=>({...previous,files:previous.files.filter(item=>pdfKey(item)!==pdfKey(file))}))}>×</button></li>)}</ul>
      </section>}
      <footer className="documents-footer"><p id="documents-import-status">Импорт пока недоступен. Выбранные файлы остаются на вашем устройстве и не отправляются на обработку.</p><button type="button" className="btn btn-red" disabled aria-describedby="documents-import-status">Импортировать PDF</button></footer>
    </section>
    <aside className="documents-help"><span aria-hidden="true">↗</span><p>Готовые выводы и ссылки на материалы доступны в разделе <button type="button" onClick={()=>window.showView('insights')}>«Инсайты»</button>.</p></aside>
  </div>;
}
