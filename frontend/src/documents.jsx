import React, { useEffect, useRef, useState } from 'react';
import { pdfKey, selectPdfs } from './pdfSelection.js';
import { REPORT_UPLOAD_ENABLED, reportJob, reportJobs, uploadReport } from './api.js';
import { mergeReportJobs, reportFinished, reportStatus, submitReports } from './reportUploads.js';
import './documents.css';

const sizeLabel = bytes => bytes<1024 ? `${bytes} Б` : `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits:1 }).format(bytes / (bytes<1024*1024 ? 1024 : 1024*1024))} ${bytes<1024*1024 ? 'КБ' : 'МБ'}`;

export default function DocumentsPage({ name }) {
  const input = useRef(null);
  const dragDepth = useRef(0);
  const [selection, setSelection] = useState({ files:[], errors:[] });
  const [dragging, setDragging] = useState(false);
  const [uploading,setUploading]=useState(false);
  const [fileStates,setFileStates]=useState({});
  const [jobs,setJobs]=useState([]);
  const [statusError,setStatusError]=useState('');
  const [refresh,setRefresh]=useState(0);
  const mounted=useRef(true);
  const sending=useRef(false);
  const { files, errors } = selection;
  const add = incoming => { if (REPORT_UPLOAD_ENABLED && !sending.current) setSelection(previous => selectPdfs(previous.files, incoming)); };
  useEffect(()=>{mounted.current=true;return ()=>{mounted.current=false;};},[]);
  useEffect(()=>{
    if (!REPORT_UPLOAD_ENABLED) return;
    const controller=new AbortController();
    const completed=new Map();
    let timer;
    async function update() {
      try {
        const reports=await reportJobs(controller.signal);
        const latest=await Promise.all(reports.map(async job=>{
          if (!reportFinished(job)) return job;
          if (!completed.has(job.id)) completed.set(job.id,await reportJob(job.id,controller.signal));
          return completed.get(job.id);
        }));
        if (!controller.signal.aborted) {setJobs(previous=>mergeReportJobs(previous,latest));setStatusError('');}
      } catch (error) {
        if (!controller.signal.aborted) setStatusError('Не удалось обновить статусы. Уже принятые файлы продолжают обрабатываться; повторим проверку автоматически.');
      } finally {if (!controller.signal.aborted) timer=setTimeout(update,3000);}
    }
    update();
    return ()=>{controller.abort();clearTimeout(timer);};
  },[refresh]);
  async function send() {
    if (sending.current || !files.length || !REPORT_UPLOAD_ENABLED) return;
    sending.current=true;setUploading(true);
    const change=(file,value)=>{if (mounted.current) setFileStates(previous=>({...previous,[pdfKey(file)]:value}));};
    await submitReports([...files],{upload:uploadReport,
      onStart:file=>change(file,{sending:true}),
      onAccepted:(file,job)=>{if (mounted.current) {
        setSelection(previous=>({...previous,files:previous.files.filter(item=>pdfKey(item)!==pdfKey(file))}));
        setJobs(previous=>mergeReportJobs(previous,[job]));
      }},
      onError:(file,error)=>change(file,{error:error instanceof TypeError ? 'Не удалось связаться с сервером. Проверьте статусы перед повторной отправкой.' : error.message}),
    });
    sending.current=false;
    if (mounted.current) {setUploading(false);setRefresh(value=>value+1);}
  }
  return <div className="documents-page">
    <h1 className="h-sys">Документооборот</h1>
    <p className="h-sub">{name} · аналитические материалы</p>
    <section className="card documents-import" aria-labelledby="documents-title">
      <header><span className="documents-kicker">ОТЧЁТЫ И ОБЗОРЫ</span><h2 id="documents-title">Аналитика в PDF</h2><p>Добавьте отраслевые обзоры, исследования и аналитические отчёты. После подготовки они станут доступны агенту при новом анализе.</p></header>
      <div className={`documents-drop${dragging ? ' documents-drop-active' : ''}`}
        onDragEnter={event=>{event.preventDefault();dragDepth.current++;setDragging(true);}}
        onDragOver={event=>{event.preventDefault();event.dataTransfer.dropEffect='copy';}}
        onDragLeave={event=>{event.preventDefault();if (--dragDepth.current<=0) {dragDepth.current=0;setDragging(false);}}}
        onDrop={event=>{event.preventDefault();dragDepth.current=0;setDragging(false);add(Array.from(event.dataTransfer.files));}}>
        <span className="documents-pdf-icon" aria-hidden="true">PDF</span>
        <strong>Перетащите PDF сюда</strong><p>Или выберите файлы на компьютере</p>
        <button type="button" className="btn btn-gray" disabled={!REPORT_UPLOAD_ENABLED || uploading} onClick={()=>input.current?.click()}>Выбрать PDF</button>
        <input ref={input} disabled={!REPORT_UPLOAD_ENABLED || uploading} className="documents-input" type="file" accept="application/pdf,.pdf" multiple aria-label="Выберите PDF с аналитикой"
          onChange={event=>{add(Array.from(event.target.files));event.target.value='';}} />
        <small>До 32 файлов · до 50 МБ каждый</small>
      </div>
      {!!errors.length && <div className="documents-errors" role="alert"><strong>Некоторые файлы не добавлены</strong><ul>{errors.map((error,index)=><li key={index}>{error}</li>)}</ul></div>}
      {!!files.length && <section className="documents-selection" aria-label="Выбранные файлы">
        <div className="documents-list-heading"><h3>Выбрано · {files.length}</h3><button type="button" disabled={uploading} onClick={()=>{setSelection({files:[],errors:[]});setFileStates({});}}>Очистить список</button></div>
        <ul>{files.map(file=><li key={pdfKey(file)}><span className="documents-file-badge" aria-hidden="true">PDF</span><div><strong title={file.name}>{file.name}</strong><small>{sizeLabel(file.size)} · {fileStates[pdfKey(file)]?.sending ? 'Отправляем на сервер…' : 'ожидает отправки'}</small>
          {fileStates[pdfKey(file)]?.error && <p className="documents-file-error" role="alert">{fileStates[pdfKey(file)].error}</p>}</div>
          <button type="button" disabled={uploading} className="documents-remove" aria-label={`Удалить ${file.name}`} onClick={()=>setSelection(previous=>({...previous,files:previous.files.filter(item=>pdfKey(item)!==pdfKey(file))}))}>×</button></li>)}</ul>
      </section>}
      <footer className="documents-footer"><p id="documents-import-status" role="status">{REPORT_UPLOAD_ENABLED
        ? uploading ? 'Отправляем выбранные файлы. Подготовка отчётов продолжится в фоне.' : 'Отчёты добавляются в общую библиотеку аналитики для всех кабинетов. Готовые инсайты автоматически не пересчитываются.'
        : 'Загрузка PDF доступна в полной версии приложения. В демо можно посмотреть готовые инсайты.'}</p>
        <button type="button" className="btn btn-red" disabled={!REPORT_UPLOAD_ENABLED || uploading || !files.length} aria-describedby="documents-import-status" onClick={send}>{uploading ? 'Отправляем PDF…' : 'Импортировать PDF'}</button></footer>
    </section>
    {REPORT_UPLOAD_ENABLED && <section className="card documents-jobs" aria-labelledby="documents-jobs-title">
      <div className="documents-list-heading"><h2 id="documents-jobs-title">Обработка отчётов</h2><button type="button" onClick={()=>setRefresh(value=>value+1)}>Обновить статусы</button></div>
      {statusError && <p role="alert" className="documents-file-error">{statusError}</p>}
      {!jobs.length && !statusError && <p className="documents-empty">Здесь появятся принятые сервером PDF и результаты их обработки.</p>}
      <ul aria-label="Обработка PDF" aria-live="polite">{jobs.map(job=>{const status=reportStatus(job);return <li key={job.id}>
        <span className={`documents-job-dot documents-job-${status.tone}`} aria-hidden="true"/>
        <div><strong>{job.request?.file || 'Отчёт PDF'}</strong><p className={`documents-job-status documents-job-${status.tone}`}>{status.label}</p>
          {status.hint && <small>{status.hint}</small>}</div>
        <time dateTime={job.created_at}>{job.created_at && new Date(job.created_at).toLocaleString('ru-RU',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})}</time>
      </li>;})}</ul>
    </section>}
    <aside className="documents-help"><span aria-hidden="true">↗</span><p>Готовые выводы и ссылки на материалы доступны в разделе <button type="button" onClick={()=>window.showView('insights')}>«Инсайты»</button>.</p></aside>
  </div>;
}
