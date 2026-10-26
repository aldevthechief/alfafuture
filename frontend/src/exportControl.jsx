import React, { useEffect, useId, useRef, useState } from 'react';
import { API_ENABLED, downloadExport } from './api.js';
import './exportControl.css';

function ExportForm({kind, params, description, inline}) {
  const id = useId();
  const controller = useRef(null);
  const [format,setFormat] = useState('xlsx');
  const [state,setState] = useState({busy:false,error:'',done:false});
  useEffect(() => () => controller.current?.abort(), []);
  async function download() {
    if (controller.current || !params.company_id) return;
    const request = new AbortController();
    controller.current = request;
    setState({busy:true,error:'',done:false});
    try {
      await downloadExport(kind, {...params,format}, request.signal);
      if (!request.signal.aborted) setState({busy:false,error:'',done:true});
    } catch (error) {
      if (!request.signal.aborted) setState({busy:false,error:error.message || 'Не удалось скачать файл.',done:false});
    } finally { if (controller.current === request) controller.current = null; }
  }
  const panel = <div className={`export-panel${inline ? ' export-panel-inline' : ''}`}>
    <p id={`${id}-scope`}>{description}</p>
    <div className="export-actions"><label htmlFor={`${id}-format`}>Формат</label>
      <select id={`${id}-format`} value={format} disabled={state.busy} onChange={event=>setFormat(event.target.value)}>
        <option value="xlsx">Excel · XLSX</option>
        <option value="csv">{kind==='payments' ? 'CSV · UTF-8' : 'CSV · архив ZIP'}</option>
        <option value="json">JSON</option>
      </select>
      <button type="button" className="btn btn-red" disabled={state.busy || !API_ENABLED || !params.company_id}
        aria-describedby={`${id}-scope`} onClick={download}>{state.busy ? 'Готовим файл…' : state.error ? 'Повторить скачивание' : 'Скачать'}</button>
    </div>
    {state.busy && <p className="export-status" role="status">Подготавливаем выгрузку…</p>}
    {state.error && <p className="export-error" role="alert">{state.error}</p>}
    {state.done && <p className="export-status" role="status">Скачивание началось.</p>}
  </div>;
  return inline ? panel : <details className="export-control"><summary>Экспорт</summary>{panel}</details>;
}

export function ExportControl(props) {
  // Changing company or role aborts an old download and clears its status.
  return <ExportForm key={`${props.kind}:${JSON.stringify(props.params)}`} {...props} />;
}
