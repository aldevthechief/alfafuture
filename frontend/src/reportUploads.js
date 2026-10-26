export const reportFinished = job => ['succeeded','partial','failed','interrupted'].includes(job.status);

export function reportStatus(job) {
  if (job.status==='succeeded') return job.result?.published===false
    ? {label:'Есть более новая версия',tone:'muted'} : {label:'Обработка завершена',tone:'success'};
  if (job.status==='failed') return /Local OCR needs rus\/eng language data/.test(job.error || '')
    ? {label:'На сервере не настроено распознавание',tone:'error',hint:'PDF сохранён на сервере. Для обработки нужно восстановить сервис распознавания.'}
    : {label:'Не удалось обработать PDF',tone:'error',hint:'PDF сохранён на сервере, но обработка не завершилась. Попробуйте позже.'};
  if (job.status==='interrupted') return {label:'Обработка прервана',tone:'error',hint:'Сервер был перезапущен. Отправьте PDF ещё раз.'};
  if (job.status==='partial') return {label:'Подготовлен не полностью',tone:'error',hint:'Отправьте PDF ещё раз, чтобы завершить подготовку.'};
  if (job.status==='queued') return {label:'В очереди на обработку',tone:'pending'};
  return {label:'Распознаём и подготавливаем отчёт',tone:'pending'};
}

export function mergeReportJobs(previous,incoming) {
  const known=new Map(previous.map(job=>[job.id,job]));
  for (const job of incoming) {
    const old=known.get(job.id);
    known.set(job.id,{...old,...job,...(!job.result && old?.result ? {result:old.result} : {})});
  }
  return [...known.values()].sort((a,b)=>Number(reportFinished(a))-Number(reportFinished(b)) || (b.created_at || '').localeCompare(a.created_at || ''));
}

// Per-file acknowledgement keeps accepted PDFs out of retries when another file fails or the queue fills up.
export async function submitReports(files,{upload,onStart,onAccepted,onError}) {
  for (const file of files) {
    onStart(file);
    try { onAccepted(file,await upload(file)); }
    catch (error) { onError(file,error); }
  }
}
