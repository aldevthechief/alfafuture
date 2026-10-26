export const MAX_PDFS = 32;
export const MAX_PDF_BYTES = 50 * 1024 * 1024;

export const pdfKey = file => JSON.stringify([file.name, file.size, file.lastModified]);

// Mirrors upload limits without reading files or starting server-side processing.
export function selectPdfs(current, incoming) {
  const files = [...current];
  const keys = new Set(files.map(pdfKey));
  const errors = [];
  for (const file of incoming) {
    if (!/\.pdf$/i.test(file.name) || (file.type && file.type !== 'application/pdf')) {
      errors.push(`${file.name}: выберите файл PDF.`);
    } else if (!file.size) {
      errors.push(`${file.name}: файл пуст.`);
    } else if (file.size > MAX_PDF_BYTES) {
      errors.push(`${file.name}: размер больше 50 МБ.`);
    } else if (keys.has(pdfKey(file))) {
      continue;
    } else if (files.length >= MAX_PDFS) {
      errors.push('Можно выбрать не более 32 файлов. Удалите лишние и попробуйте снова.');
      break;
    } else {
      files.push(file);
      keys.add(pdfKey(file));
    }
  }
  return { files, errors };
}
