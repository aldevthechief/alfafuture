import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_PDF_BYTES, selectPdfs } from '../src/pdfSelection.js';

const pdf = (name='report.pdf', extra={}) => ({ name, size:1024, type:'application/pdf', lastModified:1, ...extra });

test('accepts PDF files, preserves existing selection, and skips duplicates', () => {
  const existing = pdf();
  const next = pdf('Overview.PDF', {type:''});
  const result = selectPdfs([existing], [pdf(), next]);
  assert.deepEqual(result, { files:[existing,next], errors:[] });
});

test('rejects wrong format, empty files, and oversized PDFs while keeping valid files', () => {
  const valid = pdf('valid.pdf', {size:MAX_PDF_BYTES});
  const result = selectPdfs([], [pdf('image.png'), pdf('fake.pdf',{type:'image/png'}), pdf('empty.pdf',{size:0}), pdf('large.pdf',{size:MAX_PDF_BYTES+1}), valid]);
  assert.deepEqual(result.files, [valid]);
  assert.equal(result.errors.length, 4);
});

test('enforces the total limit across multiple selections', () => {
  const current = Array.from({length:31}, (_,i)=>pdf(`${i}.pdf`));
  const result = selectPdfs(current, [pdf('31.pdf'), pdf('32.pdf')]);
  assert.equal(result.files.length,32);
  assert.equal(current.length,31);
  assert.equal(result.errors.length,1);
});
