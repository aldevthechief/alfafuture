import React, { useRef, useState } from 'react';
import { InsightAssessment } from './insightAssessment.jsx';
import { InsightIcon } from './insightIcons.jsx';
import { view } from './insightView.js';
import { ExcerptAction, useExcerptSelection } from './excerptSelection.jsx';

export function QuotedExcerpt({ text, source }) {
  if (!text) return null;
  return <blockquote className="chat-quoted-excerpt"><span>{source === 'answer' ? 'Фрагмент ответа агента' : 'Фрагмент инсайта'}</span><p>{text}</p></blockquote>;
}

const ReaderBlock = ({ title, items, ordered }) => {
  if (!items?.length) return null;
  const List = ordered ? 'ol' : 'ul';
  return <section className="insight-reader-section"><h4>{title}</h4><List>{items.map((item, index) =>
    <li key={index}>{typeof item === 'string' ? item : <>{item.label && <strong>{item.label}. </strong>}{item.text}</>}</li>)}</List></section>;
};

export function InsightReader({ insight, compact, onQuote, onClose, canAsk, children }) {
  const [full, setFull] = useState(false);
  const panel = useRef(null);
  const content = useRef(null);
  const [selection, clearSelection] = useExcerptSelection({ root:content, anchor:panel, enabled:canAsk });
  const v = view(insight, full ? null : compact);

  return <aside className="insight-reader" ref={panel} aria-label="Текст инсайта">
    <header className="insight-reader-header"><span><InsightIcon name="document" size={17} />Текст инсайта</span>
      <button type="button" className="workbench-icon-button" onClick={onClose} aria-label="Закрыть текст инсайта"><InsightIcon name="close" size={17} /></button>
    </header>
    {compact && <div className="insight-reader-modes" role="group" aria-label="Версия текста">
      <button type="button" aria-pressed={!full} onClick={() => { setFull(false); clearSelection(); }}>Кратко</button>
      <button type="button" aria-pressed={full} onClick={() => { setFull(true); clearSelection(); }}>Полностью</button>
    </div>}
    <div className="insight-reader-scroll" tabIndex={0} aria-label="Материалы инсайта" onScroll={clearSelection}>
      <div className="insight-reader-copy" ref={content}>
        <h3>{v.title}</h3>
        <InsightAssessment insight={insight} />
        {compact && !full && v.lead && <p className="insight-reader-lead">{v.lead}</p>}
        <ReaderBlock title="Что произошло" items={v.facts} />
        <ReaderBlock title="Почему это важно" items={v.why} />
        <ReaderBlock title="Особенно важно, если" items={v.important_if} />
        <ReaderBlock title="Риски" items={v.risks} />
        <ReaderBlock title="Возможности" items={v.opportunities} />
        <ReaderBlock title="Что сделать" items={v.actions} ordered />
        <ReaderBlock title="Позиция Банка" items={v.bank_view} />
        {(!compact || full) && insight.ai_conclusion && <section className="insight-reader-section"><h4>Итог AI</h4><p>{insight.ai_conclusion}</p></section>}
        {insight.bank_product && <section className="insight-reader-section"><h4>Чем может помочь Банк</h4><p>{insight.bank_product}</p>
          {insight.bank_product_reason && <p className="bank-product-reason">Почему это предложено: {insight.bank_product_reason}</p>}</section>}
      </div>
    </div>
    {children && <div className="insight-reader-materials">{children}</div>}
    <p className="insight-reader-hint">{canAsk ? 'Выделите фразу, чтобы задать вопрос именно о ней.' : 'Читайте материалы, не закрывая диалог.'}</p>
    <ExcerptAction selection={selection} onQuote={text => { onQuote(text); clearSelection(); }} />
  </aside>;
}
