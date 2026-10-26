import React from 'react';
import './insightAssessment.css';

export function InsightAssessment({ insight }) {
  const basis=insight.confidence_basis || [];
  const source=insight.credibility?.label;
  if (!basis.length && !source) return null;
  return <details className="insight-assessment">
    <summary>Почему такая оценка уверенности</summary>
    {basis.length>0 && <ul>{basis.map((reason,index)=><li key={index}>{reason}</li>)}</ul>}
    {source && !basis.includes(source) && <p>{source}</p>}
    <p className="assessment-note">Оценка учитывает доступные источники и допущения. Она не гарантирует наступление прогноза.</p>
  </details>;
}
