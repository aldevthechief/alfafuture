import { useEffect, useRef, useState } from 'react';
import { compactOf } from './api.js';

export const compactKey = card => JSON.stringify([card.job_id, card.id]);

// A cabinet can merge several saved jobs. Fetch each job's projection once per set of ready cards.
export function useCompactFeed(cards) {
  const [versions, setVersions] = useState({});
  const requested = useRef(new Map());
  const signature = JSON.stringify(cards.map(card => [card.job_id, card.id]).filter(([job]) => job).sort());
  useEffect(() => {
    let active = true;
    const groups = new Map();
    for (const [job, id] of JSON.parse(signature)) groups.set(job, [...(groups.get(job) || []), id]);
    for (const [job, ids] of groups) {
      const key = JSON.stringify([job, ids]);
      if (!requested.current.has(key)) {
        requested.current.set(key, compactOf(job).catch(error => {
          requested.current.delete(key);
          console.warn('Краткая версия инсайта недоступна:', error.message);
          return {};
        }));
      }
      requested.current.get(key).then(items => {
        if (active) setVersions(previous => ({ ...previous,
          ...Object.fromEntries(ids.filter(id => items[id]).map(id => [JSON.stringify([job, id]), items[id]])),
        }));
      });
    }
    return () => { active = false; };
  }, [signature]);
  return versions;
}
