export const CABINET_MODE = import.meta.env?.VITE_CABINET_MODE || 'full';
export const ALLOW_INFERENCE = CABINET_MODE === 'full';
export const SESSION_KEY = `alfa-cabinet:${CABINET_MODE}:v1`;

export function validSession(session) {
  return !!session && typeof session.company_id === 'string' && typeof session.role === 'string'
    && typeof session.job_id === 'string' && !!session.company?.name
    && (session.kind === 'cached' || (ALLOW_INFERENCE && session.kind === 'live'));
}
