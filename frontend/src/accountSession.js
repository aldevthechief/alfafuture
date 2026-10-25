import { validSession } from './cabinetMode.js';

export function companyLabel(company, id = company.company_id || company.id) {
  const activity = company.sector || company.industry || 'Компания';
  return `${activity} №${String(id).replace(/^U_/, '')}`;
}

export const accountSession = account => ({ kind: account.kind || 'cached', account_id: account.id, company_id: account.company_id,
  role: account.role, role_title: account.role_title, job_id: account.job_id,
  company: { ...account, name: companyLabel(account) },
});

export const sessionAccount = session => ({ ...session.company, id: session.account_id || `${session.company_id}:${session.role}`,
  company_id: session.company_id, role: session.role, role_title: session.role_title, job_id: session.job_id, kind: session.kind,
});

export function initialSession(saved, accounts) {
  if (validSession(saved)) {
    if (saved.kind === 'live') return { ...saved, company: { ...saved.company, name: companyLabel(saved.company, saved.company_id) } };
    const account = accounts.find(item => item.id === saved.account_id || (item.company_id === saved.company_id && item.role === saved.role));
    if (account) return accountSession(account);
  }
  return accounts.length ? accountSession(accounts[0]) : null;
}
