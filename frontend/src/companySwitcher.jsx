import React, { useEffect, useRef } from 'react';
import { accountSession } from './accountSession.js';
import './companySwitcher.css';

export function CompanySwitcher({ accounts, session, onEnter, onManage, onClose }) {
  const menu = useRef(null);
  useEffect(() => {
    const trigger = document.getElementById('whoBtn');
    trigger.setAttribute('aria-expanded', 'true');
    menu.current?.querySelector('button')?.focus();
    const outside = event => { if (!event.target.closest('.who')) onClose(); };
    const escape = event => { if (event.key === 'Escape') { onClose(); trigger.focus(); } };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => {
      trigger.setAttribute('aria-expanded', 'false');
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
    };
  }, [onClose]);
  function navigate(event) {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    const buttons = [...menu.current.querySelectorAll('button')];
    const index = buttons.indexOf(document.activeElement);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
    event.preventDefault(); buttons[next].focus();
  }
  return <div className="company-switcher" ref={menu} role="menu" aria-label="Компании и кабинеты" onKeyDown={navigate}>
    <p className="company-switcher-caption">Ваши кабинеты</p>
    {accounts.map(account => {
      const active = account.company_id === session.company_id && account.role === session.role;
      return <button key={account.id} type="button" role="menuitemradio" aria-checked={active}
        onClick={() => { if (!active) onEnter(accountSession(account)); onClose(); document.getElementById('whoBtn').focus(); }}>
        <span className={`company-switcher-avatar entry-avatar-${account.color || 'blue'}`}>{account.initials || 'К'}</span>
        <span><strong>{accountSession(account).company.name}</strong><small>{account.role_title}</small></span>
        {active && <span className="company-switcher-check" aria-hidden="true">✓</span>}
      </button>;
    })}
    <button className="company-switcher-login" type="button" role="menuitem" onClick={onManage}>Войти в другой аккаунт<span aria-hidden="true">→</span></button>
  </div>;
}
