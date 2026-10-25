import React from 'react';

export function ThemeToggle({ theme, onToggle }) {
  const dark = theme === 'dark';
  return <button type="button" className="theme-toggle" onClick={onToggle} aria-pressed={dark}
    aria-label="Тёмная тема" title={dark ? 'Включить светлую тему' : 'Включить тёмную тему'}>
    <span aria-hidden="true">{dark ? '☀️' : '🌙'}</span>
    <span>{dark ? 'Светлая тема' : 'Тёмная тема'}</span>
  </button>;
}
