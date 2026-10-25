import '@alfalab/core-components-vars/index.css';
import darkTokens from '@alfalab/core-components-themes/dark.css?raw';

// Scope the design system's own dark palette to html so portals inherit it too.
const STYLE_ID = 'alfa-dark-theme';
export function applyTheme(theme) {
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = darkTokens.replaceAll(':root', ':root[data-theme="dark"]');
    document.head.append(style);
  }
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  try { localStorage.setItem('alfa-insights-theme', theme); } catch {}
}
