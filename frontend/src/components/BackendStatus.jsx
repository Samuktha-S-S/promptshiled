import React, { useEffect, useRef, useState } from 'react';
import { healthProblems } from '../hooks/useBackendHealth.js';
import { RefreshIcon } from './Icons.jsx';

const LABELS = { checking: 'Checking…', healthy: 'Healthy', degraded: 'Degraded', offline: 'Offline' };

/** Status pill with a popover listing each dependency, as reported by /api/health. */
export function BackendStatusPill({ state, health, checkedAt, onRefresh }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const esc = (e) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc); };
  }, [open]);

  const components = health?.components;
  return (
    <div className="status-wrap" ref={ref}>
      <button type="button" className={`status-pill ${state}`} onClick={() => setOpen(!open)}
        aria-expanded={open} aria-haspopup="dialog" aria-label={`Backend: ${LABELS[state]}`}
        data-testid="backend-status">
        <span className="status-dot" aria-hidden="true" />
        <span className="status-label">Backend: {LABELS[state]}</span>
      </button>
      {open && (
        <div className="status-popover" role="dialog" aria-label="Backend status">
          {state === 'offline' && <p className="status-row-msg">The API at /api/health did not respond.</p>}
          {components && (
            <ul className="status-list">
              {[['Database', components.database], ['ML detector', components.detector], ['Gemini', components.llm]]
                .map(([label, c]) => {
                  const ok = label === 'Gemini' ? c.status === 'configured' : c.ok;
                  return (
                    <li key={label}>
                      <span className={`dot ${ok ? 'ok' : label === 'Gemini' ? 'warn' : 'bad'}`} aria-hidden="true" />
                      <div>
                        <div className="status-row-title">{label}{label === 'ML detector' && health.classifier
                          ? ` (${health.classifier.replace('_', ' ')})` : ''}</div>
                        <div className="status-row-msg">{c.message}</div>
                      </div>
                    </li>
                  );
                })}
            </ul>
          )}
          <div className="status-footer">
            <span>{checkedAt ? `Checked ${checkedAt.toLocaleTimeString()}` : 'Not checked yet'}</span>
            <button type="button" className="link-btn" onClick={onRefresh}><RefreshIcon size={13} /> Recheck</button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Banner above the conversation when a required dependency is down. */
export function BackendBanner({ state, health, onRefresh }) {
  if (state === 'healthy' || state === 'checking') return null;
  const problems = state === 'offline'
    ? [{ level: 'error', label: 'Backend', message: 'The PromptShield API is not responding. Start it with uvicorn on port 8000.' }]
    : healthProblems(health).filter((p) => p.level === 'error');
  if (!problems.length) return null;
  return (
    <div className="backend-banner" role="alert">
      <div>
        {problems.map((p) => (
          <p key={p.label}><strong>{p.label}:</strong> {p.message}</p>
        ))}
        <p className="banner-hint">Messages cannot be processed until this is fixed.</p>
      </div>
      <button type="button" className="retry-btn" onClick={onRefresh}><RefreshIcon size={14} /> Retry</button>
    </div>
  );
}
