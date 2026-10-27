import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth } from '../services/api.js';

const POLL_MS = 30000;

/**
 * Live backend state derived from GET /api/health.
 * state: 'checking' | 'healthy' | 'degraded' | 'offline'
 * Polls every 30 s while the tab is visible, plus on demand via refresh().
 */
export default function useBackendHealth() {
  const [state, setState] = useState('checking');
  const [health, setHealth] = useState(null);
  const [checkedAt, setCheckedAt] = useState(null);
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setState((s) => (s === 'offline' || s === 'checking' ? 'checking' : s));
    const res = await checkHealth();
    inFlight.current = false;
    setCheckedAt(new Date());
    if (res.status === 'healthy' || res.status === 'degraded') {
      setHealth(res);
      setState(res.status);
    } else {
      setHealth(null);
      setState('offline');
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') refresh();
    }, POLL_MS);
    const onVisible = () => document.visibilityState === 'visible' && refresh();
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [refresh]);

  return { state, health, checkedAt, refresh };
}

/** Human-readable problems from a health payload (required dependencies first). */
export function healthProblems(health) {
  if (!health || !health.components) return [];
  const problems = [];
  const { database, detector, llm } = health.components;
  if (database && !database.ok) problems.push({ level: 'error', label: 'Database', message: database.message });
  if (detector && !detector.ok) problems.push({ level: 'error', label: 'ML detector', message: detector.message });
  if (llm && llm.status !== 'configured') problems.push({ level: 'warning', label: 'Gemini', message: llm.message });
  return problems;
}
