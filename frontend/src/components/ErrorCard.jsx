import React from 'react';
import { AlertCircleIcon, RefreshIcon } from './Icons.jsx';

/**
 * A failed request. No classification or AI text is shown, because none exists.
 * Retry re-sends the same message only when the user clicks it.
 */
export default function ErrorCard({ title, hint, detail, onRetry, retrying = false, tone = 'error' }) {
  return (
    <div className={`error-card ${tone}`} role="alert">
      <div className="error-card-head">
        <AlertCircleIcon size={18} />
        <span className="error-card-title">{title}</span>
      </div>
      {hint && <p className="error-card-hint">{hint}</p>}
      {detail && detail !== hint && <p className="error-card-detail">{detail}</p>}
      {onRetry && (
        <button type="button" className="retry-btn" onClick={onRetry} disabled={retrying}>
          <RefreshIcon size={14} /> {retrying ? 'Retrying…' : 'Retry'}
        </button>
      )}
    </div>
  );
}
