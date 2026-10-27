import React from 'react';
import { ShieldAlertIcon, ShieldCheckIcon, AlertTriangleIcon } from './Icons.jsx';

const LABELS = {
  benign: 'Benign',
  prompt_injection: 'Prompt injection',
};

const ICONS = { ALLOW: ShieldCheckIcon, FLAG: AlertTriangleIcon, BLOCK: ShieldAlertIcon };

/** Compact decision chip. Shows the score only when the backend returned a valid number. */
export default function SecurityBadge({ action, classification, confidence }) {
  const Icon = ICONS[action] || ShieldAlertIcon;
  const hasScore = typeof confidence === 'number' && Number.isFinite(confidence);
  return (
    <span className={`security-badge ${action.toLowerCase()}`} data-testid="security-badge">
      <Icon size={14} />
      <span className="badge-action">{action}</span>
      {classification && <span className="badge-sep">·</span>}
      {classification && <span>{LABELS[classification] || classification}</span>}
      {hasScore && <span className="badge-score">{(confidence * 100).toFixed(1)}%</span>}
    </span>
  );
}
