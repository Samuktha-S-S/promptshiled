import React, { useState } from 'react';
import DetectionDetails from './DetectionDetails.jsx';
import SecurityBadge from './SecurityBadge.jsx';
import { ShieldAlertIcon, AlertTriangleIcon, ChevronDownIcon } from './Icons.jsx';

const CONTENT = {
  BLOCK: {
    Icon: ShieldAlertIcon,
    title: 'Request blocked',
    note: 'This message was not sent to the AI model.',
  },
  FLAG: {
    Icon: AlertTriangleIcon,
    title: 'Request flagged for review',
    note: 'This message was not sent to the AI model. It is held until an administrator reviews it.',
  },
};

/** Amber (FLAG) or red (BLOCK) card. Only shows text and scores returned by the backend. */
export default function SecurityCard({ message }) {
  const [open, setOpen] = useState(false);
  const { Icon, title, note } = CONTENT[message.action];
  return (
    <div className={`security-card ${message.action.toLowerCase()}`} role="status">
      <div className="security-card-head">
        <span className="security-card-icon"><Icon size={18} /></span>
        <div>
          <div className="security-card-title">{title}</div>
          <div className="security-card-note">{note}</div>
        </div>
      </div>
      <p className="security-card-text">{message.content}</p>
      <div className="message-meta">
        <SecurityBadge action={message.action} classification={message.classification} confidence={message.confidence} />
        {message.event_id != null && <span className="meta-muted">Event #{message.event_id}</span>}
        <button type="button" className={`details-toggle ${open ? 'open' : ''}`} onClick={() => setOpen(!open)}
          aria-expanded={open}>
          Details <ChevronDownIcon size={14} />
        </button>
      </div>
      {open && <DetectionDetails message={message} />}
    </div>
  );
}
