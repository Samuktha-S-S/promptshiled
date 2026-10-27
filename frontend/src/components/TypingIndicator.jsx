import React from 'react';
import { ShieldIcon } from './Icons.jsx';

export default function TypingIndicator() {
  return (
    <div className="message-row assistant" role="status" aria-live="polite">
      <div className="avatar assistant" aria-hidden="true"><ShieldIcon size={16} /></div>
      <div className="typing">
        <span className="typing-label">Checking your message and generating a reply</span>
        <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
      </div>
    </div>
  );
}
