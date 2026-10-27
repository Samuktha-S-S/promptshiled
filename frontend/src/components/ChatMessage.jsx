import React, { useState } from 'react';
import MarkdownText from './MarkdownText.jsx';
import SecurityBadge from './SecurityBadge.jsx';
import SecurityCard from './SecurityCard.jsx';
import ErrorCard from './ErrorCard.jsx';
import DetectionDetails from './DetectionDetails.jsx';
import { ShieldIcon, ChevronDownIcon } from './Icons.jsx';
import { LLM_ERRORS, RETRYABLE } from '../lib/messages.js';

function AssistantAvatar() {
  return <div className="avatar assistant" aria-hidden="true"><ShieldIcon size={16} /></div>;
}

function AllowedReply({ message, onRetry, busy }) {
  const [open, setOpen] = useState(false);
  const llmFailed = message.llm_status === 'error';
  const code = message.llm_error_code || 'llm_unavailable';
  return (
    <>
      {llmFailed ? (
        <ErrorCard
          tone="warning"
          title={LLM_ERRORS[code] || LLM_ERRORS.llm_unavailable}
          hint="Your message passed the security check, but no AI response was generated."
          detail={message.content}
          onRetry={RETRYABLE.has(code) && onRetry ? () => onRetry(message) : null}
          retrying={busy}
        />
      ) : (
        <MarkdownText content={message.content} />
      )}
      <div className="message-meta">
        <SecurityBadge action="ALLOW" classification={message.classification} confidence={message.confidence} />
        <button type="button" className={`details-toggle ${open ? 'open' : ''}`} onClick={() => setOpen(!open)}
          aria-expanded={open}>
          Details <ChevronDownIcon size={14} />
        </button>
      </div>
      {open && <DetectionDetails message={message} />}
    </>
  );
}

/**
 * One row of the conversation.
 *  - user:                 right-aligned bubble
 *  - assistant / result:   ALLOW reply (or LLM failure notice), FLAG / BLOCK card
 *  - assistant / error:    request failed before a decision existed
 *  - assistant / info:     static UI text (no classification attached)
 */
export default function ChatMessage({ message, onRetry, busy }) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="user-bubble">{message.content}</div>
      </div>
    );
  }

  let body;
  if (message.kind === 'error') {
    body = (
      <ErrorCard
        title={message.error.title}
        hint={message.error.hint}
        detail={message.error.detail}
        onRetry={message.retryable && onRetry ? () => onRetry(message) : null}
        retrying={busy}
      />
    );
  } else if (message.action === 'FLAG' || message.action === 'BLOCK') {
    body = <SecurityCard message={message} />;
  } else if (message.action === 'ALLOW') {
    body = <AllowedReply message={message} onRetry={onRetry} busy={busy} />;
  } else {
    body = <MarkdownText content={message.content} />;
  }

  return (
    <div className="message-row assistant">
      <AssistantAvatar />
      <div className="assistant-body">{body}</div>
    </div>
  );
}
