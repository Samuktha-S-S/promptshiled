import React, { useEffect, useImperativeHandle, useRef, forwardRef } from 'react';
import { SendIcon } from './Icons.jsx';

export const MAX_LENGTH = 4096; // matches the backend's ChatRequest limit
const MAX_HEIGHT = 200;

/**
 * Chat input: Enter sends, Shift+Enter inserts a newline, grows with content,
 * and cannot submit while a request is in flight or the text is empty.
 */
const MessageComposer = forwardRef(function MessageComposer({ value, onChange, onSubmit, busy }, ref) {
  const textareaRef = useRef(null);
  useImperativeHandle(ref, () => ({ focus: () => textareaRef.current?.focus() }));

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  const trimmed = value.trim();
  const tooLong = value.length > MAX_LENGTH;
  const canSend = trimmed.length > 0 && !busy && !tooLong;

  const submit = (e) => {
    e?.preventDefault();
    if (canSend) onSubmit(trimmed);
  };

  return (
    <form className="composer" onSubmit={submit}>
      <div className={`composer-box ${busy ? 'busy' : ''}`}>
        <label htmlFor="chat-input" className="visually-hidden">Message PromptShield</label>
        <textarea
          id="chat-input"
          ref={textareaRef}
          className="composer-input"
          value={value}
          rows={1}
          placeholder="Message PromptShield…"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button type="submit" className="send-btn" disabled={!canSend}
          aria-label={busy ? 'Waiting for response' : 'Send message'} id="send-message-btn">
          {busy ? <span className="spinner" aria-hidden="true" /> : <SendIcon size={18} />}
        </button>
      </div>
      <div className="composer-caption">
        <span>Protected by PromptShield semantic ML detection.</span>
        {value.length > MAX_LENGTH * 0.8 && (
          <span className={tooLong ? 'count over' : 'count'}>{value.length} / {MAX_LENGTH}</span>
        )}
      </div>
    </form>
  );
});

export default MessageComposer;
