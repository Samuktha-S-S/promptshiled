import React from 'react';
import { ShieldIcon, PlusIcon, ChatIcon, SidebarIcon, ShieldAlertIcon, AlertTriangleIcon } from './Icons.jsx';

/**
 * Conversations and security events from THIS browser session only.
 * The backend stores every message in PostgreSQL, but exposes no endpoint to
 * list conversations (there are no user accounts, so listing them would show
 * everyone's chats). The UI therefore does not claim to restore old chats.
 */
export default function Sidebar({
  open, onToggle, conversations, activeId, onSelect, onNewChat, securityEvents, onSelectEvent,
}) {
  return (
    <>
      <div className={`sidebar-scrim ${open ? 'show' : ''}`} onClick={onToggle} aria-hidden="true" />
      <aside className={`sidebar ${open ? 'open' : 'closed'}`} aria-label="Sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <span className="brand-logo"><ShieldIcon size={18} /></span>
            <div className="brand-text">
              <span className="brand-name">PromptShield</span>
              <span className="brand-sub">Semantic ML-Based Prompt Injection Security</span>
            </div>
          </div>
          <button type="button" className="icon-btn" onClick={onToggle} aria-label="Close sidebar"
            title="Close sidebar">
            <SidebarIcon />
          </button>
        </div>

        <button type="button" className="new-chat-btn" onClick={onNewChat}>
          <PlusIcon size={16} /> New chat
        </button>

        <nav className="sidebar-scroll">
          <h4 className="sidebar-heading">Recent chats <span className="sidebar-scope">this session</span></h4>
          {conversations.length === 0 ? (
            <p className="sidebar-empty">No conversations yet.</p>
          ) : (
            <ul className="conv-list">
              {conversations.map((c) => (
                <li key={c.id}>
                  <button type="button" className={`conv-item ${c.id === activeId ? 'active' : ''}`}
                    onClick={() => onSelect(c.id)} title={c.title}>
                    <ChatIcon size={15} />
                    <span className="conv-title">{c.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <h4 className="sidebar-heading">Security events <span className="sidebar-scope">this session</span></h4>
          {securityEvents.length === 0 ? (
            <p className="sidebar-empty">No flagged or blocked messages.</p>
          ) : (
            <ul className="event-list">
              {securityEvents.map((e) => (
                <li key={`${e.conversationId}-${e.eventId}`}>
                  <button type="button" className={`event-item ${e.action.toLowerCase()}`}
                    onClick={() => onSelectEvent(e.conversationId)} title={e.prompt}>
                    {e.action === 'BLOCK' ? <ShieldAlertIcon size={14} /> : <AlertTriangleIcon size={14} />}
                    <span className="event-action">{e.action}</span>
                    <span className="event-prompt">{e.prompt}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>

        <p className="sidebar-note">
          Messages are stored in PostgreSQL by the backend. This list is kept only for the current
          browser session and clears when the page is reloaded.
        </p>
      </aside>
    </>
  );
}
