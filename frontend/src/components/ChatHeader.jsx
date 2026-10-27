import React from 'react';
import { SidebarIcon, PlusIcon } from './Icons.jsx';
import { BackendStatusPill } from './BackendStatus.jsx';

export default function ChatHeader({ sidebarOpen, onToggleSidebar, onNewChat, health }) {
  return (
    <header className="chat-header">
      <div className="chat-header-left">
        {!sidebarOpen && (
          <>
            <button type="button" className="icon-btn" onClick={onToggleSidebar} aria-label="Open sidebar"
              title="Open sidebar">
              <SidebarIcon />
            </button>
            <button type="button" className="icon-btn" onClick={onNewChat} aria-label="New chat" title="New chat">
              <PlusIcon />
            </button>
          </>
        )}
        <div className="chat-header-title">
          <h1>PromptShield</h1>
          <span className="chat-header-sub">Semantic ML-Based Prompt Injection Security</span>
        </div>
      </div>
      <BackendStatusPill {...health} onRefresh={health.refresh} />
    </header>
  );
}
