import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatHeader from './components/ChatHeader.jsx';
import WelcomeScreen from './components/WelcomeScreen.jsx';
import ChatMessage from './components/ChatMessage.jsx';
import MessageComposer from './components/MessageComposer.jsx';
import TypingIndicator from './components/TypingIndicator.jsx';
import { BackendBanner } from './components/BackendStatus.jsx';
import useBackendHealth from './hooks/useBackendHealth.js';
import { sendMessage } from './services/api.js';
import { describeRequestError, RETRYABLE } from './lib/messages.js';

const MOBILE_QUERY = '(max-width: 900px)';
let nextId = 0;
const uid = (prefix) => `${prefix}-${Date.now().toString(36)}-${(nextId++).toString(36)}`;
const titleFrom = (text) => (text.length > 42 ? `${text.slice(0, 42).trimEnd()}…` : text);

function useIsMobile() {
  const [mobile, setMobile] = useState(() => window.matchMedia?.(MOBILE_QUERY).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(MOBILE_QUERY);
    if (!mq) return undefined;
    const onChange = (e) => setMobile(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return mobile;
}

export default function App() {
  const health = useBackendHealth();
  const isMobile = useIsMobile();
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  // {id, serverId, title, messages[]} - kept for this browser session only
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [securityEvents, setSecurityEvents] = useState([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const endRef = useRef(null);
  const composerRef = useRef(null);

  useEffect(() => setSidebarOpen(!isMobile), [isMobile]);

  const active = useMemo(() => conversations.find((c) => c.id === activeId) || null, [conversations, activeId]);
  const messages = active?.messages || [];

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' });
  }, [messages.length, busy]);

  const updateConversation = useCallback((id, fn) => {
    setConversations((list) => list.map((c) => (c.id === id ? fn(c) : c)));
  }, []);

  const closeSidebarOnMobile = () => isMobile && setSidebarOpen(false);

  /** Sends `text`. `retryOf` is the failed assistant message being retried (no new user bubble). */
  const send = useCallback(async (text, retryOf = null) => {
    if (busyRef.current || !text.trim()) return; // one request at a time: no duplicate submissions
    busyRef.current = true;
    setBusy(true);

    let convId = activeId;
    let serverId = active?.serverId ?? null;
    if (!convId) {
      convId = uid('conv');
      setConversations((list) => [{ id: convId, serverId: null, title: titleFrom(text), messages: [] }, ...list]);
      setActiveId(convId);
    }
    if (retryOf) {
      updateConversation(convId, (c) => ({
        ...c, messages: c.messages.map((m) => (m.id === retryOf.id ? { ...m, retried: true } : m)),
      }));
    } else {
      updateConversation(convId, (c) => ({ ...c, messages: [...c.messages, { id: uid('u'), role: 'user', content: text }] }));
      setDraft('');
    }

    try {
      const res = await sendMessage(text, serverId);
      const reply = {
        id: uid('a'), role: 'assistant', kind: 'result', prompt: text,
        content: res.response,
        action: res.action,
        classification: res.classification,
        confidence: res.confidence,
        p_injection: res.p_injection,
        score_type: res.score_type,
        classifier: res.classifier,
        reason: res.reason,
        llm_status: res.llm_status,
        llm_error_code: res.llm_error_code,
        latency_ms: res.latency_ms,
        event_id: res.event_id,
      };
      updateConversation(convId, (c) => ({ ...c, serverId: res.conversation_id, messages: [...c.messages, reply] }));
      if (res.action === 'FLAG' || res.action === 'BLOCK') {
        setSecurityEvents((list) => [
          { conversationId: convId, eventId: res.event_id, action: res.action, prompt: titleFrom(text) }, ...list,
        ]);
      }
    } catch (err) {
      const code = err.code || 'network_error';
      updateConversation(convId, (c) => ({
        ...c,
        messages: [...c.messages, {
          id: uid('e'), role: 'assistant', kind: 'error', prompt: text,
          error: describeRequestError(err), retryable: RETRYABLE.has(code),
        }],
      }));
      health.refresh(); // the failure may reflect a dependency outage; update the status indicator
    } finally {
      busyRef.current = false;
      setBusy(false);
      composerRef.current?.focus();
    }
  }, [activeId, active, updateConversation, health]);

  const retry = useCallback((message) => {
    if (!message.retried) send(message.prompt, message);
  }, [send]);

  const newChat = () => {
    setActiveId(null);
    setDraft('');
    closeSidebarOnMobile();
    composerRef.current?.focus();
  };

  const selectConversation = (id) => {
    setActiveId(id);
    closeSidebarOnMobile();
  };

  return (
    <div className={`app ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        conversations={conversations}
        activeId={activeId}
        onSelect={selectConversation}
        onNewChat={newChat}
        securityEvents={securityEvents}
        onSelectEvent={selectConversation}
      />

      <main className="main">
        <ChatHeader
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(true)}
          onNewChat={newChat}
          health={health}
        />
        <BackendBanner state={health.state} health={health.health} onRefresh={health.refresh} />

        <div className="chat-scroll">
          {messages.length === 0 && !busy ? (
            <WelcomeScreen onPick={(p) => { setDraft(p); composerRef.current?.focus(); }} />
          ) : (
            <div className="message-list">
              {messages.map((m) => (
                <ChatMessage key={m.id} message={m} busy={busy} onRetry={m.retried ? null : retry} />
              ))}
              {busy && <TypingIndicator />}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <MessageComposer ref={composerRef} value={draft} onChange={setDraft} onSubmit={(t) => send(t)} busy={busy} />
      </main>
    </div>
  );
}
