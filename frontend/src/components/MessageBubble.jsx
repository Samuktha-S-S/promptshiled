import React from 'react';
import SecurityMessage from './SecurityMessage.jsx';

export default function MessageBubble({ message }) {
  const { role, content, action, classification, confidence } = message;
  const isUser = role === 'user';
  const isSecurityAction = action === 'FLAG' || action === 'BLOCK';

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`message-avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? '👤' : '🛡️'}
      </div>

      {isUser ? (
        <div className="message-content user">{content}</div>
      ) : isSecurityAction ? (
        <SecurityMessage
          action={action}
          classification={classification}
          confidence={confidence}
          response={content}
        />
      ) : (
        <div className="message-content assistant">
          {content.split('\n').map((line, i) => {
            if (!line.trim()) return <br key={i} />;
            // Simple markdown bold support
            const parts = line.split(/\*\*(.*?)\*\*/g);
            return (
              <p key={i}>
                {parts.map((part, j) =>
                  j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                )}
              </p>
            );
          })}
        </div>
      )}
    </div>
  );
}
