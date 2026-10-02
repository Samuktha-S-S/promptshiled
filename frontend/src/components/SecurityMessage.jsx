import React, { useState } from 'react';

export default function SecurityMessage({ action, classification, confidence, response }) {
  const [showDetails, setShowDetails] = useState(false);

  const isFlag = action === 'FLAG';
  const isBlock = action === 'BLOCK';
  const className = isBlock ? 'block' : 'flag';

  return (
    <div className={`security-message ${className}`}>
      <span className="security-icon">
        {isBlock ? '🛡️' : '⚠️'}
      </span>
      <strong>{isBlock ? 'Request Blocked' : 'Request Flagged'}</strong>
      <span className="security-text">{response}</span>

      <button
        className="security-details-toggle"
        onClick={() => setShowDetails(!showDetails)}
        id={`security-details-toggle-${Date.now()}`}
      >
        {showDetails ? '▲ Hide details' : '▼ View details'}
      </button>

      {showDetails && (
        <div className="security-details">
          <div className="detail-row">
            <span className="detail-label">Classification</span>
            <span className="detail-value">{classification}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Confidence</span>
            <span className="detail-value">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Action</span>
            <span className={`detail-value ${action.toLowerCase()}`}>{action}</span>
          </div>
        </div>
      )}
    </div>
  );
}
