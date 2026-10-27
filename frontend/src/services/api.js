/**
 * PromptShield - API Service
 * Handles communication with the FastAPI backend.
 * The browser never talks to Gemini directly; all generation happens server-side.
 */

const API_BASE = '/api';

/** Error carrying the backend's structured fields: {detail, code, category}. */
export class ApiError extends Error {
  constructor(message, { status = 0, code = 'network_error', category = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.category = category;
  }
}

export async function sendMessage(message, conversationId = null) {
  let response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    });
  } catch {
    throw new ApiError('The PromptShield backend could not be reached.', { code: 'network_error' });
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    // FastAPI validation errors (422) return a list in `detail`
    const detail = typeof error.detail === 'string' ? error.detail : `Request rejected (HTTP ${response.status})`;
    const code = error.code || (response.status === 422 ? 'invalid_input' : 'http_error');
    throw new ApiError(detail, { status: response.status, code, category: error.category || null });
  }

  return response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) return { status: 'unhealthy' };
    return response.json();
  } catch {
    return { status: 'unreachable' };
  }
}
