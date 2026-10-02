/**
 * PromptShield - API Service
 * Handles communication with the FastAPI backend.
 */

const API_BASE = '/api';

export async function sendMessage(message) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
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
