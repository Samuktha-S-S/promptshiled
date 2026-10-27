/**
 * User-facing wording for the backend's structured error and status codes.
 * Every title corresponds to a code the backend actually returns.
 */

export const REQUEST_ERRORS = {
  database_unavailable: {
    title: 'Unable to connect to the database',
    hint: '', // the backend detail already says the message was not processed, and why
  },
  detector_unavailable: {
    title: 'Prompt detection is temporarily unavailable',
    hint: 'Your message was not checked, so it was not sent to the AI.',
  },
  invalid_input: { title: 'Message not accepted', hint: '' },
  network_error: {
    title: 'Cannot reach the PromptShield backend',
    hint: 'Check that the FastAPI server is running on port 8000.',
  },
  internal_error: { title: 'The request could not be processed', hint: '' },
  http_error: { title: 'The request was rejected', hint: '' },
};

export const LLM_ERRORS = {
  llm_not_configured: 'The AI response service is not configured on the server.',
  llm_auth_failed: 'The AI response service rejected the server’s credentials.',
  llm_rate_limited: 'The AI response service is receiving too many requests.',
  llm_overloaded: 'The AI model is temporarily overloaded.',
  llm_timeout: 'The AI response service took too long to answer.',
  llm_empty_response: 'The AI response service returned no answer.',
  llm_unavailable: 'The AI response service is unavailable.',
};

/** Codes for which re-sending the same message can reasonably succeed. */
export const RETRYABLE = new Set([
  'database_unavailable', 'detector_unavailable', 'network_error', 'internal_error',
  'llm_rate_limited', 'llm_overloaded', 'llm_timeout', 'llm_unavailable', 'llm_empty_response',
]);

export function describeRequestError(error) {
  const known = REQUEST_ERRORS[error.code] || REQUEST_ERRORS.http_error;
  return { title: known.title, hint: known.hint, detail: error.message };
}
