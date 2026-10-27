# LLM package – Gemini integration

Generates the assistant's reply with Google's Gemini API, **only after** the backend's security
gate has authorised the prompt. Nothing in this folder classifies prompts, and nothing in
`ml_detector/` imports this folder (enforced by `tests/test_detection.py`).

## Files

| File | Purpose |
|---|---|
| `gemini_client.py` | Creates the `google-genai` client, sends `generate_content` requests, extracts text, applies the timeout, maps provider errors to controlled exceptions, redacts the API key |
| `chat_service.py` | `ChatService.generate_reply(authorization, prompt, history)` – refuses to call Gemini without a valid `GenerationAuthorization` |
| `prompt_templates.py` | The trusted system instruction and the `ChatTurn` type for untrusted conversation turns |

## SDK and API

* Package: `google-genai` (Google Gen AI SDK for Python), tested with 2.33.0.
* Call: `client.models.generate_content(model=..., contents=[...], config=GenerateContentConfig(system_instruction=...))`.
  The newer stateful Interactions API also exists; `generate_content` is used because the
  backend rebuilds the allowed history from PostgreSQL on every request and does not want
  Google to store conversation state.
* Default model: `gemini-3.8-flash`. The Gemini model list (ai.google.dev/gemini-api/docs/models)
  recommended it for new projects when this was written. Override it with `GEMINI_MODEL`.

## Configuration (backend environment only)

```
GEMINI_API_KEY=            # required for responses; never sent to the browser
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TIMEOUT_SECONDS=30
```

The key is read in `GeminiConfig.from_env()`, is hidden in `repr()`, and is replaced by `***`
in every error message. `/api/health` only reports whether a key is configured.

## Execution flow

```
SecurityGate (backend)                     llm/
──────────────────────                     ────
decision == ALLOW ──► authorize_generation() ──► GenerationAuthorization(action="ALLOW")
                                                       │
                     ChatService.generate_reply ◄──────┘
                       ├─ check_authorization()  → refuses BLOCK, unapproved FLAG, or anything else
                       ├─ build_conversation(history, prompt)
                       └─ GeminiClient.generate(SYSTEM_INSTRUCTION, turns)
```

A FLAG can reach Gemini only through `POST /api/review/{id}` after an administrator approves it
(`GenerationAuthorization(action="FLAG", human_approved=True)`). A BLOCK can never be authorised.

## Prompt separation

* `SYSTEM_INSTRUCTION` is the only privileged text and travels in Gemini's `system_instruction` field.
* The user's prompt and earlier turns are sent as ordinary `user` / `model` contents and are
  never formatted into the system instruction.
* History only contains earlier turns whose detection decision allowed them **and** that Gemini
  answered. Blocked, flagged and failed prompts are never replayed.

## Error handling

| Situation | Exception | Message shown to the user |
|---|---|---|
| No `GEMINI_API_KEY` | `GeminiConfigurationError` | "The AI assistant is not configured on the server." |
| HTTP 401 / 403 / invalid key | `GeminiAuthenticationError` | "…could not authenticate with its provider." |
| HTTP 429 | `GeminiRateLimitError` | "…receiving too many requests…" |
| Timeout / HTTP 408 / 504 | `GeminiTimeoutError` | "…took too long to respond…" |
| Empty or safety-blocked response | `GeminiEmptyResponseError` | "…returned no answer…" |
| Anything else | `GeminiProviderError` | "…currently unavailable…" |

On any of these, no reply is invented: the API returns `llm_status: "error"` with the message
above, and nothing is stored as an assistant message.

## Testing

`tests/test_gemini.py` uses a fake SDK object (no network, no credentials) to check request
construction, role separation, every error mapping, key redaction and authorisation refusal.
A live smoke test is included but skipped unless you opt in:

```bash
# PowerShell
$env:GEMINI_API_KEY="..."; $env:RUN_LIVE_GEMINI="1"; python -m pytest tests/test_gemini.py -k live
```

## Limitations

* The live Gemini API has **not** been exercised by the automated tests (no key was available).
* No automatic retry on 429; the user is asked to try again.
* The system instruction reduces, but cannot eliminate, the effect of an injection the classifier missed.
