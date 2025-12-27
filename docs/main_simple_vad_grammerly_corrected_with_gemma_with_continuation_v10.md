# main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py

## Overview
This module implements a FastAPI-based speech-to-text service with on-demand feedback generation. It performs continuous audio transcription using OpenAI Whisper, a simple energy-based VAD (voice activity detection), optional text refinement and feedback generation via an Ollama model, and session history persistence to `session_history.json`.

Key features:
- WebSocket audio ingestion for continuous streaming transcription.
- Energy-based VAD to skip silent segments.
- Whisper transcription (`whisper` library) for speech-to-text.
- Optional text refinement and feedback generation using Ollama (`ollama.Client`).
- Thread-safe session history manager with JSON persistence.
- On-demand feedback generation endpoint for saved sessions.
- Graceful shutdown that finalizes active sessions.

## Files
- Module: `main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py`
- Session history: `session_history.json`

## Requirements (principal packages)
- Python 3.10+
- fastapi
- uvicorn
- whisper
- numpy
- ollama (optional — required for refinement/feedback)

Install example:

```bash
pip install fastapi uvicorn numpy whisper ollama
```

(Adjust packages to your environment and use your virtualenv `v-speech-to-text`.)

## How to run
Run the module directly (it starts a Uvicorn server):

```bash
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py
# or use uvicorn explicitly
uvicorn main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10:app --host 0.0.0.0 --port 8000
```

Notes:
- The refinement/feedback features require an Ollama instance at `http://localhost:11434` and a model (default `phi3:mini`). If Ollama is unavailable the refinement engine disables itself and the app keeps running.
- Whisper model `base` is loaded on startup; ensure sufficient local resources.

## HTTP API Endpoints
- GET `/` — Service basic info and session stats.
- GET `/health` — Health status, current connections, total sessions.
- GET `/api/sessions` — Return all saved sessions (full list).
- GET `/api/sessions/recent?limit=10` — Recent N sessions (default 10).
- GET `/api/sessions/{session_id}` — Return a specific session by ID.
- POST `/api/sessions/{session_id}/generate-feedback` — Generate and save speaking feedback for the given session on-demand. Returns `GenerateFeedbackResponse`.
- GET `/api/stats` — Returns aggregated session statistics.

WebSocket:
- `ws://<host>:8000/ws/audio/{client_id}?mode=<raw|refined|full|all>` — Audio streaming endpoint. The server expects raw PCM bytes (16-bit, 16kHz, mono) via repeated `websocket.send` operations. The `mode` query controls the output format.

Messages from server (JSON):
- `type: "connection"` — initial readiness message.
- `type: "transcription"` — interim or final transcription payloads with keys such as `text`, `raw_text`, `refined_text`, `speaking_feedback`, `session_id`, `stats`, and flags `interim`/`skipped`.

## Data models
- `GenerateFeedbackRequest` (incoming model) — contains `session_id: int` (not currently used as body in the endpoint; route param used instead).
- `GenerateFeedbackResponse` (response model) — fields: `session_id`, `speaking_feedback`, `timestamp`, `status`.

## Main Components and Responsibilities
- `SessionHistoryManager`:
  - Persists sessions to `session_history.json`.
  - Thread-safe operations using `threading.Lock`.
  - Methods: `_load_history()`, `_save_history()`, `add_session()`, `update_session_feedback()`, `get_all_sessions()`, `get_recent_sessions()`, `get_session_by_id()`, `get_stats()`, `invalidate_stats_cache()`.
  - Sessions have fields: `session_id`, `timestamp`, `timestamp_readable`, `client_id`, `raw_text`, `refined_text`, `speaking_feedback`, `feedback_generated_at`, and `character_counts` (raw/refined/feedback lengths).

- `SimpleVAD`:
  - Energy-based voice activity detection using RMS energy calculation on PCM 16-bit audio.
  - `energy_threshold` is configurable (default 0.01 in class, set to 0.02 in `ConnectionManager`).
  - `has_speech(wav_path: str) -> bool` returns True when speech detected.

- `EnhancedRefinementEngine`:
  - Wraps an Ollama `Client` for text refinement and feedback generation.
  - Lazy async-safe locking with `asyncio.Lock()` to prevent concurrent Ollama requests.
  - `refine_text(raw_text)` returns a cleaned/refined version of the raw transcription.
  - `generate_feedback(raw_text)` returns constructive speaking feedback (3–5 points).
  - If Ollama is unavailable the engine disables itself and methods return fallbacks.

- `ContinuousAudioProcessor`:
  - Buffers incoming PCM chunks, periodically writes a temp WAV, runs VAD, and transcribes with Whisper.
  - Accumulates interim transcriptions and finalizes into a saved session when silence is detected (or `force_finalize` called).
  - Uses `output_mode` (`OutputMode.RAW_ONLY`, `REFINED_ONLY`, `REFINED_WITH_FEEDBACK`, `ALL`) to determine what is emitted/saved.
  - Saves sessions via `SessionHistoryManager.add_session()` without generating feedback automatically.

- `ConnectionManager`:
  - Loads shared resources on startup: Whisper model, VAD, refinement engine, and history manager.
  - Maintains `connections` (websockets) and `processors` (one `ContinuousAudioProcessor` per client).
  - Handles incoming audio via `handle_audio(client_id, audio_data)` and cleans up on disconnect.

## Session lifecycle and feedback generation
- While a client streams audio, the server transcribes interim audio and returns interim transcriptions.
- When sustained silence is detected (configurable threshold), the processor finalizes the session and saves `raw_text` and optionally `refined_text`.
- Speaking feedback is NOT generated automatically to avoid latency and cost; use the HTTP endpoint `POST /api/sessions/{id}/generate-feedback` to generate and persist feedback for a saved session.

## Tuning & operational notes
- VAD energy threshold: increase to be more aggressive at ignoring noise/silence, decrease to capture softer speech.
- Whisper model: `base` is used for balanced performance; swapping to `small`/`tiny` speeds up but reduces accuracy.
- Ollama: ensure an Ollama server is running and the chosen model (`phi3:mini`) is available to enable refinement and feedback.
- Concurrency: Ollama client calls are protected with an asyncio lock to prevent concurrent generate calls from corrupting the client state.
- Persistence: `SessionHistoryManager` writes the entire sessions list to `session_history.json` on each modification — acceptable for small to moderate volumes; consider a DB for scale.

## Graceful shutdown
On FastAPI shutdown, the server finalizes active processors (calls `force_finalize`) and persists any accumulated text.

## Quick example
Start the server:

```bash
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py
```

Open a WebSocket client, connect to `ws://localhost:8000/ws/audio/myclient1?mode=all` and send 16kHz mono 16-bit PCM audio bytes. Listen for `transcription` messages and call the feedback endpoint later if desired.

## Where I placed this doc
Created: `docs/main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.md`

---
If you want, I can:
- Add inline docstrings to the module and classes (`apply_patch`).
- Add a short `README.md` entry linking to this doc.
- Generate a simplified OpenAPI snippet for the endpoints.
