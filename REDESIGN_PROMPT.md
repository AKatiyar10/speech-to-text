# Role: Principal Full-Stack Architect

## Objective
Rebuild an existing local Speech-to-Text application from the ground up. The goal is to transform a prototype script into a production-grade, modular, and aesthetically premium application.

## Core Features (Must-Have)
1.  **Local-First Architecture**: All processing (STT, VAD, Diarization, LLM) must happen locally. No cloud APIs.
2.  **Real-Time Transcription**:
    *   **Input**: Microphone audio stream.
    *   **VAD**: Silero VAD to detect speech segments.
    *   **STT**: OpenAI Whisper (Local) for transcription.
3.  **Speaker Diarization**:
    *   **Engine**: Resemblyzer.
    *   **Flow**:
        *   Extract voice embeddings (256-dim).
        *   Compare cosine similarity (0.75 threshold).
        *   Match against enrolled speakers or assign "UNKNOWN_XX".
        *   **UI**: Allow users to click "UNKNOWN_XX" and rename/color-code (Enrollment).
4.  **Dual-Mode AI Refinement**:
    *   **Abstraction Layer**: Create a generic `LLMProvider` interface.
    *   **Implementations**:
        *   `OllamaProvider` (Local Gemma).
        *   `OpenAIProvider` (Public API).
    *   **Configuration**: Switchable via `.env` (e.g., `LLM_PROVIDER=openai`).
    *   **Task**: Correct grammar and punctuation of raw transcripts.
    *   **Context**: Pass the last 10-12 turns of conversation to the LLM for context-aware correction.
5.  **Session Management**:
    *   Persist conversation history in Markdown/JSON.
    *   Allow loading/viewing past sessions.

## Technical Stack & Architecture

### Backend (Python)
*   **Framework**: FastAPI (Async).
*   **Structure**: Clean Architecture / Domain-Driven Design.
    *   `app/core/`: Config, Logging (Rich), Exceptions.
    *   `app/api/`: Routers (v1/endpoints/...).
    *   `app/services/`:
        *   `AudioService` (VAD, buffering).
        *   `TranscriptionService` (Whisper).
        *   `DiarizationService` (Resemblyzer, Embedding storage).
        *   `RefinementService` (Ollama client).
    *   `app/schemas/`: Pydantic models (Strict typing).
    *   `app/websockets/`: Manager for real-time client comms.
*   **Quality**: 100% Type Hinting, Async/Await wherever possible, structured logging.

### Frontend (Modern Web)
*   **Framework**: React (Vite) OR Next.js (App Router).
*   **Language**: TypeScript.
*   **Styling**: TailwindCSS.
*   **Design System**: "Premium" feel. Dark mode, glassmorphism, smooth framer-motion animations.
    *   *Constraint*: Must look "State of the Art". No basic Bootstrap/Material look.
*   **Components**:
    *   `LiveTranscript`: Chat-bubble interface with speaker badges.
    *   `SpeakerManager`: Modal for enrolling/renaming speakers.
    *   `AudioVisualizer`: Real-time waveform/frequency bars.

## Deliverables
1.  **Directory Structure**: A tree view of the new organization.
2.  **Step-by-Step Implementation Plan**:
    *   Phase 1: Backend Core & Services.
    *   Phase 2: API & Websocket Layer.
    *   Phase 3: Frontend Foundation & Design System.
    *   Phase 4: Integration & polish.
3.  **Code Scaffolding**: Essential files (`main.py`, `config.py`, `App.tsx`, `useWebSocket.ts`).

## Constraints
*   **Performance**: Diarization must not block the main WebSocket loop (use `asyncio.to_thread`).
*   **Robustness**: Graceful degradation if Ollama or Resemblyzer is missing.

Generate the complete project structure and the initial setup code.
