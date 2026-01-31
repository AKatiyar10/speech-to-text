# Project Features

## 🎙️ Core Speech Processing

### Real-time Transcription
- **Engine**: OpenAI Whisper (Local).
- **Functionality**: Continuously listens to microphone input and provides immediate text output.
- **VAD**: Silero VAD ensures only active speech is processed, reducing hallucination and processing load.

### Speaker Diarization (Identification)
- **Engine**: Resemblyzer (100% Local).
- **Functionality**:
    - Extracts 256-dimensional voice vectors from audio chunks.
    - Identifies speakers with a configurable 75% confidence threshold.
    - **Enrollment**: Users can "enroll" their voice to be automatically recognized.
    - **Relabeling**: Unknown speakers appear as "UNKNOWN_XX". Users can click to rename them (e.g., to "Bob") and assign custom colors.
    - **Persistence**: Speaker profiles and embeddings are saved locally in `sessions/speaker_labels.json` and `speakers/embeddings/`.

## 🧠 AI Refinement & Context

### Grammar Correction
- **Engine**: Ollama (Gemma Model).
- **Functionality**: Raw transcripts are sent to the local LLM to fix grammar, punctuation, and sentence structure without altering the meaning.

### Context Awareness
- **Memory**: The system maintains a markdown-based conversation history (`sessions/conversations.md`).
- **Functionality**: The LLM uses the past 10-12 conversation turns to understand context, improving the accuracy of corrections (e.g., knowing who is speaking to whom).

## 💻 User Interface

### React Frontend
- **Live Feed**: Displays transcripts in real-time as chat bubbles.
- **Speaker Badges**: Shows speaker name and confidence score.
- **Interactive Management**:
    - **Relabel Modal**: Click any speaker badge to rename and color-code.
    - **History View**: Browse past conversations.
    - **Recording Control**: One-click start/stop.
- **WebSockets**: Uses WebSocket connection for low-latency updates.

## 🔒 Privacy & Data
- **Local Storage**: Voice embeddings (`.npy`), session history (`.json`), and transcripts (`.md`) are stored locally.
- **Offline Capable**: Once models are downloaded, no internet connection is required for operation.
