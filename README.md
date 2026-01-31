# Speech-to-Text App with Speaker Diarization

A local, privacy-first speech-to-text application featuring real-time transcription, speaker identification (diarization), and AI-powered grammar refinement.

## 🚀 Key Features

- **Real-time Transcription**: Uses OpenAI Whisper for accurate, on-the-fly speech recognition.
- **Speaker Diarization**: Identifies speakers locally using Resemblyzer. Supports speaker enrollment and relabeling (e.g., "UNKNOWN_01" -> "Alice").
- **AI Refinement**: Corrects grammar and improves readability using a local LLM (Ollama with Gemma).
- **Context Awareness**: Uses recent conversation history to improve transcription context.
- **Privacy First**: All processing (transcription, embedding, refinement) happens locally on your machine.
- **Modern UI**: React-based frontend for managing recordings, speakers, and history.

## 🛠️ Prerequisites

- **Python 3.8+**
- **Node.js & npm** (for the React frontend)
- **Ollama** (installed and running with `gemma` model)
- **C++ Build Tools** (Required for `resemblyzer` on Windows)

## 📦 Installation

### 1. Backend Setup

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd speech-to-text-app

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

## 🏃 Usage

### 1. Start the Backend Server

```bash
# From the project root
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py
```
The server will start at `http://localhost:8000`.

### 2. Start the Frontend

```bash
# From the frontend directory
npm start
```
The application will open at `http://localhost:3000`.

## 🧪 Testing

Run the backend test suite:
```bash
pytest test_main_simple_vad_gemma.py
```
