# AGENTS.md

This file contains guidelines for agentic coding agents working in this repository.

## Project Overview

Speech-to-Text application with real-time WebSocket transcription, text refinement, on-demand speaking feedback, and speaker diarization. Backend uses FastAPI with Whisper (STT), Resemblyzer (speaker embeddings), and Ollama (LLM), frontend uses React.

### Key Features

- **Speaker Diarization:** 100% local voice recognition using Resemblyzer
  - ALPHA/UNKNOWN labeling with color coding
  - Click-to-relabel workflow
  - 75% confidence threshold
  - Stores voice embeddings locally (.npy files)

- **Context-Aware Refinement:** Past 10-12 conversations sent to LLM for better refinement

- **Conversation History:** Stored in `sessions/conversations.md` markdown file

### Speaker Diarization Setup

See `SPEAKER_SETUP.md` for detailed setup instructions including:
- Installing Resemblyzer (requires C++ build tools on Windows)
- Voice enrollment workflow
- API endpoints for speaker management

**Quick Start:**
1. Install C++ build tools (Windows) or build-essential (Linux)
2. `pip install resemblyzer`
3. Start app and record → relabel as "ALPHA"
4. Future recordings automatically recognized!

**UI Integration Summary:**
- `frontend-simple_v9.html` - Original HTML UI, **UPDATED** with speaker diarization:
  - Speaker badges (ALPHA/UNKNOWN with colors)
  - Click-to-relabel workflow
  - Confidence percentage display
  - Works directly in browser (no npm required)
- `frontend/src/App.js` - React version with full speaker diarization features
  - Both integrate with the same backend API

See `UI_INTEGRATION_SUMMARY.md` for detailed UI changes.
1. Install C++ build tools (Windows) or build-essential (Linux)
2. `pip install resemblyzer`
3. Start app and record → relabel as "ALPHA"
4. Future recordings automatically recognized!

---

## Commands

### UI Options

You can use either:
1. **Original HTML** (`frontend-simple_v9.html`) - Updated with speaker diarization
   - Open `frontend-simple_v9.html` directly in browser
   - Full speaker relabeling workflow
   - Click-to-rename speakers
   - Shows confidence percentage
   - Works directly in browser (no npm required)
2. **React App** (`frontend/src/App.js`) - Full-featured React version
   - Requires `npm start` from `frontend/` directory
   - Same speaker diarization features

**Speaker Diarization Setup:** See `SPEAKER_SETUP.md` for detailed setup instructions including:
- Installing Resemblyzer (requires C++ build tools on Windows)
- Voice enrollment workflow
- API endpoints for speaker management

**Quick Start:**
1. Install C++ build tools (Windows) or build-essential (Linux)
2. `pip install resemblyzer`
3. Start app and record → relabel as "ALPHA"
4. Future recordings automatically recognized!

**UI Integration Summary:**
- `frontend-simple_v9.html` has been updated with speaker badges and relabeling
- React version at `frontend/src/App.js` includes full speaker features
- Both versions integrate with the same backend API

### Backend (Python)

**Run the server:**
```bash
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py
```

**Run all tests:**
```bash
pytest test_main_simple_vad_gemma.py -v
```

**Run a single test:**
```bash
# Run specific test class
pytest test_main_simple_vad_gemma.py::TestSessionHistoryManager -v

# Run specific test method
pytest test_main_simple_vad_gemma.py::TestSessionHistoryManager::test_add_session -v
```

**Run tests with coverage:**
```bash
pytest test_main_simple_vad_gemma.py --cov=main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 --cov-report=html
```

**Run tests with markers:**
```bash
# Run only unit tests
pytest test_main_simple_vad_gemma.py -m unit -v

# Run only integration tests
pytest test_main_simple_vad_gemma.py -m integration -v

# Skip slow tests
pytest test_main_simple_vad_gemma.py -m "not slow" -v
```

**Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Frontend (React)

**Start development server:**
```bash
cd frontend && npm start
```

**Build for production:**
```bash
cd frontend && npm run build
```

**Run tests:**
```bash
cd frontend && npm test
```

---

## Code Style Guidelines

### Python (Backend)

#### Imports

- Group imports into three sections: standard library, third-party, local
- Separate each section with a blank line
- Use `from x import y` for specific imports, `import x` for modules

```python
# Standard library
import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock

# Third-party
import numpy as np
import ollama
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel

# Local imports
from main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 import SessionHistoryManager
```

#### Type Hints

- Use type hints for all function parameters and return values
- Use `Optional[T]` for nullable types, `Dict[str, Any]` for flexible dicts

```python
def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
    """Process audio data and return result dictionary."""
    pass

async def refine_text(self, raw_text: str) -> str:
    """Refine text using LLM."""
    pass

def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
    """Get session by ID, returns None if not found."""
    pass
```

#### Naming Conventions

- **Classes**: `PascalCase` - `SessionHistoryManager`, `SimpleVAD`
- **Functions/Methods**: `snake_case` - `add_session`, `refine_text`, `get_stats`
- **Variables**: `snake_case` - `client_id`, `raw_text`, `energy_threshold`
- **Constants**: `UPPER_SNAKE_CASE` - `DEFAULT_THRESHOLD`, `MAX_RETRY_COUNT`
- **Private methods**: leading underscore - `_save_history`, `_rebuild_cache`
- **Enum values**: `UPPER_SNAKE_CASE` - `OutputMode.RAW_ONLY`

#### Class Structure

- Start with docstring describing class purpose
- Use `__init__` for initialization with type hints
- Group methods logically: public, then private
- Use thread-safety primitives (`threading.Lock`, `asyncio.Lock`) for shared state

```python
class SessionHistoryManager:
    """Manages session history in JSON file with thread-safe operations"""
    
    def __init__(self, history_file: str = "session_history.json"):
        self.history_file = Path(history_file)
        self.sessions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load_history()
    
    def add_session(self, raw_text: str, refined_text: str, client_id: str) -> Dict[str, Any]:
        """Add a new session to history."""
        with self._lock:
            # ... implementation
            pass
    
    def _load_history(self) -> None:
        """Load existing history from file."""
        pass
```

#### Error Handling

- Use try/except for I/O operations and external API calls
- Log errors using `logger.error()`
- Return fallback values for graceful degradation
- Raise `HTTPException` for API errors with appropriate status codes

```python
try:
    with open(self.history_file, 'r', encoding='utf-8') as f:
        self.sessions = json.load(f)
except Exception as e:
    logger.error(f"Error loading history: {e}")
    self.sessions = []

# API endpoints
if not session:
    raise HTTPException(status_code=404, detail=f"Session #{session_id} not found")
```

#### Async/Await Patterns

- Use `asyncio.to_thread()` for blocking operations (file I/O, Whisper transcription)
- Use `asyncio.Lock()` for thread-safe async operations
- Always return `Dict[str, Any]` or typed response objects from async endpoints

```python
async def process_audio(self) -> Dict[str, Any]:
    """Process audio with blocking operation."""
    transcription = await asyncio.to_thread(self._transcribe, wav_path)
    return {"text": transcription}

async with self._lock:
    result = await self.client.generate(...)
```

#### Logging

- Use `logger.info()` for normal operations
- Use `logger.warning()` for recoverable issues
- Use `logger.error()` for errors
- Include relevant context in log messages

```python
logger.info(f"Session #{session_id} saved (without feedback)")
logger.warning(f"Failed to send message to {client_id}: {e}")
logger.error(f"VAD error: {e}")
```

#### Docstrings

- Use triple-quoted docstrings for all classes and public methods
- Keep docstrings concise but descriptive

```python
def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent N sessions (newest first)."""
    pass
```

### React (Frontend)

#### Component Structure

- Use functional components with hooks
- Follow existing patterns in `src/App.js`

#### State Management

- Use `useState` for local component state
- Keep components focused and modular

---

## Testing Guidelines

### Test Structure

- Use `pytest` framework with `pytest-asyncio` for async tests
- Mark async tests with `@pytest.mark.asyncio`
- Use fixtures for setup (`@pytest.fixture`)

```python
@pytest.fixture
def session_manager(temp_history_file):
    """Create a SessionHistoryManager with temporary file."""
    return SessionHistoryManager(history_file=temp_history_file)

@pytest.mark.asyncio
async def test_add_session(self, session_manager):
    """Test adding a new session."""
    session = session_manager.add_session("raw", "refined", "client1")
    assert session['session_id'] == 1
```

### Test Naming

- Test classes: `Test<ClassName>` - `TestSessionHistoryManager`
- Test methods: `test_<feature>_<scenario>` - `test_add_session`, `test_add_multiple_sessions`

### Mocking

- Mock external dependencies (Whisper, Ollama) for fast, offline tests
- Use `unittest.mock.Mock` and `unittest.mock.patch`

```python
from unittest.mock import Mock, patch

@pytest.fixture
def mock_whisper():
    """Mock Whisper model for testing."""
    mock = Mock()
    mock.transcribe = Mock(return_value={"text": "hello world"})
    return mock
```

---

## File Organization

- `main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py` - Main FastAPI application
- `test_main_simple_vad_gemma.py` - Test suite
- `requirements.txt` - Production dependencies
- `requirements-test.txt` - Test dependencies
- `pytest.ini` - Pytest configuration
- `frontend/` - React application
  - `src/App.js` - Main React component
  - `src/index.js` - Entry point

---

## Important Notes

1. **Thread Safety**: Always use locks when accessing shared state (sessions, files, external clients)
2. **Temp Files**: Always clean up temp files in `finally` blocks
3. **Graceful Degradation**: External dependencies (Ollama) should be optional - handle unavailability gracefully
4. **Encoding**: Always use `encoding='utf-8'` when reading/writing files
5. **Session IDs**: Auto-increment based on list length in `SessionHistoryManager`
6. **WebSocket**: Handle `WebSocketDisconnect` exception, cleanup in `finally` block
7. **Logging**: Use logger for all significant operations

---

## Common Patterns

### File I/O with cleanup

```python
wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
os.close(wav_fd)  # Close file descriptor immediately

try:
    # Use wav_path
    pass
finally:
    if wav_path:
        try:
            os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")
```

### Thread-safe operations

```python
with self._lock:
    # Critical section
    result = self._do_operation()
```

### Async lock operations

```python
lock = self._get_lock()
async with lock:
    result = await self.client.generate(...)
```
