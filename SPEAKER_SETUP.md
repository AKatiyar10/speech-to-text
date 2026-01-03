# Speaker Diarization Setup Guide

## Overview

This guide explains how to set up the speaker diarization feature using Resemblyzer (100% local, no external APIs).

## Prerequisites

### Option 1: Windows (Recommended)
Install Microsoft Visual C++ Build Tools:
```bash
# Download and install from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use Chocolatey:
choco install visualcpp-build-tools
```

### Option 2: Linux
```bash
sudo apt-get install build-essential python3-dev
```

### Option 3: macOS
```bash
xcode-select --install
```

## Installation

### Step 1: Install Python Dependencies
```bash
pip install resemblyzer
```

### Step 2: Verify Installation
```python
from resemblyzer import VoiceEncoder, preprocess_wav
encoder = VoiceEncoder()
print("✓ Resemblyzer installed successfully")
```

### Step 3: Start the Application
```bash
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py
```

## Troubleshooting

### Problem: "Failed building wheel for webrtcvad"
**Solution:** Install C++ build tools (see Prerequisites above)

### Problem: Resemblyzer import error
**Solution:** The app will still work! Speaker diarization will be disabled but transcription continues.

Check logs for:
```
✓ Resemblyzer VoiceEncoder loaded (local, offline)
```

If you see:
```
✗ Resemblyzer not installed: ...
```

Then speaker features are disabled but core app still works.

## Workflow

### Initial Setup (First Time User)
1. Start recording
2. System detects speech → assigns "UNKNOWN_01" (blue)
3. Transcription shows with blue badge: "UNKNOWN_01 (0%)"
4. **Click the badge** → opens relabel modal
5. Enter "ALPHA" and select green color
6. Click Save
7. Future recordings → recognized as ALPHA (75%+ confidence)

### Ongoing Use
1. New person speaks → "UNKNOWN_02" (blue)
2. Click badge → "Mark as John"
3. Future matches → shows "John"

## API Endpoints

### Speaker Management
- `POST /api/speakers/enroll` - Direct enrollment with voice sample
- `POST /api/speakers/relabel` - Rename speaker (UNKNOWN → ALPHA/Custom)
- `GET /api/speakers/list` - List all enrolled speakers
- `DELETE /api/speakers/{name}` - Delete speaker

### Conversation Management
- `GET /api/conversations/recent?limit=10` - Get recent entries
- `GET /api/conversations/context?entries=12` - Get LLM context string
- `GET /api/conversations/speaker/{name}?limit=50` - Get speaker's conversations

## File Structure

```
speech-to-text-app/
├── sessions/
│   ├── speaker_labels.json      # Speaker metadata & colors
│   ├── conversations.md        # Conversation history (LLM context)
│   └── session_history.json   # Existing: Session data
├── speakers/
│   └── embeddings/           # Voice embedding files (.npy)
│       ├── ALPHA.npy
│       ├── UNKNOWN_01.npy
│       └── John.npy
└── frontend/
    └── src/
        ├── App.js               # Updated with speaker UI
        ├── components/
        │   ├── RelabelModal.js     # Speaker relabeling modal
        │   └── ConversationHistory.js  # History viewer
        └── styles/
            ├── App.css
            ├── RelabelModal.css
            └── ConversationHistory.css
```

## Configuration

### Confidence Threshold
Default: 75% (0.75)
- Higher = fewer false matches, more UNKNOWNs
- Lower = more matches, more false positives

### Context Window
Default: 12 entries
- Number of past conversations sent to LLM for context
- Recommended: 10-15 entries

## Color Coding

| Speaker Type | Default Color | Hex |
|-------------|---------------|------|
| ALPHA (you) | Green | `#22c55e` |
| UNKNOWN_01, UNKNOWN_02, etc. | Blue | `#3b82f6` |
| Custom relabeled | User-specified | Any hex |

## How It Works

1. **Audio Recording** → User speaks into microphone
2. **VAD Detection** → Voice activity detected every 2s
3. **Silence Detection** → 2 consecutive silent chunks = end of speech
4. **Transcription** → Whisper converts audio to text
5. **Speaker Embedding** → Resemblyzer extracts 256-dim voice vector
6. **Speaker Matching** → Cosine similarity with enrolled speakers:
   - If similarity ≥ 75% → assign enrolled name
   - If similarity < 75% → create "UNKNOWN_XX"
7. **Conversation Storage** → Save to `conversations.md`
8. **WebSocket Response** → Send transcription + speaker + color to UI

## Next Steps

1. Install C++ build tools (see Prerequisites)
2. Install resemblyzer: `pip install resemblyzer`
3. Start the application
4. Record first speech → relabel as "ALPHA"
5. Start using full speaker diarization!
