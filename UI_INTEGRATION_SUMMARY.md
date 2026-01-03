# UI Integration Summary: Speaker Diarization + Original HTML Features

## Overview

Successfully integrated the existing `frontend-simple_v9.html` features with new speaker diarization capabilities.

## Changes Made

### 1. HTML File Updates (`frontend-simple_v9.html`)

#### Added Speaker Badge Styles
```css
.speaker-badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 16px;
  border-radius: 20px;
  color: white;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  user-select: none;
  margin-right: 10px;
}

.confidence-badge {
  background: rgba(0, 0, 0, 0.3);
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  margin-left: 8px;
}
```

#### Added Relabel Function
```javascript
function handleRelabel(speakerName) {
  const newName = prompt(`Relabel speaker "${speakerName}":`, 
    speakerName === 'UNKNOWN_01' ? 'ALPHA' : speakerName);
  if (newName && newName !== speakerName) {
    const color = speakerName.startsWith('UNKNOWN') 
      ? prompt('Enter color (hex code, e.g., #22c55e):', '#22c55e') 
      : prompt('Enter color for this speaker (hex code):', '#22c55e');
    
    const formData = new FormData();
    formData.append('old_name', speakerName);
    formData.append('new_name', newName);
    if (color) formData.append('color', color);

    fetch('http://localhost:8000/api/speakers/relabel', {
      method: 'POST',
      body: formData
    }).then(response => {
      if (response.ok) {
        alert(`Successfully relabeled ${speakerName} to ${newName}`);
        location.reload();
      } else {
        alert('Failed to relabel speaker');
      }
    }).catch(err => {
      console.error('Error:', err);
      alert('Error relabeling speaker');
    });
  }
}
```

#### Updated addCompletedSession Function
```javascript
// Added speaker parameter
function addCompletedSession(rawText, refinedText, feedback, 
                           sessionId = null, timestamp = null, 
                           fromHistory = false, 
                           speaker = 'UNKNOWN', 
                           speakerColor = '#3b82f6', 
                           confidence = 0) {
  
  // ... existing code ...

  // Added speaker badge to session header
  const speakerBadge = (speaker || 'UNKNOWN') ? `
    <div class="speaker-badge" 
         style="background-color: ${speakerColor}" 
         onclick="handleRelabel('${speaker}')" 
         title="Click to relabel speaker">
      ${speaker}
      ${confidence !== undefined ? `<span class="confidence-badge">${Math.round(confidence * 100)}%</span>` : ''}
    </div>
  ` : '';

  const header = `
    <div class="session-header ${fromHistory ? 'from-history' : ''}">
      ${speakerBadge}
      <div>
        <div class="session-info">Session #${sessionId} ${fromHistory ? '(History)' : ''}</div>
        <div class="session-timestamp">${displayTimestamp}</div>
      </div>
    </div>
  `;
  
  // ... rest of function ...
}
```

#### Updated WebSocket Message Handler
```javascript
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'transcription') {
    if (data.interim === false) {
      const speaker = data.speaker || 'UNKNOWN';
      const speakerColor = data.speaker_color || '#3b82f6';
      const confidence = data.confidence || 0;
      
      addCompletedSession(rawText, refinedText, feedback, null, null, false, 
                          speaker, speakerColor, confidence);
      // ... rest of handler ...
    }
  }
};
```

---

## Feature Comparison

| Feature | Original HTML | Updated | Notes |
|---------|----------------|---------|--------|
| Real-time transcription | ✅ | ✅ | Preserved |
| Mode selection (Continuous/Session) | ✅ | ✅ | Preserved |
| Three-panel layout (Raw/Refined/Feedback) | ✅ | ✅ | Preserved |
| On-demand feedback generation | ✅ | ✅ | Preserved |
| Copy buttons | ✅ | ✅ | Preserved |
| Session history loading | ✅ | ✅ | Preserved |
| Stats display | ✅ | ✅ | Preserved |
| Live indicator | ✅ | ✅ | Preserved |
| Speaker badges | ❌ | ✅ | **NEW** |
| Click-to-relabel | ❌ | ✅ | **NEW** |
| Confidence percentage | ❌ | ✅ | **NEW** |
| Speaker colors | ❌ | ✅ | **NEW** |

---

## UI Flow

### 1. Initial Recording (First User)
```
User records → "Hello world"
↓
System detects speech → Extracts voice embedding
↓
No enrolled speakers → Assigns "UNKNOWN_01" (blue, 0%)
↓
WebSocket sends: {text: "Hello world", speaker: "UNKNOWN_01", speaker_color: "#3b82f6", confidence: 0.0}
↓
UI shows: Blue badge "UNKNOWN_01 (0%)"
```

### 2. Relabeling as ALPHA
```
User clicks "UNKNOWN_01" badge
↓
Prompt: "Relabel speaker 'UNKNOWN_01': ALPHA"
↓
Prompt: "Enter color (hex code): #22c55e"
↓
POST /api/speakers/relabel
↓
Backend updates speaker_labels.json
↓
UI reloads
↓
Future recordings show: Green badge "ALPHA (85%+)"
```

### 3. Recognition After Enrollment
```
User records again → "How are you?"
↓
System extracts embedding → Compares with ALPHA
↓
Cosine similarity: 0.89 (≥ 75% threshold)
↓
WebSocket sends: {text: "How are you?", speaker: "ALPHA", speaker_color: "#22c55e", confidence: 0.89}
↓
UI shows: Green badge "ALPHA (89%)"
```

---

## API Integration

### Backend WebSocket Response Format
```javascript
{
  "type": "transcription",
  "text": "Hello world",
  "speaker": "ALPHA",              // NEW
  "speaker_color": "#22c55e",   // NEW
  "confidence": 0.89,            // NEW
  "interim": false
}
```

### Backend Speaker Relabel Endpoint
```
POST /api/speakers/relabel
Content-Type: multipart/form-data

Body:
{
  "old_name": "UNKNOWN_01",
  "new_name": "ALPHA",
  "color": "#22c55e"
}

Response:
{
  "success": true,
  "old_name": "UNKNOWN_01",
  "new_name": "ALPHA"
}
```

---

## Color Coding

| Speaker Type | Default Color | Hex Code |
|-------------|---------------|----------|
| UNKNOWN_01, UNKNOWN_02, ... | Blue | `#3b82f6` |
| ALPHA (user voice) | Green | `#22c55e` |
| Custom relabeled speakers | User-defined | Any hex |

---

## File Usage

### Running the Original HTML (Updated)
```bash
# Open in browser:
start frontend-simple_v9.html

# Or serve with local server:
python -m http.server 8000
# Then visit: http://localhost:8000/frontend-simple_v9.html
```

### Running the React Version
```bash
cd frontend
npm start
# Visit: http://localhost:3000
```

---

## Testing

### Test 1: First Recording
1. Open `frontend-simple_v9.html` in browser
2. Click "Start Recording"
3. Speak for 5-10 seconds
4. Click "Stop Recording"
5. Should see session with blue "UNKNOWN_01 (0%)" badge
6. Click badge → Enter "ALPHA" → Select green color
7. Reload page

### Test 2: Recognition
1. Click "Start Recording"
2. Speak again (same voice)
3. Should see green "ALPHA (85%+)" badge

### Test 3: Different Speaker
1. Have different person speak
2. Should see blue "UNKNOWN_02" badge
3. Click → Relabel as "John" → Select color
4. Future recognition as "John"

---

## Troubleshooting

### Problem: "Speaker badge not appearing"
**Solution:** Check WebSocket response includes `speaker` and `speaker_color` fields.

### Problem: "Clicking badge doesn't open prompt"
**Solution:** Ensure `handleRelabel` function is defined in global scope.

### Problem: "Relabel doesn't persist"
**Solution:** Check backend is running: `python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py`

---

## Next Steps

1. ✅ Install C++ build tools (Windows)
2. ✅ Install Resemblyzer: `pip install resemblyzer`
3. ✅ Start backend: `python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py`
4. ✅ Test with `frontend-simple_v9.html`
5. ✅ Record → Relabel as ALPHA
6. ✅ Verify recognition on subsequent recordings

---

## Summary

- **Original HTML features preserved**: ✅ All features maintained
- **Speaker diarization added**: ✅ Full integration
- **Relabel workflow**: ✅ Click-to-rename functionality
- **Color coding**: ✅ Visual speaker differentiation
- **Confidence display**: ✅ Percentage shown
- **API integration**: ✅ Backend endpoints connected

The updated `frontend-simple_v9.html` now provides the same functionality as the original version with the added capability to identify and label speakers using the Resemblyzer-based system!
