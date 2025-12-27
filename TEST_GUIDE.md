# Test Suite Guide

## Overview
Comprehensive test suite for `main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py` covering:
- Session history management (CRUD, persistence, caching)
- Voice Activity Detection (VAD)
- Text refinement & feedback generation
- Audio processing & transcription
- API endpoints (REST + WebSocket)
- Error handling & edge cases

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests:
```bash
pytest test_main_simple_vad_gemma.py -v
```

### Run specific test class:
```bash
pytest test_main_simple_vad_gemma.py::TestSessionHistoryManager -v
```

### Run specific test:
```bash
pytest test_main_simple_vad_gemma.py::TestSessionHistoryManager::test_add_session -v
```

### Run with coverage report:
```bash
pytest test_main_simple_vad_gemma.py --cov=main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 --cov-report=html
```

### Run only fast tests (exclude slow):
```bash
pytest test_main_simple_vad_gemma.py -m "not slow" -v
```

### Run with detailed output:
```bash
pytest test_main_simple_vad_gemma.py -vv --tb=long
```

## Test Structure

### SessionHistoryManager Tests (13 tests)
- Session CRUD operations
- Cache O(1) lookups
- Persistence to disk
- Statistics calculation
- Character count tracking

### SimpleVAD Tests (3 tests)
- Initialization
- Silence detection
- Speech/noise detection

### EnhancedRefinementEngine Tests (4 tests)
- Graceful degradation when Ollama unavailable
- Text refinement
- Feedback generation
- Short text handling

### ContinuousAudioProcessor Tests (6 tests)
- Audio buffering
- Processing triggers
- Empty state handling
- Force finalization

### API Endpoint Tests (7 tests)
- Health & root endpoints
- Session retrieval endpoints
- Feedback generation endpoint
- WebSocket validation

### ConnectionManager Tests (3 tests)
- Model loading
- Connection lifecycle
- Disconnect & cleanup

### Data Model Tests (2 tests)
- Pydantic model validation

### Edge Case Tests (4 tests)
- Unicode handling
- Very long text
- Concurrent operations
- Cache consistency

## Test Coverage

Target coverage:
- `SessionHistoryManager`: 95%+ (core business logic)
- `SimpleVAD`: 85%+ (audio processing)
- `EnhancedRefinementEngine`: 80%+ (external dependency)
- `ContinuousAudioProcessor`: 85%+ (complex async logic)
- API endpoints: 90%+ (integration points)
- `ConnectionManager`: 85%+ (lifecycle management)

## Key Fixtures

- `temp_history_file`: Temporary JSON file for session history
- `session_manager`: SessionHistoryManager instance
- `test_client`: FastAPI TestClient
- `mock_whisper`: Mocked Whisper model
- `mock_vad`: Mocked VAD instance
- `mock_refinement`: Mocked refinement engine

## Mocking Strategy

- **Whisper model**: Mocked to return fixed transcriptions (fast, no GPU needed)
- **Ollama client**: Mocked to avoid external dependencies
- **File I/O**: Using tempfile for isolation
- **WebSocket**: Using TestClient's websocket_connect

## Important Notes

1. **Async tests**: Uses `pytest-asyncio` with `asyncio_mode = auto`
2. **Thread safety**: Tests concurrent operations via ThreadPoolExecutor
3. **File cleanup**: All temp files auto-deleted via `finally` blocks
4. **Isolation**: Each test uses its own temp history file
5. **No external deps**: Tests mock Ollama and Whisper to run offline

## Running in CI/CD

Example GitHub Actions:
```yaml
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest test_main_simple_vad_gemma.py -v --cov --cov-report=xml
```

## Troubleshooting

**Issue**: `pytest: command not found`
- **Fix**: `pip install pytest pytest-asyncio`

**Issue**: `asyncio test marked but no fixture`
- **Fix**: Already configured in `pytest.ini` with `asyncio_mode = auto`

**Issue**: Import errors for main module
- **Ensure**: Both `main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py` and `test_main_simple_vad_gemma.py` are in same directory

**Issue**: Whisper model download delays
- **Workaround**: Use mocks (already done in tests) or pre-download: `whisper.load_model("base")`

## Next Steps

1. Run full test suite: `pytest test_main_simple_vad_gemma.py -v`
2. Check coverage: `pytest --cov --cov-report=html`
3. Add custom tests for your specific scenarios
4. Integrate into CI/CD pipeline
