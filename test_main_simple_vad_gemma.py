"""
Test suite for main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py

Covers:
- SessionHistoryManager (CRUD, caching, stats)
- SimpleVAD (speech detection)
- EnhancedRefinementEngine (text refinement, feedback)
- ContinuousAudioProcessor (buffering, transcription, finalization)
- API endpoints (GET, POST, WebSocket)
- Error cases and edge cases
"""

import pytest
import asyncio
import tempfile
import json
import numpy as np
import wave
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient
from main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 import (
    app,
    SessionHistoryManager,
    SimpleVAD,
    EnhancedRefinementEngine,
    ContinuousAudioProcessor,
    ConnectionManager,
    OutputMode,
    GenerateFeedbackResponse,
)


# ===== FIXTURES =====

@pytest.fixture
def temp_history_file():
    """Create a temporary history file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def session_manager(temp_history_file):
    """Create a SessionHistoryManager with temporary file."""
    return SessionHistoryManager(history_file=temp_history_file)


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_whisper():
    """Mock Whisper model for testing."""
    mock = Mock()
    mock.transcribe = Mock(return_value={"text": "hello world"})
    return mock


@pytest.fixture
def mock_vad():
    """Mock VAD for testing."""
    vad = SimpleVAD(energy_threshold=0.01)
    vad.enabled = True
    return vad


@pytest.fixture
def mock_refinement():
    """Mock refinement engine."""
    engine = Mock()
    engine.enabled = False
    engine.refine_text = AsyncMock(return_value="refined text")
    engine.generate_feedback = AsyncMock(return_value="Good job!")
    return engine


# ===== SESSION HISTORY MANAGER TESTS =====

class TestSessionHistoryManager:
    """Tests for SessionHistoryManager."""

    def test_init_creates_new_history_file(self, temp_history_file):
        """Test initialization with non-existent file."""
        Path(temp_history_file).unlink(missing_ok=True)
        manager = SessionHistoryManager(history_file=temp_history_file)
        
        assert manager.sessions == []
        assert Path(temp_history_file).exists()

    def test_add_session(self, session_manager):
        """Test adding a new session."""
        session = session_manager.add_session(
            raw_text="hello world",
            refined_text="hello world",
            client_id="client1"
        )
        
        assert session['session_id'] == 1
        assert session['raw_text'] == "hello world"
        assert session['refined_text'] == "hello world"
        assert session['client_id'] == "client1"
        assert session['speaking_feedback'] is None
        assert session['feedback_generated_at'] is None

    def test_add_multiple_sessions(self, session_manager):
        """Test adding multiple sessions increments session_id."""
        s1 = session_manager.add_session("raw1", "refined1", "client1")
        s2 = session_manager.add_session("raw2", "refined2", "client2")
        s3 = session_manager.add_session("raw3", "refined3", "client3")
        
        assert s1['session_id'] == 1
        assert s2['session_id'] == 2
        assert s3['session_id'] == 3
        assert len(session_manager.sessions) == 3

    def test_get_session_by_id(self, session_manager):
        """Test O(1) session lookup by ID."""
        session = session_manager.add_session("raw", "refined", "client1")
        
        retrieved = session_manager.get_session_by_id(session['session_id'])
        assert retrieved is not None
        assert retrieved['session_id'] == session['session_id']
        assert retrieved['raw_text'] == "raw"

    def test_get_session_by_id_not_found(self, session_manager):
        """Test retrieving non-existent session returns None."""
        result = session_manager.get_session_by_id(999)
        assert result is None

    def test_update_session_feedback(self, session_manager):
        """Test updating a session with feedback."""
        session = session_manager.add_session("raw", "refined", "client1")
        feedback = "Great speech!"
        
        updated = session_manager.update_session_feedback(session['session_id'], feedback)
        
        assert updated['speaking_feedback'] == feedback
        assert updated['feedback_generated_at'] is not None
        assert updated['character_counts']['feedback'] == len(feedback)

    def test_update_nonexistent_session_raises(self, session_manager):
        """Test updating non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            session_manager.update_session_feedback(999, "feedback")

    def test_get_all_sessions(self, session_manager):
        """Test retrieving all sessions."""
        session_manager.add_session("raw1", "refined1", "client1")
        session_manager.add_session("raw2", "refined2", "client2")
        
        all_sessions = session_manager.get_all_sessions()
        assert len(all_sessions) == 2

    def test_get_recent_sessions(self, session_manager):
        """Test retrieving recent N sessions (newest first)."""
        for i in range(5):
            session_manager.add_session(f"raw{i}", f"refined{i}", f"client{i}")
        
        recent = session_manager.get_recent_sessions(limit=3)
        assert len(recent) == 3
        assert recent[0]['session_id'] == 5  # Newest first
        assert recent[1]['session_id'] == 4
        assert recent[2]['session_id'] == 3

    def test_get_recent_sessions_empty(self, session_manager):
        """Test get_recent_sessions on empty history."""
        recent = session_manager.get_recent_sessions(limit=10)
        assert recent == []

    def test_get_stats_empty(self, session_manager):
        """Test statistics on empty history."""
        stats = session_manager.get_stats()
        
        assert stats['total_sessions'] == 0
        assert stats['total_words_spoken'] == 0
        assert stats['sessions_with_feedback'] == 0
        assert stats['average_session_length'] == 0

    def test_get_stats_with_sessions(self, session_manager):
        """Test statistics with sessions."""
        s1 = session_manager.add_session("one two three", "one two three", "client1")
        s2 = session_manager.add_session("alpha beta", "alpha beta", "client2")
        
        session_manager.update_session_feedback(s1['session_id'], "Good!")
        
        stats = session_manager.get_stats()
        
        assert stats['total_sessions'] == 2
        assert stats['total_words_spoken'] == 5  # 3 + 2 words
        assert stats['sessions_with_feedback'] == 1
        assert stats['average_session_length'] > 0
        assert stats['first_session'] is not None
        assert stats['last_session'] is not None

    def test_persistence_across_instances(self, temp_history_file):
        """Test that sessions persist to disk and load on next init."""
        manager1 = SessionHistoryManager(history_file=temp_history_file)
        manager1.add_session("raw1", "refined1", "client1")
        
        manager2 = SessionHistoryManager(history_file=temp_history_file)
        
        assert len(manager2.sessions) == 1
        assert manager2.sessions[0]['raw_text'] == "raw1"

    def test_character_counts(self, session_manager):
        """Test character count tracking."""
        raw = "hello world"
        refined = "hello world"
        session = session_manager.add_session(raw, refined, "client1")
        
        assert session['character_counts']['raw'] == len(raw)
        assert session['character_counts']['refined'] == len(refined)
        assert session['character_counts']['feedback'] == 0
        
        session_manager.update_session_feedback(session['session_id'], "Good!")
        updated = session_manager.get_session_by_id(session['session_id'])
        assert updated['character_counts']['feedback'] == 5


# ===== SIMPLE VAD TESTS =====

class TestSimpleVAD:
    """Tests for SimpleVAD."""

    def test_vad_initialization(self):
        """Test VAD initialization."""
        vad = SimpleVAD(energy_threshold=0.01)
        assert vad.energy_threshold == 0.01
        assert vad.enabled is True

    def test_vad_disabled(self):
        """Test VAD always returns True when disabled."""
        vad = SimpleVAD(energy_threshold=0.01)
        vad.enabled = False
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            result = vad.has_speech(f.name)
            assert result is True

    def test_vad_with_silence(self):
        """Test VAD detects silence (low energy audio)."""
        vad = SimpleVAD(energy_threshold=0.1)
        
        # Create a silent WAV file (zeros)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # Silent audio (all zeros)
                silent_audio = np.zeros(16000, dtype=np.int16)
                wf.writeframes(silent_audio.tobytes())
            
            result = vad.has_speech(temp_path)
            assert result is False  # Should detect silence
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_vad_with_noise(self):
        """Test VAD detects noise/speech (high energy audio)."""
        vad = SimpleVAD(energy_threshold=0.01)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # Noisy audio (random values)
                noisy_audio = np.random.randint(-20000, 20000, 16000, dtype=np.int16)
                wf.writeframes(noisy_audio.tobytes())
            
            result = vad.has_speech(temp_path)
            assert result is True  # Should detect speech/noise
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ===== ENHANCED REFINEMENT ENGINE TESTS =====

class TestEnhancedRefinementEngine:
    """Tests for EnhancedRefinementEngine."""

    @pytest.mark.asyncio
    async def test_refinement_disabled_on_init_error(self):
        """Test refinement gracefully disables if Ollama is unavailable."""
        with patch('ollama.Client', side_effect=Exception("Connection refused")):
            engine = EnhancedRefinementEngine(model_name="phi3:mini")
            assert engine.enabled is False

    @pytest.mark.asyncio
    async def test_refine_text_passthrough_when_disabled(self):
        """Test refine_text returns original when disabled."""
        engine = EnhancedRefinementEngine(model_name="phi3:mini")
        engine.enabled = False
        
        result = await engine.refine_text("hello world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_refine_text_too_short(self):
        """Test refine_text returns original for text < 5 chars."""
        engine = EnhancedRefinementEngine(model_name="phi3:mini")
        engine.enabled = True
        engine.client = Mock()
        
        result = await engine.refine_text("hi")
        assert result == "hi"  # Should not call ollama

    @pytest.mark.asyncio
    async def test_generate_feedback_passthrough_when_disabled(self):
        """Test generate_feedback returns default when disabled."""
        engine = EnhancedRefinementEngine(model_name="phi3:mini")
        engine.enabled = False
        
        result = await engine.generate_feedback("hello world")
        assert "Text too short" in result or "meaningful" in result

    @pytest.mark.asyncio
    async def test_generate_feedback_too_short(self):
        """Test generate_feedback returns default for text < 5 chars."""
        engine = EnhancedRefinementEngine(model_name="phi3:mini")
        engine.enabled = True
        engine.client = Mock()
        
        result = await engine.generate_feedback("hi")
        assert "Text too short" in result


# ===== CONTINUOUS AUDIO PROCESSOR TESTS =====

class TestContinuousAudioProcessor:
    """Tests for ContinuousAudioProcessor."""

    @pytest.fixture
    def processor(self, mock_whisper, mock_vad, mock_refinement, session_manager):
        """Create a processor for testing."""
        return ContinuousAudioProcessor(
            whisper_model=mock_whisper,
            vad=mock_vad,
            refinement_engine=mock_refinement,
            output_mode=OutputMode.ALL,
            history_manager=session_manager,
            client_id="test_client"
        )

    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.output_mode == OutputMode.ALL
        assert processor.client_id == "test_client"
        assert processor.audio_buffer == []
        assert processor.accumulated_text == []

    def test_add_audio_returns_false_for_short_duration(self, processor):
        """Test add_audio returns False for < 2s of audio."""
        # 0.1s of audio (16000 Hz * 2 bytes)
        short_audio = b'\x00' * int(16000 * 2 * 0.1)
        
        result = processor.add_audio(short_audio)
        assert result is False

    def test_add_audio_returns_true_for_long_duration(self, processor):
        """Test add_audio returns True for >= 2s of audio."""
        # 2.1s of audio
        long_audio = b'\x00' * int(16000 * 2 * 2.1)
        
        result = processor.add_audio(long_audio)
        assert result is True

    @pytest.mark.asyncio
    async def test_process_empty_buffer(self, processor):
        """Test process with empty buffer."""
        result = await processor.process()
        
        assert result.get('skipped') is True
        assert result.get('interim') is True

    @pytest.mark.asyncio
    async def test_process_short_audio(self, processor):
        """Test process with < 16000 bytes."""
        processor.audio_buffer = [b'\x00' * 1000]
        
        result = await processor.process()
        
        assert result.get('skipped') is True

    @pytest.mark.asyncio
    @patch('tempfile.mkstemp')
    @patch('wave.open')
    async def test_force_finalize_empty(self, mock_wave, mock_mkstemp, processor):
        """Test force_finalize with no accumulated text."""
        result = await processor.force_finalize()
        
        assert result.get('skipped') is True
        assert result.get('interim') is False

    @pytest.mark.asyncio
    @patch('tempfile.mkstemp')
    @patch('wave.open')
    async def test_force_finalize_with_text(self, mock_wave, mock_mkstemp, processor):
        """Test force_finalize saves accumulated text."""
        processor.accumulated_text = ["hello", "world"]
        
        result = await processor.force_finalize()
        
        assert result.get('skipped') is False
        assert result.get('interim') is False
        assert 'session_id' in result


# ===== API ENDPOINT TESTS =====

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, test_client):
        """Test GET / endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json()['status'] == 'running'
        assert 'session_stats' in response.json()

    def test_health_endpoint(self, test_client):
        """Test GET /health endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'connections' in data
        assert 'total_sessions' in data

    def test_get_all_sessions_empty(self, test_client):
        """Test GET /api/sessions with no sessions."""
        response = test_client.get("/api/sessions")
        
        assert response.status_code == 200
        assert response.json()['total'] == 0
        assert response.json()['sessions'] == []

    def test_get_recent_sessions_default_limit(self, test_client):
        """Test GET /api/sessions/recent with default limit."""
        response = test_client.get("/api/sessions/recent")
        
        assert response.status_code == 200
        assert 'limit' in response.json()
        assert 'sessions' in response.json()

    def test_get_recent_sessions_custom_limit(self, test_client):
        """Test GET /api/sessions/recent with custom limit."""
        response = test_client.get("/api/sessions/recent?limit=5")
        
        assert response.status_code == 200
        assert response.json()['limit'] == 5

    def test_get_session_not_found(self, test_client):
        """Test GET /api/sessions/{id} with non-existent ID."""
        response = test_client.get("/api/sessions/999")
        
        assert response.status_code == 404

    def test_get_stats_endpoint(self, test_client):
        """Test GET /api/stats endpoint."""
        response = test_client.get("/api/stats")
        
        assert response.status_code == 200
        stats = response.json()
        assert 'total_sessions' in stats
        assert 'total_words_spoken' in stats
        assert 'sessions_with_feedback' in stats

    @pytest.mark.asyncio
    async def test_generate_feedback_session_not_found(self, test_client):
        """Test POST /api/sessions/{id}/generate-feedback with non-existent ID."""
        response = test_client.post("/api/sessions/999/generate-feedback")
        
        assert response.status_code == 404

    def test_websocket_query_param_validation(self, test_client):
        """Test WebSocket endpoint validates mode query param."""
        # Valid modes: raw, refined, full, all
        with test_client.websocket_connect("/ws/audio/test_client?mode=all"):
            pass  # Should not raise


# ===== CONNECTION MANAGER TESTS =====

class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connection_manager_initialization(self):
        """Test ConnectionManager loads models on init."""
        with patch('whisper.load_model') as mock_load:
            mock_load.return_value = Mock()
            manager = ConnectionManager(model_name="phi3:mini")
            
            assert manager.whisper is not None
            assert manager.vad is not None
            assert manager.refinement is not None
            assert manager.history_manager is not None

    @pytest.mark.asyncio
    async def test_connect_creates_processor(self):
        """Test connect method creates a processor for the client."""
        with patch('whisper.load_model'):
            manager = ConnectionManager()
            
            mock_websocket = AsyncMock()
            await manager.connect(mock_websocket, "client1", OutputMode.ALL)
            
            assert "client1" in manager.connections
            assert "client1" in manager.processors
            mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_finalizes_and_cleans(self):
        """Test disconnect finalizes session and cleans up."""
        with patch('whisper.load_model'):
            manager = ConnectionManager()
            
            mock_websocket = AsyncMock()
            await manager.connect(mock_websocket, "client1", OutputMode.ALL)
            
            # Manually set accumulated text to trigger finalization
            manager.processors["client1"].accumulated_text = ["test"]
            
            await manager.disconnect("client1")
            
            assert "client1" not in manager.connections
            assert "client1" not in manager.processors


# ===== OUTPUT MODE TESTS =====

class TestOutputMode:
    """Tests for OutputMode enum."""

    def test_output_mode_values(self):
        """Test OutputMode enum has correct values."""
        assert OutputMode.RAW_ONLY.value == "raw"
        assert OutputMode.REFINED_ONLY.value == "refined"
        assert OutputMode.REFINED_WITH_FEEDBACK.value == "full"
        assert OutputMode.ALL.value == "all"

    def test_output_mode_from_string(self):
        """Test creating OutputMode from string."""
        mode = OutputMode("raw")
        assert mode == OutputMode.RAW_ONLY


# ===== PYDANTIC MODEL TESTS =====

class TestDataModels:
    """Tests for Pydantic models."""

    def test_generate_feedback_request_valid(self):
        """Test GenerateFeedbackRequest validation."""
        from main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 import GenerateFeedbackRequest
        
        req = GenerateFeedbackRequest(session_id=1)
        assert req.session_id == 1

    def test_generate_feedback_response_valid(self):
        """Test GenerateFeedbackResponse validation."""
        resp = GenerateFeedbackResponse(
            session_id=1,
            speaking_feedback="Great job!",
            timestamp="2025-12-25T10:00:00",
            status="generated"
        )
        
        assert resp.session_id == 1
        assert resp.speaking_feedback == "Great job!"


# ===== EDGE CASE & STRESS TESTS =====

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_session_manager_unicode_handling(self, session_manager):
        """Test SessionHistoryManager handles unicode correctly."""
        session = session_manager.add_session(
            raw_text="Hello 你好 مرحبا",
            refined_text="Hello 你好 مرحبا",
            client_id="client1"
        )
        
        assert session['raw_text'] == "Hello 你好 مرحبا"
        assert session['character_counts']['raw'] == len("Hello 你好 مرحبا")

    def test_session_manager_very_long_text(self, session_manager):
        """Test SessionHistoryManager handles very long text."""
        long_text = "word " * 10000
        session = session_manager.add_session(long_text, long_text, "client1")
        
        assert session['character_counts']['raw'] == len(long_text)

    def test_session_manager_concurrent_adds(self, session_manager):
        """Test SessionHistoryManager handles concurrent adds."""
        import concurrent.futures
        
        def add_session(i):
            return session_manager.add_session(f"raw{i}", f"refined{i}", f"client{i}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_session, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        assert len(session_manager.sessions) == 10
        assert all(r['session_id'] is not None for r in results)

    def test_session_manager_cache_consistency(self, session_manager):
        """Test that cache stays consistent with sessions list."""
        s1 = session_manager.add_session("raw1", "refined1", "client1")
        s2 = session_manager.add_session("raw2", "refined2", "client2")
        
        # Both methods should return same data
        from_cache = session_manager.get_session_by_id(s1['session_id'])
        from_list = [s for s in session_manager.sessions if s['session_id'] == s1['session_id']][0]
        
        assert from_cache == from_list


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
