
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Depending on how conftest is set up, we might need to rely on the existing client fixture
# But to be safe and independent, I'll mock the dependencies here or reuse if possible.
# I'll write a test that fits the existing pattern seen in test_api.py

def test_generate_feedback_retry_on_error(client):
    """
    Verify that if a session has an error message as feedback, 
    the API allows generating new feedback (retry).
    """
    # 1. Setup session with existing "Error" feedback
    client.app_manager.history_manager.get_session_by_id.return_value = {
        "session_id": 999,
        "speaking_feedback": "Error generating feedback: timeout",
        "raw_text": "Test speech"
    }
    
    # 2. Setup mock refinement engine to succeed this time
    client.app_manager.refinement.generate_feedback = AsyncMock(return_value="Better feedback now")
    client.app_manager.refinement.enabled = True
    
    # 3. Setup mock update
    client.app_manager.history_manager.update_session_feedback.return_value = {
        "feedback_generated_at": "2023-01-01T12:00:00"
    }
    
    # 4. Call the API
    response = client.post("/api/sessions/999/generate-feedback")
    
    # 5. Assert success (200 OK) instead of "already_exists"
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["speaking_feedback"] == "Better feedback now"

def test_generate_feedback_failure_handling(client):
    """
    Verify that if generation fails (returns None), 
    the API returns 500 and does NOT save the error to history.
    """
    # 1. Setup session with no feedback
    client.app_manager.history_manager.get_session_by_id.return_value = {
        "session_id": 888,
        "speaking_feedback": None,
        "raw_text": "Test speech"
    }
    
    # 2. Setup mock refinement engine to FAIL (return None)
    # The refinement engine now returns None on exception
    client.app_manager.refinement.generate_feedback = AsyncMock(return_value=None)
    client.app_manager.refinement.enabled = True
    
    # 3. Call the API
    response = client.post("/api/sessions/888/generate-feedback")
    
    # 4. Assert failure (500 Internal Server Error)
    # Note: validation check is done in main.py
    assert response.status_code == 500
    assert "Failed to generate feedback" in response.json()["detail"]
    
    # 5. Verify update_session_feedback was NOT called
    client.app_manager.history_manager.update_session_feedback.assert_not_called()
