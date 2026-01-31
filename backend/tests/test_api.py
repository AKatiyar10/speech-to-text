import logging
from unittest.mock import MagicMock

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    # Allow loose match or update mock return in main
    data = response.json()
    assert data["status"] == "healthy"
    assert "connections" in data

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()

def test_get_recent_sessions_empty(client):
    # Setup mock
    client.app_manager.history_manager.get_recent_sessions.return_value = []
    
    response = client.get("/api/sessions/recent")
    assert response.status_code == 200
    assert response.json()["sessions"] == []

def test_get_recent_sessions_with_data(client):
    # Setup mock
    mock_sessions = [
        {"id": 1, "created_at": "2023-01-01T00:00:00", "duration": 120},
        {"id": 2, "created_at": "2023-01-02T00:00:00", "duration": 60}
    ]
    client.app_manager.history_manager.get_recent_sessions.return_value = mock_sessions
    
    response = client.get("/api/sessions/recent?limit=10")
    assert response.status_code == 200
    assert len(response.json()["sessions"]) == 2
    assert response.json()["sessions"][0]["id"] == 1

def test_generate_feedback(client):
    # Setup mock session existence check
    client.app_manager.history_manager.get_session_by_id.return_value = {
        "id": 123, 
        "speaking_feedback": None,
        "raw_text": "Hello"
    }
    
    # Setup mock update return
    client.app_manager.history_manager.update_session_feedback.return_value = {
        "feedback_generated_at": "2023-01-01T10:00:00"
    }
    
    # Mock refinement engine response
    from unittest.mock import AsyncMock
    # refinement engine is a property of manager
    client.app_manager.refinement.generate_feedback = AsyncMock(return_value="Great pronunciation!")
    
    # We also need to ensure refinement.enabled is True
    client.app_manager.refinement.enabled = True
    
    response = client.post("/api/sessions/123/generate-feedback")
    assert response.status_code == 200
    assert response.json()["speaking_feedback"] == "Great pronunciation!"
    assert response.json()["status"] == "success"
