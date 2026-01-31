import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Name of the main application module
APP_MODULE = "main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10"

@pytest.fixture(scope="session", autouse=True)
def mock_heavy_imports():
    """
    Mock heavy libraries globally before they are even imported by the app.
    This prevents `import whisper` or model loading from happening.
    """
    # Mock whisper
    sys.modules["whisper"] = MagicMock()
    
    # Mock simple_vad and other custom engines if they load models at import time
    # (checking outline: simple_vad seems to preserve class but we can mock it)
    # Actually, let's just mock the classes used in ConnectionManager
    pass

@pytest.fixture(name="mock_manager")
def mock_manager_fixture():
    """Mocks the ConnectionManager class."""
    with patch("connection_manager.ConnectionManager") as MockClass:
        mock_instance = MockClass.return_value
        
        # Setup default behaviors for mock
        mock_instance.get_recent_sessions.return_value = []
        mock_instance.get_stats.return_value = {"total_sessions": 0}
        
        yield mock_instance

@pytest.fixture(name="client")
def client_fixture(mock_manager):
    """
    Creates a TestClient for the FastAPI app.
    Reloads the module to ensure mocks are applied.
    """
    # We need to reload the app module to make sure the mocked ConnectionManager is used
    if APP_MODULE in sys.modules:
        del sys.modules[APP_MODULE]
        
    # Import the app
    # We use importlib to handle the long filename comfortably if needed, or just import
    import importlib
    module = importlib.import_module(APP_MODULE)
    
    from fastapi.testclient import TestClient
    client = TestClient(module.app)
    
    # Attach the mock manager to the client for easy access in tests if needed
    # (The app uses 'manager' global variable, which is now our mock_instance)
    client.app_manager = module.manager 
    
    yield client
