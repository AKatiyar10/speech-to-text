import pytest
import os
import tempfile
from pathlib import Path
import sys

# Ensure backend passes imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation_manager import ConversationManager

@pytest.fixture
def temp_conv_manager():
    # Create a wrapper to use a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "test_conversations.md")
        cm = ConversationManager(md_file=tmp_file)
        yield cm

def test_conversation_manager_init(temp_conv_manager):
    assert temp_conv_manager.md_file.exists()
    assert temp_conv_manager.get_conversation_history() == []

def test_add_entry(temp_conv_manager):
    temp_conv_manager.add_entry(speaker="Speaker1", text="Hello world", confidence=0.95)
    
    history = temp_conv_manager.get_conversation_history()
    assert len(history) == 1
    assert history[0]["speaker"] == "Speaker1"
    assert "Hello world" in history[0]["text"]

def test_get_context(temp_conv_manager):
    temp_conv_manager.add_entry("A", "Hi", 1.0)
    temp_conv_manager.add_entry("B", "Hello", 1.0)
    
    context = temp_conv_manager.get_recent_context(num_entries=2)
    assert "[A (100%)]" in context
    assert "Hi" in context
    assert "[B (100%)]" in context

def test_get_history_for_speaker(temp_conv_manager):
    temp_conv_manager.add_entry("User", "Command 1", 1.0)
    temp_conv_manager.add_entry("Assistant", "Response 1", 1.0)
    temp_conv_manager.add_entry("User", "Command 2", 1.0)
    
    users = temp_conv_manager.get_conversation_history(speaker="User")
    assert len(users) == 2
    # get_conversation_history returns most recent first
    assert "Command 2" in users[0]["text"]
    assert "Command 1" in users[1]["text"]
