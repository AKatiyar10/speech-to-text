#!/bin/bash

echo "==========================================="
echo " SPEECH-TO-TEXT WITH SPEAKER DIARIZATION"
echo "==========================================="
echo

# Activate virtual environment
source v-speech-to-text/Scripts/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate virtual environment"
    echo "Make sure v-speech-to-text virtual environment exists"
    exit 1
fi

echo "Virtual environment activated successfully"
echo

# Start the server
echo "Starting server on http://localhost:8000..."
echo "Press Ctrl+C to stop the server"
echo
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py