@echo off
echo ===========================================
echo  SPEECH-TO-TEXT WITH SPEAKER DIARIZATION
echo ===========================================
echo.

REM Activate virtual environment
call v-speech-to-text\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment
    echo Make sure v-speech-to-text virtual environment exists
    pause
    exit /b 1
)

echo Virtual environment activated successfully
echo.

REM Start the server
echo Starting server on http://localhost:8000...
echo Press Ctrl+C to stop the server
echo.
python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py

pause