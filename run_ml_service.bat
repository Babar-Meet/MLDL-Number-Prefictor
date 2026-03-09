@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title NumberPredictor ML Service

set "PYTHON_CMD=python"
if exist "B:\_MeetData\installation\Python_loc\python.exe" (
    set "PYTHON_CMD=B:\_MeetData\installation\Python_loc\python.exe"
)

echo Starting FastAPI ML service on http://127.0.0.1:8000
"%PYTHON_CMD%" -m uvicorn backend.ml_server:app --host 127.0.0.1 --port 8000

echo.
echo ML service stopped.
pause