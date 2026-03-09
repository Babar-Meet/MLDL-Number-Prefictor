@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title Ultimate Neural Network Visualizer Setup

set "ROOT_DIR=%CD%"
set "PYTHON_CMD=python"

if exist "B:\_MeetData\installation\Python_loc\python.exe" (
    set "PYTHON_CMD=B:\_MeetData\installation\Python_loc\python.exe"
)

echo ==============================================================
echo Ultimate Neural Network Visualizer Setup
echo ==============================================================
echo.

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not available in PATH.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm is not installed or not available in PATH.
    pause
    exit /b 1
)

"%PYTHON_CMD%" --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or could not be started.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [INFO] Creating .env with the current showcase defaults...
    > ".env" echo MONGODB_DB_NAME=NumberPredictor
    >> ".env" echo MONGODB_COLLECTION=predictionHistory
    >> ".env" echo MAX_HISTORY=18
    >> ".env" echo PYTHON_SERVICE_URL=http://127.0.0.1:8000
    >> ".env" echo PORT=4000
    >> ".env" echo MONGODB_URI=mongodb+srv://babarmeet:CsnGjydsmoLorQT6@numberpredictor.wu2qung.mongodb.net/NumberPredictor?retryWrites=true^&w=majority^&appName=NumberPredictor
    >> ".env" echo VITE_API_BASE_URL=http://localhost:4000
    >> ".env" echo VITE_SOCKET_URL=http://localhost:4000
) else (
    echo [INFO] Existing .env found. Leaving it unchanged.
)

echo.
echo [1/5] Installing Python packages...
call "%PYTHON_CMD%" -m pip install -r backend\requirements.txt
if errorlevel 1 goto :fail

echo.
echo [2/5] Installing backend packages...
pushd backend
call npm install
if errorlevel 1 (
    popd
    goto :fail
)
popd

echo.
echo [3/5] Installing frontend packages...
pushd frontend
call npm install
if errorlevel 1 (
    popd
    goto :fail
)
popd

echo.
echo [4/5] Preparing trained model assets...
if exist "backend\mnist_visualizer_model.pth" if exist "backend\model_snapshot.json" (
    echo [INFO] Existing trained model found. Skipping retraining.
) else (
    call "%PYTHON_CMD%" train_model.py
    if errorlevel 1 goto :fail
)

echo.
echo [5/5] Building the production frontend...
pushd frontend
call npm run build
if errorlevel 1 (
    popd
    goto :fail
)
popd

echo.
echo Setup completed successfully.
echo.
choice /M "Launch the showcase now"
if errorlevel 2 goto :done
call "%ROOT_DIR%\launch_showcase.bat"
goto :done

:fail
echo.
echo [ERROR] Setup did not complete successfully.
pause
exit /b 1

:done
echo.
echo You can relaunch the project anytime with launch_showcase.bat
pause
exit /b 0