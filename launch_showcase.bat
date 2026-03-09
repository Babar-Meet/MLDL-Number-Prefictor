@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title Ultimate Neural Network Visualizer Launcher

if not exist "frontend\dist\index.html" (
    echo [ERROR] Frontend build was not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

if not exist "backend\mnist_visualizer_model.pth" (
    echo [ERROR] Trained model was not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

echo Starting the ML service and showcase app...

start "NumberPredictor ML Service" "%CD%\run_ml_service.bat"
start "NumberPredictor Showcase App" "%CD%\run_showcase_app.bat"

echo Waiting for the showcase app to respond...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(45); $ready=$false; while((Get-Date) -lt $deadline){ try { Invoke-WebRequest -Uri 'http://127.0.0.1:4000/api/health' -UseBasicParsing | Out-Null; $ready=$true; break } catch { Start-Sleep -Seconds 2 } }; if($ready){ Start-Process 'http://localhost:4000'; exit 0 } else { Write-Host 'The showcase server did not become ready in time.'; exit 1 }"
if errorlevel 1 (
    echo.
    echo [WARNING] The browser was not opened automatically because the app did not respond in time.
    echo Check the two service windows for any error messages, then open http://localhost:4000 manually.
)

echo.
echo The showcase is starting in separate windows.
echo Close those service windows when you want to stop the project.
pause
exit /b 0