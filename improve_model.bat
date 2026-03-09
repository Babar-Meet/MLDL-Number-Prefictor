@echo off
setlocal

echo ============================================
echo   MNIST Model Improvement Training
echo   (resume from saved best + keep previous best)
echo ============================================
echo.

set "EPOCHS="
set /p "EPOCHS=Enter epochs to train [12]: "
if "%EPOCHS%"=="" set "EPOCHS=12"

set "PYTHON_CMD=python"
if exist "B:\_MeetData\installation\Python_loc\python.exe" (
	set "PYTHON_CMD=B:\_MeetData\installation\Python_loc\python.exe"
)

%PYTHON_CMD% improve_model.py --epochs %EPOCHS%
pause
