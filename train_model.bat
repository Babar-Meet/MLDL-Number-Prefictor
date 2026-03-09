@echo off
setlocal

set "PYTHON_CMD=python"
if exist "B:\_MeetData\installation\Python_loc\python.exe" (
	set "PYTHON_CMD=B:\_MeetData\installation\Python_loc\python.exe"
)

%PYTHON_CMD% train_model.py
pause
