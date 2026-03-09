@echo off
setlocal EnableExtensions

cd /d "%~dp0\backend"
title NumberPredictor Showcase App

echo Starting showcase app on http://127.0.0.1:4000
node server.js

echo.
echo Showcase app stopped.
pause