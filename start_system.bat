@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" start_system.py %*
) else (
  python start_system.py %*
)
pause
