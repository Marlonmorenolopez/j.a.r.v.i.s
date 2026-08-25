@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py"
if errorlevel 1 (
  echo.
  echo MARK XXXIX-OR se cerro con un error. Revise el mensaje anterior.
  pause
)
