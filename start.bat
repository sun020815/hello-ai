@echo off
rem One-click start for Windows. ASCII only in output to avoid codepage garbling.
setlocal
cd /d "%~dp0"

if not exist ".env" (
    echo [!] .env not found, creating one from .env.example ...
    copy /Y ".env.example" ".env" >nul
    echo [!] Fill in LLM_API_KEY in .env, then run this file again.
    notepad ".env"
    exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [!] uv not found. Install it first:
    echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    exit /b 1
)

echo Starting server at http://127.0.0.1:8000
echo (Ctrl+C to stop)
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
uv run uvicorn main:app --host 127.0.0.1 --port 8000
