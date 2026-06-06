@echo off
cd /d "%~dp0"
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared was not found.
  echo Install it first, then run this file again.
  pause
  exit /b 1
)
echo Starting HTTPS tunnel for ClawBot backend...
echo Copy the https://*.trycloudflare.com address into the web page.
cloudflared tunnel --url http://127.0.0.1:8765
pause
