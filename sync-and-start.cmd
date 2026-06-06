@echo off
cd /d "%~dp0"
python sync_from_github.py
python bot.py
pause
