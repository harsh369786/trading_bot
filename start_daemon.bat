@echo off
title NSE Bot Watchdog Daemon
cd /d "C:\trading bot"
echo.
echo  ============================================
echo   NSE Trading Bot — Watchdog Daemon
echo   Auto-restarts bot on crash during market hours
echo   Press Ctrl+C to stop the daemon
echo  ============================================
echo.
"C:\Users\hshah\AppData\Local\Programs\Python\Python310\python.exe" bot_daemon.py
echo.
echo Daemon stopped. Press any key to exit.
pause >nul
