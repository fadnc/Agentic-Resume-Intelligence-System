@echo off
setlocal
set "ACTION=%~1"
if not defined ACTION set "ACTION=start"
if /I "%ACTION%"=="start" goto run
if /I "%ACTION%"=="stop" goto run
if /I "%ACTION%"=="restart" goto run
if /I "%ACTION%"=="status" goto run
echo Usage: %~nx0 [start^|stop^|restart^|status]
exit /b 2

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0local-services.ps1" -Action "%ACTION%"
exit /b %ERRORLEVEL%
