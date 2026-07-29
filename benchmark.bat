@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0benchmark.ps1" %*
exit /b %ERRORLEVEL%
