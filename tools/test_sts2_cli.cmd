@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0test_sts2_cli.ps1" %*
exit /b %ERRORLEVEL%
