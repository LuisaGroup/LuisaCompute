@echo off

set LUISA_PATH=%~dp0
set LUISA_PATH=%LUISA_PATH:~0,-1%
set PATH=%PATH%;%LUISA_PATH%\build\bin
set PYTHONPATH=%PYTHONPATH%;%LUISA_PATH%\build\bin;%LUISA_PATH%\pyscenes