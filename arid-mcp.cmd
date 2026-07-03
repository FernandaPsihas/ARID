@echo off
REM Forced command for the ARID MCP key (see administrators_authorized_keys).
REM sshd runs this; its stdin/stdout become the MCP stdio channel. Keep it silent.
REM Absolute python path: sshd forced commands get the SYSTEM PATH, where python
REM (installed under the user profile) is NOT on PATH.
cd /d C:\Users\Kieran\ARID
"C:\Users\Kieran\AppData\Local\Programs\Python\Python312\python.exe" EGEpipeline\arid_mcp.py
