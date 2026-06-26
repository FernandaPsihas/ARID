@echo off
REM Forced command for the ARID MCP key (see administrators_authorized_keys).
REM sshd runs this; its stdin/stdout become the MCP stdio channel. Keep it silent.
cd /d C:\Users\Kieran\ARID
python EGEpipeline\arid_mcp.py
