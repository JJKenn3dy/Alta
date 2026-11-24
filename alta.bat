@echo off
cd /d %~dp0
pip install textual httpx -q
python launcher.py
