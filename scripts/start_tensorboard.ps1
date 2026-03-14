$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

& .\.venv\Scripts\python.exe -m tensorboard.main --logdir runs --port 6006
