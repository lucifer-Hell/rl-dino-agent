$ErrorActionPreference = "Stop"

$python = "C:\Users\panka\AppData\Local\Programs\Python\Python312\python.exe"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & $python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m playwright install chromium

Write-Host "Setup complete. Activate with .\.venv\Scripts\Activate.ps1"

