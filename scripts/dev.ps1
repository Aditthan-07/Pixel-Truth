param (
    [Parameter(Position=0)]
    [ValidateSet("test", "run", "docker-build", "docker-up", "clean", "help")]
    [string]$Task = "help"
)

switch ($Task) {
    "test" {
        Write-Host "[*] Executing unit test suite..." -ForegroundColor Cyan
        python -m unittest discover -s tests
    }
    "run" {
        Write-Host "[*] Starting PixelTruth server..." -ForegroundColor Cyan
        python run.py
    }
    "docker-build" {
        Write-Host "[*] Building Docker container..." -ForegroundColor Cyan
        docker compose build
    }
    "docker-up" {
        Write-Host "[*] Starting Docker container..." -ForegroundColor Cyan
        docker compose up
    }
    "clean" {
        Write-Host "[*] Cleaning cache files..." -ForegroundColor Cyan
        Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
        Write-Host "[+] Clean complete." -ForegroundColor Green
    }
    default {
        Write-Host "PixelTruth Windows Task Runner" -ForegroundColor Yellow
        Write-Host "Usage: .\scripts\dev.ps1 [test | run | docker-build | docker-up | clean]"
    }
}
