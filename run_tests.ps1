# Run every automated test (backend + frontend) with coverage. From the repo root:
#     powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
#
# These tests need NO Ollama, NO embedding model download and NO database server:
# they use fakes (FakeEmbedder, FakeLLM, in-memory SQLite, mocked fetch), so they
# are fast and give the same result on any machine. The slow, model-based
# evaluations are separate (see backend/eval/README.md).
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$failed = @()

Write-Host "`n=== Backend: pytest with coverage ===" -ForegroundColor Cyan
Push-Location (Join-Path $root 'backend')
& .\.venv\Scripts\python.exe -m pytest -q --cov=app --cov-report=term-missing:skip-covered
if ($LASTEXITCODE -ne 0) { $failed += 'backend' }
Pop-Location

Write-Host "`n=== Frontend: type-check, lint, vitest with coverage ===" -ForegroundColor Cyan
Push-Location (Join-Path $root 'frontend')
npx tsc -b
if ($LASTEXITCODE -ne 0) { $failed += 'frontend-typecheck' }
npm run lint --silent
if ($LASTEXITCODE -ne 0) { $failed += 'frontend-lint' }
npm run coverage --silent
if ($LASTEXITCODE -ne 0) { $failed += 'frontend-tests' }
Pop-Location

if ($failed.Count -gt 0) {
    Write-Host "`nFAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll checks passed." -ForegroundColor Green
