$ErrorActionPreference = "Stop"

function Run-TestLoop {
    param(
        [string]$Name,
        [string]$Command,
        [int]$Iterations = 50,
        [int]$TimeoutSeconds = 30
    )
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  Starting 50x Stress & Integrity Loop: $Name" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan

    $passCount = 0
    $totalSw = [System.Diagnostics.Stopwatch]::StartNew()

    for ($i = 1; $i -le $Iterations; $i++) {
        Write-Host -NoNewline "  [$Name] Run $i/$Iterations ... "
        $iterSw = [System.Diagnostics.Stopwatch]::StartNew()

        $process = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-Command", $Command -PassThru -NoNewWindow
        $finished = $process.WaitForExit($TimeoutSeconds * 1000)

        if (-not $finished) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Write-Host "FAILED (TIMEOUT after ${TimeoutSeconds}s)" -ForegroundColor Red
            exit 1
        }

        $iterSw.Stop()
        if ($process.ExitCode -ne 0) {
            Write-Host "FAILED (ExitCode: $($process.ExitCode))" -ForegroundColor Red
            exit 1
        }

        $passCount++
        $ms = [math]::Round($iterSw.Elapsed.TotalMilliseconds, 0)
        Write-Host "PASS (${ms}ms)" -ForegroundColor Green
    }

    $totalSw.Stop()
    $totalSec = [math]::Round($totalSw.Elapsed.TotalSeconds, 1)
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  RESULT: $Name -> $passCount/$Iterations PASSED cleanly in ${totalSec}s (Zero errors/timeouts)" -ForegroundColor Green
    Write-Host "================================================================================`n" -ForegroundColor Cyan
}

# 1. Run Titan VM Distributed Graph Stress Engine 50 times (Timeout: 30s per iteration)
Run-TestLoop -Name "Titan VM Real-World Engine" -Command ".\zig-out\bin\titan_concurrent_gc_engine.exe *> `$null" -Iterations 50 -TimeoutSeconds 30

# 2. Run Concurrent GC + JavaThread Production Stress Test 50 times (Timeout: 15s per iteration)
Run-TestLoop -Name "Concurrent GC Thread Stress" -Command ".\zig-out\bin\concurrent_gc_thread_stress.exe *> `$null" -Iterations 50 -TimeoutSeconds 15

# 3. Run Aether VM Chaos Stress Engine 50 times (Timeout: 15s per iteration)
Run-TestLoop -Name "Aether VM Chaos Engine" -Command ".\zig-out\bin\chaos_concurrent_vm_stress.exe *> `$null" -Iterations 50 -TimeoutSeconds 15

# 4. Run Full Unit Test Suite (203 tests) 50 times (Timeout: 45s per iteration)
Run-TestLoop -Name "Full Unit Test Suite (203 tests)" -Command "zig build test *> `$null" -Iterations 50 -TimeoutSeconds 45

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  ALL 200 RUNS (50x PER SUITE) COMPLETED WITH 100% SUCCESS! ZERO REGRESSIONS!   " -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
