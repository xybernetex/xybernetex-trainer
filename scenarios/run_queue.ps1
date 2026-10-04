<#
Runs the pending experiment batches one after another, unattended.

    cd "G:\My Drive\Codex\Xybernetex\Xybernetex Prod\xybernetex-trainer"
    powershell -ExecutionPolicy Bypass -File scenarios\run_queue.ps1
    powershell -ExecutionPolicy Bypass -File scenarios\run_queue.ps1 -IncludeTerminalBench

- One batch at a time, so Workers AI rate limits don't starve each other.
- Safe to stop and rerun: every batch resumes, skipping tasks it already
  finished and re-running tasks lost to rate limits. Finished batches take
  seconds.
- A batch that crashes is logged and the queue moves on.
- Keeps the PC awake while it runs (for this process only; no settings change).
- Reads the Workers AI account and key from ~/.openclaw/openclaw.json into this
  process's environment. They are never printed or written to a file.
- Output: scenarios\queue-logs\<batch>.log per batch, and a summary at the end.
#>
param(
    [switch]$IncludeTerminalBench,  # adds the Terminal-Bench batches: many hours
    [string]$Set = "default"        # default | contracts2 (contract v2 against the v1 normal-effort control)
)

$ErrorActionPreference = "Continue"
$trainer = Split-Path -Parent $PSScriptRoot
Set-Location $trainer
$env:PYTHONPATH = Join-Path (Split-Path -Parent $trainer) "xybernetex-python\src"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$config = Get-Content (Join-Path $HOME ".openclaw\openclaw.json") -Raw | ConvertFrom-Json
$provider = $config.models.providers.'workers-ai'
$env:CF_ACCOUNT_ID = (($provider.baseUrl -split '/accounts/')[1] -split '/')[0]
$env:CF_WORKERS_AI_KEY = $provider.apiKey
if (-not $env:CF_ACCOUNT_ID -or -not $env:CF_WORKERS_AI_KEY) { Write-Host "No Workers AI credentials in openclaw.json"; exit 1 }

# Keep the machine awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
Add-Type -Namespace Xyb -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[void][Xyb.Power]::SetThreadExecutionState([uint32]2147483649)

# name, module, arguments. Follow-ups and fixes run on DeepSeek V4 Pro, as in lg-hard3-check.
$queue = @(
    @("lg-hard3-none",      "scenarios.lg_agent",  "--category hard3 --model glm53f"),
    @("lg-hard3-contract",  "scenarios.lg_agent",  "--category hard3 --model glm53f --verify-model dsv4p --contract auto"),
    @("sdk-hard3-none",     "scenarios.sdk_agent", "--category hard3 --model glm53f"),
    @("sdk-hard3-check",    "scenarios.sdk_agent", "--category hard3 --model glm53f --verify-model dsv4p"),
    @("sdk-hard3-contract", "scenarios.sdk_agent", "--category hard3 --model glm53f --verify-model dsv4p --contract auto"),
    @("lg-hard2-r2",        "scenarios.lg_agent",  "--category hard2 --model glm53f")
)
if ($Set -eq "contracts2") {
    $queue = @(
        @("lg-hard3-contract2",        "scenarios.lg_agent",  "--category hard3 --model glm53f --verify-model dsv4p --contract auto --contract-version v2"),
        @("sdk-hard3-contract2",       "scenarios.sdk_agent", "--category hard3 --model glm53f --verify-model dsv4p --contract auto --contract-version v2"),
        @("lg-hard3-contract-normal",  "scenarios.lg_agent",  "--category hard3 --model glm53f --verify-model dsv4p --contract auto --writer-effort normal"),
        @("sdk-hard3-contract-normal", "scenarios.sdk_agent", "--category hard3 --model glm53f --verify-model dsv4p --contract auto --writer-effort normal")
    )
}
if ($IncludeTerminalBench) {
    $queue += ,@("tbpy-sdk-survey", "scenarios.tb_py", "--framework sdk --model glm53f --from-survey")
    $queue += ,@("tbpy-lg-survey",  "scenarios.tb_py", "--framework langgraph --model glm53f --from-survey")
}

$logs = Join-Path $PSScriptRoot "queue-logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$results = @()
$started = Get-Date

foreach ($item in $queue) {
    $name, $module, $arguments = $item
    $log = Join-Path $logs "$name.log"
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=== $name   started $($t0.ToString('HH:mm'))   log: $log"
    "=== $name   started $t0" | Out-File -Append -Encoding utf8 $log
    & cmd /c "python -m $module $arguments --batch $name 2>&1" | ForEach-Object { Write-Host $_; $_ | Out-File -Append -Encoding utf8 $log }
    $code = $LASTEXITCODE
    $minutes = [math]::Round(((Get-Date) - $t0).TotalMinutes)
    $summary = Select-String -Path $log -Pattern '^\d+ tasks: ' | Select-Object -Last 1
    $line = if ($summary) { $summary.Line } else { "(no summary line - see the log)" }
    $results += "{0,-20} exit {1}  {2,4} min  {3}" -f $name, $code, $minutes, $line
}

[void][Xyb.Power]::SetThreadExecutionState([uint32]2147483648)
Write-Host ""
Write-Host "=== Queue finished in $([math]::Round(((Get-Date) - $started).TotalHours, 1)) hours"
$results | ForEach-Object { Write-Host $_ }
$results | Out-File -Encoding utf8 (Join-Path $logs "summary.txt")
