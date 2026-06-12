# setup.ps1 — Register both trading bot scheduled tasks on a new machine.
# Run once after cloning the repo:
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = (Get-Command python -ErrorAction Stop).Source

Write-Host "Project root : $ProjectRoot"
Write-Host "Python       : $Python"

# Create logs directory if missing
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

# ---------------------------------------------------------------------------
# Task 1: Daily swing bot — 6:35 AM PST Mon-Fri (~90 seconds, then exits)
# ---------------------------------------------------------------------------
$action1 = New-ScheduledTaskAction `
    -Execute "$ProjectRoot\run_scheduler.bat"

$trigger1 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "06:35AM"

$settings1 = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName "TradingBot_Daily_635AM" `
    -TaskPath "\TradingModel\" `
    -Action $action1 `
    -Trigger $trigger1 `
    -Settings $settings1 `
    -Principal $principal `
    -Description "Daily swing bot: 6:35 AM PST Mon-Fri. Logs to logs\scheduler.log" `
    -Force | Out-Null

Write-Host "Registered : TradingBot_Daily_635AM  (6:35 AM PST)"

# ---------------------------------------------------------------------------
# Task 2: Intraday scalping bot — 6:30 AM PST Mon-Fri (runs until 3:55 PM ET)
# ---------------------------------------------------------------------------
$action2 = New-ScheduledTaskAction `
    -Execute "$ProjectRoot\run_intraday.bat"

$trigger2 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "06:30AM"

$settings2 = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName "TradingBot_Intraday" `
    -TaskPath "\TradingModel\" `
    -Action $action2 `
    -Trigger $trigger2 `
    -Settings $settings2 `
    -Principal $principal `
    -Description "Intraday scalping bot: 6:30 AM PST, runs until 3:55 PM ET. Logs to logs\intraday.log" `
    -Force | Out-Null

Write-Host "Registered : TradingBot_Intraday       (6:30 AM PST)"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Scheduled tasks registered under \TradingModel\:"
Get-ScheduledTask -TaskPath "\TradingModel\" | Select-Object TaskName, State | Format-Table -AutoSize

Write-Host "Setup complete. Make sure .env is in $ProjectRoot before the first run."
