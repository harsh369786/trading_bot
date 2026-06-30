<#
Creates Windows Task Scheduler jobs for the NSE trading bot.
Run from an elevated PowerShell prompt.
#>

$BotDir = "C:\trading bot"
$PythonW = "C:\Users\hshah\AppData\Local\Programs\Python\Python310\pythonw.exe"
$Python = "C:\Users\hshah\AppData\Local\Programs\Python\Python310\python.exe"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew

# Start the watchdog each weekday.
$ActionDaemon = New-ScheduledTaskAction `
    -Execute $PythonW `
    -Argument "bot_daemon.py" `
    -WorkingDirectory $BotDir
$TriggerDaemon = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 09:05AM
Register-ScheduledTask `
    -Action $ActionDaemon `
    -Trigger $TriggerDaemon `
    -Settings $Settings `
    -TaskName "NSEBot_Daemon" `
    -Description "Starts the NSE bot watchdog at 09:05 on weekdays." `
    -Force

# Check the daemon heartbeat every 10 minutes during the market session.
$GuardScriptPath = Join-Path $BotDir "scripts\daemon_guard.ps1"
$ActionGuard = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$GuardScriptPath`""
$TriggerGuard = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 09:05AM
$GuardSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew
Register-ScheduledTask `
    -Action $ActionGuard `
    -Trigger $TriggerGuard `
    -Settings $GuardSettings `
    -TaskName "NSEBot_DaemonGuard" `
    -Description "Restarts the daemon when its heartbeat is stale." `
    -Force
$xml = Export-ScheduledTask -TaskName "NSEBot_DaemonGuard"
$xml = $xml -replace '(<Triggers>[\s\S]*?<CalendarTrigger>[\s\S]*?)(</CalendarTrigger>)', `
    '$1<Repetition><Interval>PT10M</Interval><Duration>PT8H</Duration><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>$2'
Register-ScheduledTask -Xml $xml -TaskName "NSEBot_DaemonGuard" -Force | Out-Null

# Stop only this bot's runtime processes. Training is intentionally excluded.
$StopScriptPath = Join-Path $BotDir "scripts\stop_bot_processes.ps1"
$ActionStop = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StopScriptPath`""
$TriggerStop = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 17:10PM
Register-ScheduledTask `
    -Action $ActionStop `
    -Trigger $TriggerStop `
    -Settings $Settings `
    -TaskName "NSEBot_Stop" `
    -Description "Stops only NSE bot runtime processes at 17:10." `
    -Force

# Send the deduplicated EOD email.
$ActionEOD = New-ScheduledTaskAction `
    -Execute $PythonW `
    -Argument "scripts\eod_summary.py" `
    -WorkingDirectory $BotDir
$TriggerEOD = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 17:06PM
Register-ScheduledTask `
    -Action $ActionEOD `
    -Trigger $TriggerEOD `
    -Settings $Settings `
    -TaskName "NSEBot_EOD" `
    -Description "Sends the EOD paper-trading summary." `
    -Force

# Refresh history and retrain after Friday shutdown.
$ActionRetrain = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-B scripts\retrain_weekly.py" `
    -WorkingDirectory $BotDir
$TriggerRetrain = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Friday `
    -At 17:15PM
Register-ScheduledTask `
    -Action $ActionRetrain `
    -Trigger $TriggerRetrain `
    -Settings $Settings `
    -TaskName "NSEBot_Retrain" `
    -Description "Refreshes history and deploys only validated models." `
    -Force

Write-Host ""
Write-Host "NSE bot scheduled tasks registered." -ForegroundColor Green
Write-Host "  NSEBot_Daemon      09:05 weekdays"
Write-Host "  NSEBot_DaemonGuard every 10 minutes for 8 hours"
Write-Host "  NSEBot_EOD         17:06 weekdays"
Write-Host "  NSEBot_Stop        17:10 weekdays"
Write-Host "  NSEBot_Retrain     17:15 Fridays"
