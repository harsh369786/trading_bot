$pythonw = 'C:\Users\hshah\AppData\Local\Programs\Python\Python310\pythonw.exe'
$botDir = 'C:\trading bot'
$heartbeatPath = Join-Path $botDir 'data\daemon_heartbeat.txt'
$now = [DateTimeOffset]::Now

if ($now.TimeOfDay -lt [TimeSpan]'09:05' -or $now.TimeOfDay -gt [TimeSpan]'17:10') { exit 0 }

$fresh = $false
if (Test-Path $heartbeatPath) {
    try {
        $timestamp = (Get-Content $heartbeatPath -Raw).Trim().Split(' ')[0]
        $lastBeat = [DateTimeOffset]::Parse($timestamp)
        $fresh = ($now - $lastBeat).TotalSeconds -lt 180
    } catch {
        $fresh = $false
    }
}

if (-not $fresh) {
    Write-Host 'DaemonGuard: heartbeat stale - restarting daemon.'
    Start-Process -FilePath $pythonw -ArgumentList 'bot_daemon.py' -WorkingDirectory $botDir -WindowStyle Hidden
} else {
    Write-Host 'DaemonGuard: heartbeat OK.'
}
