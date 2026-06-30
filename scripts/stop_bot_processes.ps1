$BotDir = "C:\trading bot"
$escapedBotDir = [regex]::Escape($BotDir)

Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @('python.exe', 'pythonw.exe') -and
        $_.CommandLine -match $escapedBotDir -and
        $_.CommandLine -match '(bot_daemon\.py|run_bot\.py|main\.py)'
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
