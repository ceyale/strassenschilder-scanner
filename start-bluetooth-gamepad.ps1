$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Usbipd = 'C:\Program Files\usbipd-win\usbipd.exe'
$Cloudflared = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
$BusId = '2-10'

# Start the Windows HTTP/API server if it is not already listening.
$httpPort = Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue
if (-not $httpPort) {
    $python = (Get-Command python -ErrorAction Stop).Source
    Start-Process -FilePath $python -ArgumentList @('-u', 'server.py') `
        -WorkingDirectory $Project -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Project 'backend\server.stdout.log') `
        -RedirectStandardError (Join-Path $Project 'backend\server.stderr.log')
}

# Expose the HTTPS website/API endpoint for the phone using a temporary tunnel.
if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) {
    Clear-Content -Path (Join-Path $Project 'backend\cloudflared.log') -ErrorAction SilentlyContinue
    Start-Process -FilePath $Cloudflared `
        -ArgumentList @('tunnel', '--url', 'http://127.0.0.1:8765', '--no-autoupdate') `
        -WorkingDirectory $Project -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Project 'backend\cloudflared.log') `
        -RedirectStandardError (Join-Path $Project 'backend\cloudflared.log')
}

# Keep WSL2 active, then attach the host Bluetooth adapter (non-persistent).
$runningDistros = ((& wsl.exe --list --running --quiet 2>$null | Out-String) -replace "`0", '')
if ($runningDistros -notmatch 'Ubuntu') {
    Start-Process -FilePath 'wsl.exe' -ArgumentList @(
        '-d', 'Ubuntu', '-u', 'root', '--exec', 'sleep', 'infinity'
    ) -WindowStyle Hidden
}
$attached = $false
for ($i = 0; $i -lt 20; $i++) {
    $device = & $Usbipd list | Select-String '^2-10\s'
    if (-not $device) {
        throw "Bluetooth USB device $BusId not found. Check the adapter in usbipd list."
    }
    if ($device.Line -match 'Attached') {
        $attached = $true
        break
    }
    if ($device.Line -match 'Shared') {
        & $Usbipd attach --wsl --busid $BusId
        if ($LASTEXITCODE -eq 0) {
            $attached = $true
            break
        }
    }
    Start-Sleep -Seconds 1
}
if (-not $attached) {
    throw "Bluetooth USB device $BusId could not be attached to WSL."
}
& wsl.exe -d Ubuntu -u root --exec systemctl restart bluetooth.service
& wsl.exe -d Ubuntu -u root --exec systemctl restart ev3-gamepad.service

# Record the current random Quick Tunnel URL for the phone's PC-Backend field.
$tunnelLog = Join-Path $Project 'backend\cloudflared.log'
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path $tunnelLog) {
        $match = [regex]::Match((Get-Content $tunnelLog -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($match.Success) {
            Set-Content -Path (Join-Path $Project 'backend\cloudflare-url.txt') `
                -Value ($match.Value + '/api/signs') -Encoding utf8
            break
        }
    }
    Start-Sleep -Seconds 1
}
