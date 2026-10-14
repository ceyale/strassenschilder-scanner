$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build-cloudflare.ps1')
if (-not $?) { throw 'Cloudflare Pages build preparation failed.' }
Push-Location $root
try {
  npx wrangler pages deploy dist --project-name strassenschilder-scanner --branch main
  if ($LASTEXITCODE -ne 0) { throw 'Cloudflare Pages deployment failed.' }
} finally {
  Pop-Location
}
