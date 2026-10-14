$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root 'dist'
if (Test-Path -LiteralPath $dist) {
  Remove-Item -LiteralPath $dist -Recurse -Force
}
New-Item -ItemType Directory -Path $dist | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dist 'src') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dist 'models') | Out-Null

Copy-Item -LiteralPath (Join-Path $root 'index.html') -Destination $dist
Copy-Item -LiteralPath (Join-Path $root 'src/app.js') -Destination (Join-Path $dist 'src/app.js')
Copy-Item -LiteralPath (Join-Path $root 'src/detector.js') -Destination (Join-Path $dist 'src/detector.js')
Copy-Item -LiteralPath (Join-Path $root 'src/style.css') -Destination (Join-Path $dist 'src/style.css')
Copy-Item -LiteralPath (Join-Path $root 'models/gtsrb-sign-cnn.onnx') -Destination (Join-Path $dist 'models/gtsrb-sign-cnn.onnx')
Copy-Item -LiteralPath (Join-Path $root 'models/emnist-alphanumeric.onnx') -Destination (Join-Path $dist 'models/emnist-alphanumeric.onnx')
Copy-Item -LiteralPath (Join-Path $root 'models/sign-binary-32.onnx') -Destination (Join-Path $dist 'models/sign-binary-32.onnx')
Copy-Item -LiteralPath (Join-Path $root 'models/learnable-box-head.json') -Destination (Join-Path $dist 'models/learnable-box-head.json')
Write-Host "Cloudflare Pages files prepared at $dist"
