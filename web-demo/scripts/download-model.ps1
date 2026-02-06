$ErrorActionPreference = "Stop"

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$assetsDir = Join-Path $rootDir "public\assets"

$modelUrl = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx?download=1"
$labelsUrl = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_Labels.csv?download=1"

$modelPath = Join-Path $assetsDir "BirdNET+_V3.0-preview3_Global_11K_FP32.onnx"
$labelsPath = Join-Path $assetsDir "BirdNET+_V3.0-preview3_Global_11K_Labels.csv"

New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null

Write-Host "Downloading model..."
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing

Write-Host "Downloading labels..."
Invoke-WebRequest -Uri $labelsUrl -OutFile $labelsPath -UseBasicParsing

Write-Host "Done:"
Get-Item $modelPath, $labelsPath | Format-Table -AutoSize
