$ErrorActionPreference = "Stop"

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$assetsDir = Join-Path $rootDir "public\assets"

$modelUrl = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx?download=1"
$labelsUrl = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1"

$modelPath = Join-Path $assetsDir "BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx"
$labelsPath = Join-Path $assetsDir "BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv"

New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null

Write-Host "Downloading model..."
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath -UseBasicParsing

Write-Host "Downloading labels..."
Invoke-WebRequest -Uri $labelsUrl -OutFile $labelsPath -UseBasicParsing

Write-Host "Done:"
Get-Item $modelPath, $labelsPath | Format-Table -AutoSize
