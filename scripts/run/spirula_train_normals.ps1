param(
    [Parameter(Mandatory)][string]$DatasetRoot,
    [Parameter(Mandatory)][string]$Spirula,
    [string]$GeometryCli = (Join-Path $PSScriptRoot '../../src-tauri/target/debug/spherealign-geometry-cli.exe'),
    [string]$RunName = ('normals-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)
$ErrorActionPreference = 'Stop'
$dataset = (Resolve-Path -LiteralPath $DatasetRoot).Path
$trainer = (Resolve-Path -LiteralPath $Spirula).Path
$geometry = (Resolve-Path -LiteralPath $GeometryCli).Path
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscores or hyphens.' }
if (Test-Path -LiteralPath (Join-Path $dataset "outputs/$RunName")) { throw 'Training output already exists; choose a new RunName.' }
# Uses the same production export, source hashes, resume and cleanup as the app.
& $geometry normals $dataset
if ($LASTEXITCODE -ne 0) { throw 'Normal generation or verification failed; training was not started.' }
$config = Get-Content -LiteralPath (Join-Path $dataset 'geometry/spirula-normals.json') -Raw | ConvertFrom-Json
$trainerArgs = @('train', [string]$config.preset)
foreach ($property in $config.PSObject.Properties) {
    if ($property.Name -eq 'preset') { continue }
    $value = if ($property.Value -is [bool]) { [int]$property.Value } else { $property.Value }
    $trainerArgs += ('--' + $property.Name.Replace('_', '-'))
    $trainerArgs += [Convert]::ToString($value, [Globalization.CultureInfo]::InvariantCulture)
}
$trainerArgs += @('--output-dir-prefix', (Join-Path $dataset 'outputs'), '--output-dir-name', $RunName)
& $trainer @trainerArgs
if ($LASTEXITCODE -ne 0) { throw "Spirula training failed with exit code $LASTEXITCODE" }
