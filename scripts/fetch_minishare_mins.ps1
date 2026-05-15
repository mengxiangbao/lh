$ErrorActionPreference = "Stop"

param(
    [string]$Codes = "600000.SH",
    [string]$Freq = "5min",
    [string]$Start = "20250210 09:00:00",
    [string]$End = "20250210 19:00:00",
    [switch]$Combine
)

if (-not $env:MINISHARE_TOKEN) {
    throw "Please set MINISHARE_TOKEN first, for example: `$env:MINISHARE_TOKEN='your-token'"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}

$localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
if ((-not $python) -and (Test-Path $localPython)) {
    $pythonPath = $localPython
} elseif ($python) {
    $pythonPath = $python.Source
} else {
    throw "Python was not found. Install Python 3.11+ or add it to PATH."
}

$combineArg = @()
if ($Combine) {
    $combineArg = @("--combine")
}

if ($python -and $python.Name -eq "py.exe") {
    py -3 main.py fetch-minishare-mins --codes $Codes --freq $Freq --start $Start --end $End @combineArg
} else {
    & $pythonPath main.py fetch-minishare-mins --codes $Codes --freq $Freq --start $Start --end $End @combineArg
}
