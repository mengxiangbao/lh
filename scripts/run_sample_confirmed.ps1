$ErrorActionPreference = "Stop"

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
    throw "Python was not found. Install Python 3.11+ or add it to PATH, then rerun this script."
}

if ($python -and $python.Name -eq "py.exe") {
    py -3 main.py generate-sample --out data/raw
    py -3 main.py check-data --data data/raw/daily_price.csv --out data/data_check
    py -3 main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
} else {
    & $pythonPath main.py generate-sample --out data/raw
    & $pythonPath main.py check-data --data data/raw/daily_price.csv --out data/data_check
    & $pythonPath main.py backtest --config config/default.toml --data data/raw/daily_price.csv --out data/backtest_result/confirmed --mode confirmed
}
