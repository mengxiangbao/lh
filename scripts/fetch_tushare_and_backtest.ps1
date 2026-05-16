$ErrorActionPreference = "Stop"

param(
    [string]$Start = "20210101",
    [string]$End = "20241231",
    [string]$Mode = "confirmed",
    [string]$DataRoot = "data"
)

if (-not $env:TUSHARE_TOKEN) {
    throw "Please set TUSHARE_TOKEN first, for example: `$env:TUSHARE_TOKEN='your-token'"
}

if (-not $env:TUSHARE_HTTP_URL) {
    $env:TUSHARE_HTTP_URL = "http://101.35.233.113:8020/"
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

$dailyPath = Join-Path $DataRoot "raw\daily_price.csv"
$checkDir = Join-Path $DataRoot "data_check"
$outDir = Join-Path $DataRoot "backtest_result\$Mode"

if ($python -and $python.Name -eq "py.exe") {
    py -3 main.py fetch-tushare --start $Start --end $End --out $dailyPath
    py -3 main.py check-data --data $dailyPath --out $checkDir
    py -3 main.py backtest --config config/default.toml --data $dailyPath --out $outDir --mode $Mode
} else {
    & $pythonPath main.py fetch-tushare --start $Start --end $End --out $dailyPath
    & $pythonPath main.py check-data --data $dailyPath --out $checkDir
    & $pythonPath main.py backtest --config config/default.toml --data $dailyPath --out $outDir --mode $Mode
}
