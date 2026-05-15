param(
    [int]$Port = 8501
)

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
    throw "Python was not found. Install Python 3.11+ or add it to PATH."
}

if ($python -and $python.Name -eq "py.exe") {
    py -3 -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port $Port
} else {
    & $pythonPath -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port $Port
}
