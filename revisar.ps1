# Revisão das reconstruções propostas.
# Uso:  .\revisar.ps1 serpiente    ou    .\revisar.ps1 tren    ou    .\revisar.ps1 vida
param([Parameter(Mandatory=$true)][ValidateSet('serpiente','tren','vida')][string]$livro)
$raiz = Split-Path -Parent $PSScriptRoot
$pastas = @{ serpiente = 'La Serpiente - limpo'; tren = 'Invención del Tren Fantasma - limpo'; vida = 'La Vida Nueva - limpo' }
$pasta = Join-Path $raiz $pastas[$livro]
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
py -I (Join-Path $PSScriptRoot 'revisar.py') $pasta
