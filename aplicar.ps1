# Aplica as reconstruções aprovadas e refaz os PDFs (limpo e papel).
# Uso:  .\aplicar.ps1 serpiente    ou    .\aplicar.ps1 tren    ou    .\aplicar.ps1 vida
param([Parameter(Mandatory=$true)][ValidateSet('serpiente','tren','vida','berkeley')][string]$livro)
$raiz = Split-Path -Parent $PSScriptRoot
$pastas = @{ serpiente = 'La Serpiente - limpo'; tren = 'Invención del Tren Fantasma - limpo'; vida = 'La Vida Nueva - limpo'; berkeley = 'Berkeley em Bellagio - limpo' }
$titulos = @{ serpiente = 'La Serpiente'; tren = 'Invención del Tren Fantasma'; vida = 'La Vida Nueva'; berkeley = 'Berkeley em Bellagio' }
$pasta = Join-Path $raiz $pastas[$livro]; $titulo = $titulos[$livro]
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
py -I (Join-Path $PSScriptRoot 'aplicar.py') $pasta
if ($?) { py -I (Join-Path $PSScriptRoot 'pdfs_livro.py') $PSScriptRoot $pasta $titulo }
