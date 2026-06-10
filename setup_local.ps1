# Setup automático del entorno local (Windows + GPU NVIDIA).
# Uso:  click derecho > "Ejecutar con PowerShell"   o   .\setup_local.ps1
# Crea un entorno aislado con Python 3.12 y PyTorch CUDA, sin tocar tu Python global.

$ErrorActionPreference = "Stop"
Write-Host "== 1/4  Instalando uv (gestor de entornos) ==" -ForegroundColor Cyan
python -m pip install --quiet uv

Write-Host "== 2/4  Instalando Python 3.12 (aislado) ==" -ForegroundColor Cyan
python -m uv python install 3.12

Write-Host "== 3/4  Creando entorno .venv ==" -ForegroundColor Cyan
python -m uv venv --python 3.12 .venv

Write-Host "== 4/4  Instalando PyTorch CUDA + dependencias (~2.5 GB, tarda) ==" -ForegroundColor Cyan
python -m uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m uv pip install --python .venv matplotlib scikit-learn torchinfo

Write-Host ""
Write-Host "Listo. Verifica la GPU con:" -ForegroundColor Green
Write-Host "  .venv\Scripts\python.exe -c `"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))`""
Write-Host "Y entrena con:" -ForegroundColor Green
Write-Host "  .venv\Scripts\python.exe train_local.py"
