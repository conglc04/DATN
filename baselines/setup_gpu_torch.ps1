# setup_gpu_torch.ps1
# Chạy sau khi update NVIDIA driver + restart máy.
# Mục đích: thay torch CPU bằng torch CUDA (GPU), verify hoạt động.
#
# Cách dùng:
#   1. Mở PowerShell tại F:\Do an\baselines
#   2. Set-ExecutionPolicy -Scope Process Bypass     (nếu lần đầu)
#   3. .\setup_gpu_torch.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = "F:\Do an\baselines"
$VenvPython  = "$ProjectRoot\.venv\Scripts\python.exe"

Write-Host "============================================================"
Write-Host "  PPO  —  GPU torch setup"
Write-Host "============================================================"

# --- Step 1: kiểm tra venv ---
if (-not (Test-Path $VenvPython)) {
    Write-Error "Khong tim thay venv tai $VenvPython. Chay 'python -m venv .venv' truoc."
    exit 1
}

# --- Step 2: kiểm tra NVIDIA driver ---
Write-Host ""
Write-Host "[1/5] Kiem tra NVIDIA driver..."
$smi = nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvidia-smi loi — driver chua duoc cai dat hoac chua restart."
    exit 1
}
$driverVer = ($smi -split ',')[0].Trim()
$gpuName   = ($smi -split ',')[1].Trim()
Write-Host "    GPU:    $gpuName"
Write-Host "    Driver: $driverVer"

# Driver 555+ moi co the dung CUDA 12.6 torch
$driverMajor = [int]($driverVer -split '\.')[0]
if ($driverMajor -lt 555) {
    Write-Warning "Driver $driverVer < 555 — torch CUDA 12.6 co the loi."
    Write-Warning "Update driver tai https://www.nvidia.com/Download/index.aspx"
    Write-Host ""
    $resp = Read-Host "Tiep tuc voi cu118 (CUDA 11.8, ho tro driver 452+) thay vi cu126? [Y/n]"
    if ($resp -eq "" -or $resp -eq "Y" -or $resp -eq "y") {
        $torchIndex = "https://download.pytorch.org/whl/cu118"
        $torchPkg   = "torch>=2.4,<2.7"
        Write-Host "    -> Se cai $torchPkg + $torchIndex"
    } else {
        Write-Error "Dung lai. Update driver roi chay lai script."
        exit 1
    }
} else {
    $torchIndex = "https://download.pytorch.org/whl/cu126"
    $torchPkg   = "torch"
    Write-Host "    -> Driver OK, se cai torch moi nhat voi cu126"
}

# --- Step 3: uninstall CPU torch ---
Write-Host ""
Write-Host "[2/5] Go bo torch CPU hien tai..."
& $VenvPython -m pip uninstall torch -y

# --- Step 4: install GPU torch ---
Write-Host ""
Write-Host "[3/5] Cai torch GPU (~700MB - 2GB, can vai phut)..."
& $VenvPython -m pip install --upgrade "$torchPkg" --index-url $torchIndex

# --- Step 5: verify ---
Write-Host ""
Write-Host "[4/5] Verify torch.cuda..."
$env:PYTHONIOENCODING = "utf-8"
& $VenvPython -c @"
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print('Tensor op on GPU: OK, shape:', tuple(y.shape))
else:
    print('CUDA KHONG kha dung — verify lai driver hoac torch wheel.')
    exit(1)
"@

# --- Step 6: re-run pytest to confirm nothing broke ---
Write-Host ""
Write-Host "[5/5] Chay lai pytest de chac chan khong vo gi..."
& $VenvPython -m pytest tests/ -v

Write-Host ""
Write-Host "============================================================"
Write-Host "  HOAN THANH. torch GPU san sang cho Week 6/10 training."
Write-Host "============================================================"
