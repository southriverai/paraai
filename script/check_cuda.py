"""
CUDA Debugging Script
This script helps diagnose CUDA/PyTorch GPU issues.
"""

import subprocess
import sys

print("=" * 60)
print("CUDA/PyTorch GPU Debugging Script")
print("=" * 60)
print()

# 1. Check Python version
print("1. Python Version:")
print(f"   Python {sys.version}")
print()

# 2. Check PyTorch installation
print("2. PyTorch Installation:")
try:
    import torch

    print(f"   [OK] PyTorch version: {torch.__version__}")
except ImportError:
    print("   [ERROR] PyTorch is not installed!")
    print("   Please activate your virtual environment and install PyTorch:")
    print("   pip install torch")
    sys.exit(1)
except OSError as e:
    print("   [ERROR] PyTorch import failed with DLL/library loading error:")
    print(f"   {e}")
    print()
    print("   This usually means:")
    print("   1. PyTorch was installed with CUDA support but CUDA libraries are missing")
    print("   2. Or there's a corrupted PyTorch installation")
    print()
    print("   Solutions:")
    print("   - Try reinstalling PyTorch: pip uninstall torch torchvision torchaudio")
    print("   - Install CPU-only version: pip install torch torchvision torchaudio")
    print("   - Or install CUDA version with proper CUDA runtime")
    sys.exit(1)
except Exception as e:
    print(f"   [ERROR] Unexpected error importing PyTorch: {e}")
    sys.exit(1)
print()

# 3. Check CUDA availability in PyTorch
print("3. PyTorch CUDA Support:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version (PyTorch): {torch.version.cuda if torch.version.cuda else 'N/A'}")
print(f"   cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print()

# 4. Check GPU devices
print("4. GPU Devices:")
if torch.cuda.is_available():
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      - Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"      - Compute Capability: {props.major}.{props.minor}")
        print(f"      - Multiprocessors: {props.multi_processor_count}")
else:
    print("   [ERROR] No CUDA devices detected")
print()

# 5. Check CUDA driver (via nvidia-smi)
print("5. NVIDIA Driver (via nvidia-smi):")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode == 0:
        print("   [OK] nvidia-smi output:")
        for line in result.stdout.strip().split("\n"):
            print(f"      {line}")
    else:
        print("   [ERROR] nvidia-smi failed or not found")
except FileNotFoundError:
    print("   [ERROR] nvidia-smi not found in PATH")
    print("      This usually means NVIDIA drivers are not installed")
except subprocess.TimeoutExpired:
    print("   ✗ nvidia-smi timed out")
except Exception as e:
    print(f"   ✗ Error running nvidia-smi: {e}")
print()

# 6. Check CUDA toolkit version (if available)
print("6. CUDA Toolkit Version:")
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5, check=False)
    if result.returncode == 0:
        # Extract version from nvcc output
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                print(f"   [OK] {line.strip()}")
                break
    else:
        print("   [WARNING] nvcc not found (CUDA toolkit may not be installed)")
        print("      Note: PyTorch includes CUDA, so this is often optional")
except FileNotFoundError:
    print("   [WARNING] nvcc not found in PATH")
    print("      Note: PyTorch includes its own CUDA, so this is usually fine")
except Exception as e:
    print(f"   [WARNING] Could not check CUDA toolkit: {e}")
print()

# 7. Test CUDA operations
print("7. CUDA Operation Test:")
if torch.cuda.is_available():
    try:
        # Create a test tensor
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y  # Matrix multiplication
        print("   [OK] CUDA operations working correctly")
        print(f"   [OK] Test tensor created on: {x.device}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   [ERROR] CUDA operations failed: {e}")
else:
    print("   [WARNING] Skipping CUDA test (CUDA not available)")
print()

# 8. Check PyTorch build information
print("8. PyTorch Build Information:")
try:
    print(f"   Built with CUDA: {torch.version.cuda is not None}")
    if torch.version.cuda:
        print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"   cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.enabled else 'N/A'}")

    # Try to get more info
    if hasattr(torch, "show_config"):
        print("\n   Detailed PyTorch config:")
        torch.show_config()
except Exception as e:
    print(f"   Could not get build info: {e}")
print()

# 9. Recommendations
print("9. Recommendations:")
if not torch.cuda.is_available():
    print("   [ERROR] CUDA is not available. Possible causes:")
    print()
    print("   1. NVIDIA GPU drivers not installed:")
    print("      - Download from: https://www.nvidia.com/drivers")
    print("      - Install the latest driver for your GPU")
    print()
    print("   2. PyTorch installed without CUDA support:")
    print("      - Current PyTorch: CPU-only version")
    print("      - Solution: Install PyTorch with CUDA")
    print("      - Visit: https://pytorch.org/get-started/locally/")
    print("      - Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("   3. CUDA version mismatch:")
    if torch.version.cuda:
        print(f"      - PyTorch built with CUDA: {torch.version.cuda}")
        print("      - Check if your driver supports this CUDA version")
    else:
        print("      - PyTorch was built without CUDA support")
    print()
    print("   4. GPU not detected by system:")
    print("      - Check Device Manager (Windows) for GPU")
    print("      - Verify GPU is enabled in BIOS")
    print("      - Try running 'nvidia-smi' in terminal")
else:
    print("   [OK] CUDA is available and working!")
    print("   [OK] You can use GPU-accelerated models")
print()

# 10. Quick fix suggestions
print("10. Quick Fix Commands:")
print("   To install PyTorch with CUDA 11.8:")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print()
print("   To install PyTorch with CUDA 12.1:")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print()
print("   To check what CUDA versions are available:")
print("   Visit: https://pytorch.org/get-started/locally/")
print()

print("=" * 60)
