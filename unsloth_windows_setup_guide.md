# Setting Up Unsloth on Windows 11 (Native Environment)

This guide provides the clean, verified steps required to set up a fully working Unsloth development environment natively on Windows 11 using Miniforge/Conda and modern PyTorch/CUDA builds.

## Prerequisites
* **Hardware:** An NVIDIA GPU (tested on a GTX 1650 Ti with 4GB VRAM).
* **Drivers:** Latest NVIDIA Studio or Game Ready drivers supporting CUDA.
* **Package Manager:** [Miniforge](https://github.com/conda-forge/miniforge) installed for Windows.

---

## Step-by-Step Installation

Open your **Miniforge Prompt** and execute the following commands in order:

### 1. Remove Any Old/Broken Environments (Optional)
If you have a previous failed installation attempt, clear it out first:
```cmd
conda deactivate
conda env remove --name unsloth_win -y
```

### 2. Create a Clean Conda Environment
Create a fresh environment targeting Python 3.10:
```cmd
conda create --name unsloth_win python=3.10 -y
conda activate unsloth_win
```

### 3. Install Unsloth & Dependencies via Git
Pull down the latest Unsloth release framework directly:
```cmd
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 4. Align PyTorch with a Compatible CUDA Index
To satisfy internal requirement dependencies (such as `torchao` requiring `torch.int1`), force-upgrade PyTorch using a modern CUDA 12.x release channel (`cu126`):
```cmd
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Verification

Test your installation to verify that PyTorch detects your GPU correctly and that Unsloth imports without attribute errors:

```cmd
python -c "import torch; import unsloth; print('Torch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Expected Output:
```text
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Torch version: X.X.X+cu126
CUDA Available: True
GPU: NVIDIA GeForce GTX 1650 Ti