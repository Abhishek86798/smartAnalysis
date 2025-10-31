# 🔧 Setting Up Python 3.11 Environment for PyTorch

## Current Situation
- ✅ You have **NVIDIA RTX 3050 Laptop GPU** with CUDA 12.9
- ❌ Current Python version: **3.13.8** (PyTorch not yet supported)
- ✅ Solution: Create new venv with **Python 3.11**

---

## 📋 Step-by-Step Setup

### **Step 1: Download Python 3.11**

1. Go to: https://www.python.org/downloads/
2. Download **Python 3.11.9** (latest 3.11 version)
3. During installation:
   - ✅ Check "Add Python 3.11 to PATH"
   - ✅ Select "Install for all users" (optional)
   - Install location: `C:\Python311\` (note this path)

### **Step 2: Verify Python 3.11 Installation**

Open a **NEW** PowerShell terminal:

```powershell
# Check Python 3.11 is installed
py -3.11 --version
# Expected: Python 3.11.9
```

---

### **Step 3: Backup Current Environment (Optional)**

```powershell
# Navigate to project
cd D:\CODES\BEproject\smartReview

# Deactivate current venv if active
deactivate

# Rename old venv (backup)
Rename-Item -Path "venv" -NewName "venv_python313_backup"
```

---

### **Step 4: Create New Virtual Environment with Python 3.11**

```powershell
# Create new venv with Python 3.11
py -3.11 -m venv venv

# Activate new environment
.\venv\Scripts\Activate.ps1

# Verify correct Python version
python --version
# Expected: Python 3.11.9

# Upgrade pip
python -m pip install --upgrade pip
```

---

### **Step 5: Reinstall Project Dependencies**

```powershell
# Install all previously installed packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install jupyter notebook ipykernel wordcloud

# Register kernel for Jupyter
python -m ipykernel install --user --name=smartreview --display-name="Python 3.11 (SmartReview)"
```

---

### **Step 6: Install PyTorch with CUDA 12.1 Support**

```powershell
# Install PyTorch with CUDA 12.1 (compatible with your CUDA 12.9)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

### **Step 7: Install Remaining ML Libraries**

```powershell
# Install Transformers and other ML tools
pip install transformers
pip install tensorboard
pip install pyyaml

# Verify Transformers
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

---

### **Step 8: Test BERT Model**

```powershell
# Test the BERT classifier we created
python src/models/bert_classifier.py
```

**Expected Output:**
```
Loading pretrained BERT model: bert-base-uncased
Downloading model files... (first time only)
✅ Model initialized:
   • BERT model: bert-base-uncased
   • Hidden size: 768
   • Output classes: 3
   • Dropout: 0.1
   • Total parameters: 109,483,779
   • Trainable parameters: 109,483,779
📍 Model loaded on device: cuda  ← Should say CUDA!

Testing forward pass:
   Input shape: torch.Size([4, 128])
   Output shape: torch.Size([4, 3])
   
✅ Model test passed!
```

---

### **Step 9: Update Jupyter Notebook Kernel**

```powershell
# Open Jupyter
jupyter notebook

# In Jupyter:
# 1. Open any notebook (e.g., 01_eda.ipynb)
# 2. Click "Kernel" → "Change Kernel" → "Python 3.11 (SmartReview)"
# 3. Verify by running: import sys; print(sys.version)
```

---

## 📦 Complete Installation Command List

Copy-paste this entire block (after installing Python 3.11):

```powershell
# Navigate to project
cd D:\CODES\BEproject\smartReview

# Deactivate old venv
deactivate

# Backup old venv (optional)
Rename-Item -Path "venv" -NewName "venv_python313_backup"

# Create new venv with Python 3.11
py -3.11 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Verify
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install data science libraries
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter notebook ipykernel wordcloud

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML libraries
pip install transformers tensorboard pyyaml

# Register Jupyter kernel
python -m ipykernel install --user --name=smartreview --display-name="Python 3.11 (SmartReview)"

# Test installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test BERT model
python src/models/bert_classifier.py
```

---

## 🎯 GPU Training Benefits

With your **RTX 3050 Laptop GPU** (4GB VRAM):

### Before (CPU):
- ❌ Training time: ~3-4 hours per epoch
- ❌ Total: ~9-12 hours for 3 epochs
- ❌ Very slow iteration

### After (GPU):
- ✅ Training time: ~15-20 minutes per epoch
- ✅ Total: ~45-60 minutes for 3 epochs
- ✅ Fast experimentation
- ✅ **20x faster!**

### Recommended Settings for 4GB GPU:
```yaml
batch_size: 8          # Reduce from 16 to fit in 4GB
max_length: 256        # Reduce from 512 to save memory
gradient_accumulation: 2  # Effective batch size = 16
```

---

## ⚠️ Important Notes

### Memory Management (4GB GPU)
Your RTX 3050 has 4GB VRAM. To avoid "CUDA out of memory":

1. **Close unnecessary programs** (especially browsers)
2. **Use batch_size=8** instead of 16
3. **Monitor GPU memory:**
   ```powershell
   # Watch GPU usage during training
   nvidia-smi -l 1
   ```

### First Model Download
The first time you run BERT, it will download ~500MB of model files:
```
Downloading bert-base-uncased...
  config.json: 100%
  pytorch_model.bin: 100% 440MB
  tokenizer_config.json: 100%
```

This is normal and only happens once!

---

## ✅ Verification Checklist

Before proceeding to training:

- [ ] Python 3.11.9 installed
- [ ] New venv created and activated
- [ ] `python --version` shows 3.11.x
- [ ] PyTorch installed: `import torch` works
- [ ] CUDA available: `torch.cuda.is_available()` returns `True`
- [ ] GPU detected: `torch.cuda.get_device_name(0)` shows RTX 3050
- [ ] Transformers installed: `import transformers` works
- [ ] BERT model test passes: `python src/models/bert_classifier.py`
- [ ] All previous packages reinstalled (pandas, numpy, etc.)

---

## 🚀 After Setup Complete

Once all checks pass, let me know and I'll create:

1. ✅ **Dataset class** - Tokenizes reviews for BERT
2. ✅ **Metrics module** - Evaluation metrics
3. ✅ **Trainer class** - GPU-optimized training loop
4. ✅ **Training config** - Optimized for RTX 3050 (4GB)
5. ✅ **Training notebook** - Interactive training

**Estimated setup time:** 20-30 minutes
**First training run:** 45-60 minutes (with GPU!) 🔥

---

## 💡 Pro Tips

1. **Monitor GPU during training:**
   ```powershell
   # In a separate terminal
   nvidia-smi -l 1
   ```

2. **If you get OOM (Out of Memory):**
   - Reduce `batch_size` from 8 to 4
   - Reduce `max_length` from 256 to 128
   - Close other GPU programs

3. **Training speed:**
   - RTX 3050: ~15-20 min/epoch
   - ~250-300 samples/second
   - Much faster than CPU!

---

**Ready? Start with Step 1: Download Python 3.11!** 🎓
