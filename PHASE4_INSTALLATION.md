# 🚀 Phase 4: Step-by-Step Installation & Setup

## Step 1: Install Required Libraries

Open your terminal in the project directory and activate your virtual environment:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch (CPU version - works on all machines)
pip install torch torchvision torchaudio

# Install Transformers library
pip install transformers

# Install additional dependencies
pip install scikit-learn tensorboard pyyaml

# Verify installations
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import transformers; print(f'Transformers {transformers.__version__} installed')"
python -c "print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch 2.x.x installed
Transformers 4.x.x installed
CUDA available: True/False
```

---

## Step 2: Test Model Creation

Run this to verify the BERT classifier works:

```powershell
python src/models/bert_classifier.py
```

**Expected Output:**
```
Loading pretrained BERT model: bert-base-uncased
✅ Model initialized:
   • BERT model: bert-base-uncased
   • Hidden size: 768
   • Output classes: 3
   • Dropout: 0.1
   • Total parameters: 109,483,779
   • Trainable parameters: 109,483,779
📍 Model loaded on device: cuda (or cpu)

Testing forward pass:
   Input shape: torch.Size([4, 128])
   Output shape: torch.Size([4, 3])
   Expected: (4, 3)
   
   Sample predictions (probabilities):
   tensor([0.3421, 0.3298, 0.3281])
   Sum: 1.0000 (should be 1.0)

✅ Model test passed!
```

---

## Step 3: Folder Structure Verification

Run this command to see the created structure:

```powershell
tree /F src models
```

**Expected Structure:**
```
src
├── data
│   └── preprocessor.py
├── models
│   ├── __init__.py
│   ├── bert_classifier.py    ← Just created!
│   └── trainer.py             ← Next to create
└── utils
    ├── __init__.py
    ├── dataset.py             ← Next to create
    └── metrics.py             ← Next to create

models
└── baseline
    ├── checkpoints
    ├── logs
    └── results
```

---

## 📝 Files Created So Far

✅ `PHASE4_BASELINE_GUIDE.md` - Comprehensive guide  
✅ `src/models/__init__.py` - Models module init  
✅ `src/models/bert_classifier.py` - **BERT model (300+ lines)**  
✅ `src/utils/__init__.py` - Utils module init  
✅ Directory structure for checkpoints, logs, results  

---

## 🎯 Next Steps

After installation completes, we'll create:

1. ✅ **Dataset class** (`src/utils/dataset.py`)
   - Tokenizes reviews
   - Creates PyTorch Dataset
   - Handles batching

2. ✅ **Metrics module** (`src/utils/metrics.py`)
   - Accuracy, F1, Precision, Recall
   - Confusion matrix plotting
   - Classification reports

3. ✅ **Trainer class** (`src/models/trainer.py`)
   - Training loop with class weights
   - Validation
   - Checkpoint saving

4. ✅ **Training notebook** (`notebooks/03_baseline_training.ipynb`)
   - End-to-end training pipeline
   - Visualizations
   - Model evaluation

---

## ⚠️ Troubleshooting

### Issue: "pip install torch" is slow
**Solution:** Use CPU version for faster install:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "CUDA not available" but I have GPU
**Solution:** Install CUDA-enabled PyTorch:
```powershell
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA 11.8 (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Import torch could not be resolved" in VS Code
**Solution:** 
1. Make sure virtual environment is activated
2. Reload VS Code window (Ctrl+Shift+P → "Reload Window")
3. Select correct Python interpreter (Ctrl+Shift+P → "Select Interpreter")

---

## ✅ Ready to Continue?

Once you've successfully installed the libraries and tested the model, let me know and we'll proceed with:
- **Dataset class** (Step 3)
- **Metrics module** (Step 5)
- **Trainer class** (Step 4)

Type "continue" or "next" when ready! 🚀
