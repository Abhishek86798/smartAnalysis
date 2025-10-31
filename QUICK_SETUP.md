# ⚡ Quick Setup Commands - Python 3.11 + PyTorch GPU

## 🚀 Copy-Paste This (After installing Python 3.11)

```powershell
# 1. Navigate to project
cd D:\CODES\BEproject\smartReview

# 2. Deactivate current environment
deactivate

# 3. Backup old venv (optional)
Rename-Item -Path "venv" -NewName "venv_backup"

# 4. Create new Python 3.11 venv
py -3.11 -m venv venv

# 5. Activate
.\venv\Scripts\Activate.ps1

# 6. Upgrade pip
python -m pip install --upgrade pip

# 7. Install everything at once
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter notebook ipykernel wordcloud torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 transformers tensorboard pyyaml

# 8. Register Jupyter kernel
python -m ipykernel install --user --name=smartreview --display-name="Python 3.11 (SmartReview)"

# 9. Test CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 10. Test BERT model
python src/models/bert_classifier.py
```

---

## ✅ Success Indicators

You should see:
```
✅ CUDA Available: True
✅ Model loaded on device: cuda
✅ Model test passed!
```

---

## 🎯 Next Steps After Setup

Type **"setup complete"** or **"continue"** and I'll create:
1. Dataset class (tokenization)
2. Metrics module (evaluation)
3. Trainer class (GPU training)
4. Training notebook (end-to-end)

**Total setup time:** ~20-30 minutes
**First training:** ~45-60 minutes with your RTX 3050! 🚀
