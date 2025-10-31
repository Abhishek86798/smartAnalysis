# 🚀 Quick Start Guide - Train BERT Model

## ✅ Prerequisites Completed

- [x] Python 3.11 installed
- [x] Virtual environment created
- [x] PyTorch with CUDA installed
- [x] GPU detected (RTX 3050)
- [x] All code files created

---

## 🎯 Ready to Train!

### **Option 1: Using Jupyter Notebook (RECOMMENDED)**

1. **Start Jupyter:**
   ```powershell
   # Make sure venv is activated
   .\venv\Scripts\Activate.ps1
   
   # Start Jupyter
   jupyter notebook
   ```

2. **Open Training Notebook:**
   - Navigate to `notebooks/03_baseline_training.ipynb`
   - Select kernel: `Python 3.11 (SmartReview)`

3. **Run All Cells:**
   - Click "Cell" → "Run All"
   - Or press `Shift+Enter` for each cell
   - Training will take ~45-60 minutes

4. **Monitor Progress:**
   - Watch the progress bars
   - Check GPU usage: Open new terminal and run `nvidia-smi -l 1`

---

### **Option 2: Test Individual Components First**

If you want to verify everything works before training:

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Test BERT model
python src/models/bert_classifier.py

# Test dataset
python src/utils/dataset.py

# Test metrics
python src/utils/metrics.py
```

**Expected Output:**
```
✅ Model test passed!
✅ Dataset test passed!
✅ Metrics module test passed!
```

---

## 📊 What to Expect During Training

### **Epoch 1 (~15-20 minutes):**
```
Epoch 1/3: 100%|████████| 5374/5374 [18:23<00:00]
   Training Loss: ~0.45
   Training Acc: ~0.82
   Val Loss: ~0.41
   Val Acc: ~0.84
   Val F1: ~0.78
```

### **Epoch 2 (~15-20 minutes):**
```
Epoch 2/3: 100%|████████| 5374/5374 [18:15<00:00]
   Training Loss: ~0.35
   Training Acc: ~0.87
   Val Loss: ~0.38
   Val Acc: ~0.86
   Val F1: ~0.81
```

### **Epoch 3 (~15-20 minutes):**
```
Epoch 3/3: 100%|████████| 5374/5374 [18:20<00:00]
   Training Loss: ~0.28
   Training Acc: ~0.90
   Val Loss: ~0.37
   Val Acc: ~0.87
   Val F1: ~0.82
```

### **Final Test Results:**
```
📊 Test Set Metrics:
   Accuracy: 0.8450
   Macro F1: 0.7980
   
   Per-Class F1:
   Positive: 0.8850
   Negative: 0.7890
   Neutral: 0.7200
```

---

## 📁 Files Created After Training

```
smartReview/
├── models/baseline/
│   ├── checkpoints/
│   │   ├── best_model.pt          ← Use this for deployment
│   │   ├── epoch_1.pt
│   │   ├── epoch_2.pt
│   │   └── epoch_3.pt
│   ├── logs/
│   │   └── training_history.json
│   └── results/
│       ├── test_results.json
│       ├── confusion_matrix.png
│       ├── per_class_f1.png
│       └── classification_report.txt
└── outputs/figures/training/
    └── training_curves.png
```

---

## ⚠️ Troubleshooting

### **Issue: CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```

**Solution:** Open `notebooks/03_baseline_training.ipynb`, find cell 4 and change:
```python
'batch_size': 8,      # Change to 4
'max_length': 256,    # Change to 128
```

---

### **Issue: Slow Training (Using CPU)**
```
Device: cpu
```

**Solution:** Make sure CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

If False, reinstall PyTorch with CUDA:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### **Issue: Module Not Found**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:** Install missing package:
```powershell
pip install transformers
```

---

## 🎯 Success Criteria

Your training is successful if you see:

✅ **Training completes without errors**  
✅ **Overall accuracy ≥ 80%**  
✅ **Macro F1 ≥ 0.75**  
✅ **All 3 sentiment classes have F1 ≥ 0.60**  
✅ **Best model saved to checkpoints/**  
✅ **All visualizations generated**

---

## 📊 GPU Monitoring

Open a **separate PowerShell** terminal:

```powershell
# Watch GPU usage in real-time
nvidia-smi -l 1
```

**Expected GPU Usage During Training:**
- **GPU Memory:** 2.5-3.5 GB / 4.0 GB
- **GPU Utilization:** 80-95%
- **Power Draw:** 40-60W / 80W
- **Temperature:** 60-75°C

---

## 🚀 After Training Completes

1. **Check Results:**
   - Open `models/baseline/results/test_results.json`
   - View confusion matrix: `models/baseline/results/confusion_matrix.png`
   - Read classification report: `models/baseline/results/classification_report.txt`

2. **Analyze Performance:**
   - Which class performed best? (Usually Positive)
   - Which class needs improvement? (Usually Neutral)
   - Are there common misclassifications?

3. **Next Steps:**
   - Proceed to **Stage 2: Enhanced RoBERTa**
   - Implement continued pretraining
   - Compare BERT vs RoBERTa performance

---

## 💡 Pro Tips

1. **During Training:**
   - Don't close the notebook
   - Don't put computer to sleep
   - Keep GPU well-ventilated

2. **Save Often:**
   - Checkpoints are saved automatically
   - Training history saved after completion

3. **Monitor Progress:**
   - Watch progress bars
   - Check `nvidia-smi` for GPU usage
   - Loss should decrease, accuracy should increase

---

## 📞 Need Help?

If you encounter issues:
1. Check error messages carefully
2. Verify virtual environment is activated
3. Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check file paths are correct
5. Review troubleshooting section above

---

## ✅ Ready?

**Start training now:**
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Start Jupyter
jupyter notebook

# Open: notebooks/03_baseline_training.ipynb
# Click: Cell → Run All
# Wait: ~45-60 minutes
# Celebrate: When you see "Phase 4 Complete!" 🎉
```

**Good luck!** 🚀
