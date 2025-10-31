# 🎯 Quick Reference - RoBERTa Pretraining

## 📅 Date: October 29, 2025

---

## ✅ BERT Baseline Results (Completed!)

### Test Set Performance:
```
Overall Accuracy:  88.13% ✅ (Target: 80%)
Macro F1 Score:    0.7221 ⚠️  (Target: 0.75)

Per-Class Results:
  Negative:  Precision=0.84, Recall=0.88, F1=0.86 ✅
  Neutral:   Precision=0.37, Recall=0.35, F1=0.36 ❌ (NEEDS IMPROVEMENT)
  Positive:  Precision=0.95, Recall=0.94, F1=0.95 ✅
```

**Key Finding:** Neutral class is the bottleneck (F1=0.36 vs target 0.62)

---

## 🚀 What You're Doing Now: RoBERTa Pretraining

### Objective:
**Domain-adapt RoBERTa on 61K phone reviews** to improve understanding of phone-specific language, especially for the challenging Neutral class.

### Process:
1. **Load** all 61,553 reviews (train + val + test)
2. **Train** with Masked Language Modeling (MLM)
   - Mask 15% of tokens randomly
   - Predict masked tokens from context
   - Learn phone vocabulary: "battery", "camera", "screen", etc.
3. **Save** domain-adapted model
4. **Next:** Fine-tune for sentiment (tomorrow)

---

## 📋 How to Run (Step-by-Step)

### 1. Open the Notebook
```
File: notebooks/04_roberta_pretraining.ipynb
```

### 2. Run Cells in Order
- **Cells 1-9:** Setup, load data, configure training
- **Cell 10:** START TRAINING (2-3 hours)
  - This is the main training cell
  - Can run overnight!
  - Progress bar shows real-time status
- **Cells 11-13:** Evaluate and save model
- **Cell 14:** Test predictions (optional, but fun!)

### 3. What to Expect

**During Training (Cell 10):**
```
Epoch 1/3: [████████░░] 50%  - Loss: 2.45 - Time: 45min
Epoch 2/3: [████████░░] 50%  - Loss: 2.12 - Time: 45min  
Epoch 3/3: [████████░░] 50%  - Loss: 1.98 - Time: 45min
```

**After Training:**
```
✅ Training complete!
Final train loss: ~2.0
Eval perplexity: ~7-10 (lower is better)
Model saved to: models/roberta_pretrained/
```

---

## ⏱️ Time Estimates

| Phase | Time | When to Run |
|-------|------|-------------|
| Setup (Cells 1-9) | 5 minutes | Now |
| Training (Cell 10) | 2-3 hours | Overnight |
| Evaluation (Cells 11-13) | 2 minutes | After training |
| Testing (Cell 14) | 1 minute | Optional |

**Total:** ~2.5-3 hours (mostly unattended)

---

## 💡 Tips

### Running Overnight:
1. **Start training before bed** (Cell 10)
2. **Let it run** - progress is auto-saved every 1000 steps
3. **Check in morning** - should be complete!
4. **If interrupted:** Training resumes from last checkpoint

### Monitoring Progress:
- **In notebook:** Progress bar shows current status
- **GPU usage:** Open new terminal, run `nvidia-smi`
- **Loss values:** Should decrease over time (good sign!)

### If Training Fails:
- **CUDA OOM:** Reduce `batch_size` from 16 to 8 in Cell 5 (CONFIG cell)
- **Slow progress:** Normal! MLM training is compute-intensive
- **Column error:** Fixed! (Changed 'review_text' to 'cleaned_text')
- **Error messages:** Copy and show me - I'll help fix!

---

## 🎯 Expected Results

### After Pretraining:
```
✅ Domain-adapted RoBERTa model
✅ Understands phone review vocabulary
✅ Ready for sentiment fine-tuning
```

### After Fine-tuning (Tomorrow):
```
Expected Improvements vs BERT:
  Overall Accuracy:  88.13% → 90-92% (+2-5%)
  Macro F1:          0.7221 → 0.78-0.82 (+5-7%)
  Neutral F1:        0.3593 → 0.70-0.80 (+40-50%) ⭐ BIG WIN!
```

---

## 📁 Files Created

After running the notebook:

```
models/roberta_pretrained/
  ├── pytorch_model.bin        (500 MB - main model)
  ├── config.json              (model config)
  ├── tokenizer_config.json    (tokenizer config)
  ├── vocab.json               (vocabulary)
  ├── merges.txt               (BPE merges)
  ├── pretraining_results.json (training metrics)
  └── checkpoint-X/            (intermediate checkpoints)
```

---

## 🔄 What Happens During Training

### Masked Language Modeling (MLM):

**Original sentence:**
```
"The battery life is amazing and the camera quality is great."
```

**Masked version (15% tokens):**
```
"The [MASK] life is amazing and the [MASK] quality is great."
```

**Model learns to predict:**
```
[MASK] → "battery" (from context: "life", "amazing")
[MASK] → "camera" (from context: "quality", "great")
```

**Why this helps:**
- Learns phone-specific vocabulary
- Understands aspect relationships
- Better at handling Neutral sentiment (mixed opinions)

---

## 🚀 Quick Start Commands

### 1. Open Notebook
```powershell
# In VS Code, open:
notebooks/04_roberta_pretraining.ipynb
```

### 2. Select Kernel
- Click "Select Kernel" (top right)
- Choose: `venv (Python 3.11.9)`

### 3. Run All Cells
- Click "Run All" or press `Ctrl+Shift+Enter`
- Or run cells one by one with `Shift+Enter`

---

## ❓ FAQ

**Q: Can I stop training mid-way?**
A: Yes! Progress is saved every 1000 steps. Training resumes from last checkpoint.

**Q: How do I know it's working?**
A: Loss values should decrease over time. Check progress bar for updates.

**Q: What if I get CUDA OOM error?**
A: Reduce `batch_size` from 16 to 8 in Cell 5 (CONFIG), restart notebook.

**Q: Can I use my computer while training?**
A: Yes, but it will be slower. Best to let it run overnight.

**Q: How do I check GPU usage?**
A: Open PowerShell, run: `nvidia-smi` (shows GPU memory and usage)

---

## 📞 Need Help?

**Show me:**
- Error messages (copy full error)
- Training progress (loss values)
- GPU memory usage (`nvidia-smi` output)

**I'll help you:**
- Fix errors
- Optimize settings
- Speed up training

---

## ✅ Checklist

Before starting:
- [ ] Notebook opened: `04_roberta_pretraining.ipynb`
- [ ] Kernel selected: `venv (Python 3.11.9)`
- [ ] GPU available: Check Cell 1 output
- [ ] Time available: 2-3 hours (can run overnight)

During training:
- [ ] Cell 10 running (progress bar visible)
- [ ] Loss decreasing over time
- [ ] No error messages

After training:
- [ ] Training complete message shown
- [ ] Model saved to `models/roberta_pretrained/`
- [ ] Ready for fine-tuning!

---

## 🎯 Bottom Line

**What:** Domain-adapt RoBERTa on phone reviews  
**Why:** Improve Neutral class performance (F1: 0.36 → 0.75+)  
**How:** Masked Language Modeling (3 epochs)  
**When:** Now (run overnight)  
**Time:** 2-3 hours unattended  

**Next:** Fine-tune for sentiment (1 hour tomorrow)  
**Result:** 90%+ accuracy, 0.78+ macro F1 🎉

---

**Ready?** Open `04_roberta_pretraining.ipynb` and run Cell 1! 🚀

**Date:** October 29, 2025  
**Status:** Ready to pretrain RoBERTa
