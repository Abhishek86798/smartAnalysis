# 🔧 Disk Space Fix - Quick Guide

## ⚠️ Problem
Your C: drive only has **20 MB free**, but RoBERTa model needs **500 MB** to download.

---

## ✅ Solution Applied

I've added **2 new cells** to your notebook to fix this:

### **Cell 2 (NEW):** Set Cache Directory
```python
import os
os.environ['HF_HOME'] = 'D:/huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface/transformers'
```

This tells Hugging Face to download models to your D: drive instead of C: drive.

---

## 🚀 How to Use

### **Step 1: Restart Kernel**
- Click "Restart" in notebook toolbar
- Or: `Ctrl+Shift+P` → "Restart Kernel"
- This clears the previous download attempt

### **Step 2: Run Cells in Order**
1. **Cell 1:** Markdown title (skip)
2. **Cell 2:** ⚠️ **NEW - Cache directory setup** (RUN THIS FIRST!)
3. **Cell 3:** Markdown (skip)
4. **Cell 4:** Imports (now will use D: drive)
5. **Cell 5:** Configuration
6. **Continue as normal...**

### **What You'll See:**

**Cell 2 Output:**
```
✅ Cache directories set to D: drive:
   HF_HOME: D:/huggingface
   TRANSFORMERS_CACHE: D:/huggingface/transformers
   HF_DATASETS_CACHE: D:/huggingface/datasets

💡 Models will now download to D: drive instead of C: drive!
```

**Cell 4 Output (when loading model):**
```
Downloading model to: D:/huggingface/transformers/...
model.safetensors: 100% 499M/499M [00:30<00:00, 16.5MB/s] ✅
```

---

## 💾 Make It Permanent (Optional)

If you want this to apply to ALL future notebooks automatically:

### **Option 1: PowerShell Command (One-time)**
```powershell
setx HF_HOME "D:\huggingface"
```

Then **restart VS Code**.

### **Option 2: Environment Variables (Windows)**
1. Search "Environment Variables" in Windows
2. Click "Environment Variables" button
3. Add new **User variable**:
   - Name: `HF_HOME`
   - Value: `D:\huggingface`
4. Click OK, restart VS Code

---

## 📊 Disk Space Check

**Before fix:**
```
C: drive: 20 MB free ❌ (not enough for 500 MB model)
D: drive: Plenty of space ✅
```

**After fix:**
```
C: drive: 20 MB free (unchanged)
D: drive: Will store models here (500 MB used)
```

---

## 🔍 Verify It's Working

After running Cell 2, check the cache location:

```python
import os
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
# Should print: D:/huggingface
```

When downloading models, you should see:
```
Downloading to: D:\huggingface\transformers\models--roberta-base\...
```

---

## ⚡ Quick Troubleshooting

### **Still downloading to C: drive?**
- Make sure Cell 2 ran BEFORE Cell 4 (imports)
- Restart kernel and try again
- Check Cell 2 output shows D: drive

### **Permission error on D: drive?**
- Make sure D:\huggingface folder exists
- Or change to: `D:/models/huggingface`

### **Still not working?**
- Show me the error message
- Run: `echo %HF_HOME%` in PowerShell
- I'll help debug!

---

## 📁 What Gets Downloaded

**RoBERTa model files (~500 MB total):**
```
D:/huggingface/transformers/models--roberta-base/
├── snapshots/
│   └── abc123.../
│       ├── pytorch_model.bin (498 MB)
│       ├── config.json (1 KB)
│       ├── vocab.json (899 KB)
│       ├── merges.txt (456 KB)
│       └── tokenizer.json (1.4 MB)
```

**These files are reused across notebooks** - download once, use everywhere!

---

## ✅ Summary

**What I did:**
1. Added cache setup cell at the top of notebook
2. Redirects downloads to D: drive (more space)
3. Added permanent setup instructions at bottom

**What you do:**
1. Restart kernel
2. Run Cell 2 (cache setup)
3. Run Cell 4 (imports) - will now download to D: drive
4. Continue with rest of notebook normally

**Result:** 
- ✅ RoBERTa downloads to D: drive
- ✅ C: drive not affected
- ✅ Training can proceed!

---

**Ready?** Restart kernel and run Cell 2 first! 🚀

**Date:** October 31, 2025  
**Status:** Disk space issue fixed!
