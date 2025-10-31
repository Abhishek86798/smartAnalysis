# 🚀 Phase 4: Baseline BERT Model Training
## Stage 1 of Hybrid Approach - Quick Baseline

**Date:** October 28, 2025  
**Approach:** Hybrid (Stage 1/3)  
**Model:** BERT-base-uncased + Classification Head  
**Goal:** Establish baseline performance for sentiment classification

---

## 📋 Overview

### What We're Building
A **BERT-based sentiment classifier** that:
- Takes smartphone review text as input
- Classifies sentiment as Positive/Negative/Neutral
- Uses transfer learning from pretrained BERT
- Applies class weights to handle imbalance

### Why This Approach?
✅ **Fast implementation** (~2-3 days)  
✅ **Proven architecture** (BERT is industry standard)  
✅ **Strong baseline** (expected 80-85% accuracy)  
✅ **Foundation for Stage 2** (RoBERTa enhancement)  
✅ **Fallback option** if time runs short

---

## 📁 Folder Structure

```
smartReview/
│
├── src/
│   ├── data/
│   │   └── preprocessor.py          # ✅ Already created
│   │
│   ├── models/
│   │   ├── __init__.py              # 🆕 Will create
│   │   ├── bert_classifier.py       # 🆕 BERT model definition
│   │   └── trainer.py               # 🆕 Training logic
│   │
│   └── utils/
│       ├── __init__.py              # 🆕 Will create
│       ├── metrics.py               # 🆕 Evaluation metrics
│       └── dataset.py               # 🆕 PyTorch Dataset class
│
├── notebooks/
│   ├── 01_eda.ipynb                 # ✅ Completed
│   ├── 02_preprocessing.ipynb       # ✅ Completed
│   └── 03_baseline_training.ipynb   # 🆕 Will create
│
├── config/
│   ├── aspects.json                 # ✅ Already exists
│   └── training_config.yaml         # 🆕 Will create
│
├── models/
│   └── baseline/
│       ├── checkpoints/             # 🆕 Model checkpoints
│       ├── logs/                    # 🆕 Training logs
│       └── results/                 # 🆕 Evaluation results
│
└── outputs/
    └── figures/
        └── training/                # 🆕 Training visualizations
```

---

## 🎯 Phase 4 Steps Breakdown

### **Step 1: Environment Setup** ⏱️ 10-15 minutes
Install PyTorch and Transformers libraries:
```bash
# Check if GPU is available (optional but recommended)
pip install torch torchvision torchaudio

# Install Transformers library
pip install transformers

# Install additional dependencies
pip install scikit-learn tensorboard
```

**What to verify:**
- PyTorch installation (CPU or GPU)
- Transformers version ≥ 4.30.0
- CUDA availability (if using GPU)

---

### **Step 2: Create Model Architecture** ⏱️ 30 minutes

**File: `src/models/bert_classifier.py`**

```python
"""
BERT-based Sentiment Classifier
- Uses BERT-base-uncased (110M parameters)
- Adds classification head for 3-class sentiment
- Supports class weighting for imbalanced data
"""
```

**Key Components:**
1. **BERTSentimentClassifier class**
   - Loads pretrained BERT
   - Adds dropout + linear layer
   - Returns logits for 3 classes

2. **Configuration**
   - Model: `bert-base-uncased`
   - Hidden size: 768
   - Dropout: 0.1
   - Output: 3 classes (Positive/Negative/Neutral)

---

### **Step 3: Create Dataset Class** ⏱️ 20 minutes

**File: `src/utils/dataset.py`**

```python
"""
PyTorch Dataset for Review Sentiment Classification
- Tokenizes text using BERT tokenizer
- Creates attention masks
- Returns tensors for training
"""
```

**Key Features:**
- Tokenization with max_length=512
- Padding and truncation
- Attention mask generation
- Label encoding (Positive=2, Neutral=1, Negative=0)

---

### **Step 4: Create Training Pipeline** ⏱️ 45 minutes

**File: `src/models/trainer.py`**

```python
"""
Training Pipeline for BERT Sentiment Classifier
- Handles training loop
- Computes weighted loss
- Tracks metrics (accuracy, F1)
- Saves checkpoints
"""
```

**Training Configuration:**
- **Optimizer:** AdamW (lr=2e-5)
- **Loss:** CrossEntropyLoss with class weights
- **Batch size:** 16 (GPU) or 8 (CPU)
- **Epochs:** 3
- **Gradient clipping:** max_norm=1.0
- **Warmup steps:** 500

**Class Weights (from Phase 2.5):**
```python
class_weights = {
    'Negative': 0.676,
    'Neutral': 3.165,
    'Positive': 0.454
}
```

---

### **Step 5: Create Metrics Module** ⏱️ 15 minutes

**File: `src/utils/metrics.py`**

```python
"""
Evaluation Metrics for Sentiment Classification
- Accuracy, Precision, Recall, F1
- Per-class metrics
- Confusion matrix
"""
```

**Metrics to Track:**
- Overall accuracy
- Per-class precision/recall/F1
- Macro-averaged F1
- Weighted F1
- Confusion matrix

---

### **Step 6: Create Training Configuration** ⏱️ 10 minutes

**File: `config/training_config.yaml`**

```yaml
# Baseline BERT Training Configuration
model:
  name: "bert-base-uncased"
  hidden_size: 768
  num_labels: 3
  dropout: 0.1

training:
  batch_size: 16          # Reduce to 8 if GPU memory issues
  num_epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_grad_norm: 1.0
  
data:
  max_length: 512
  train_path: "Dataset/processed/train.csv"
  val_path: "Dataset/processed/val.csv"
  test_path: "Dataset/processed/test.csv"
  
class_weights:
  Negative: 0.676
  Neutral: 3.165
  Positive: 0.454
```

---

### **Step 7: Create Training Notebook** ⏱️ 30 minutes

**File: `notebooks/03_baseline_training.ipynb`**

**Notebook Structure:**
1. ✅ Setup & imports
2. ✅ Load processed data
3. ✅ Create datasets and dataloaders
4. ✅ Initialize model
5. ✅ Training loop (3 epochs)
6. ✅ Validation after each epoch
7. ✅ Save best model
8. ✅ Final evaluation on test set
9. ✅ Visualizations (loss curves, confusion matrix)
10. ✅ Performance report

---

### **Step 8: Train the Model** ⏱️ 2-3 hours

**Expected Timeline:**
- **With GPU (CUDA):** ~45-60 minutes per epoch = 2-3 hours total
- **With CPU:** ~3-4 hours per epoch = 9-12 hours total

**What Happens:**
1. Load train/val datasets (~43K train, ~9K val)
2. Train for 3 epochs with validation
3. Save best checkpoint based on val F1 score
4. Generate training curves
5. Evaluate on test set (~9K samples)

---

### **Step 9: Evaluate Performance** ⏱️ 30 minutes

**Evaluation Metrics:**
```python
Expected Performance (Baseline):
├── Overall Accuracy: 80-85%
├── Macro F1: 0.75-0.80
│
├── Positive (68% of data):
│   ├── Precision: 0.85-0.88
│   ├── Recall: 0.90-0.93
│   └── F1: 0.87-0.90
│
├── Negative (25% of data):
│   ├── Precision: 0.78-0.82
│   ├── Recall: 0.75-0.80
│   └── F1: 0.76-0.81
│
└── Neutral (7% of data):
    ├── Precision: 0.60-0.70
    ├── Recall: 0.65-0.75
    └── F1: 0.62-0.72
```

**Visualizations:**
- Training/validation loss curves
- Accuracy curves
- Confusion matrix
- Per-class F1 scores
- Classification report

---

## 📊 Expected Outputs

### Model Artifacts
```
models/baseline/
├── checkpoints/
│   ├── epoch_1.pt              # Checkpoint after epoch 1
│   ├── epoch_2.pt              # Checkpoint after epoch 2
│   ├── epoch_3.pt              # Checkpoint after epoch 3
│   └── best_model.pt           # Best model (highest val F1)
│
├── logs/
│   ├── training_log.txt        # Training metrics per batch
│   └── tensorboard/            # TensorBoard logs
│
└── results/
    ├── test_results.json       # Final test metrics
    ├── confusion_matrix.png    # Confusion matrix
    ├── training_curves.png     # Loss/accuracy curves
    └── classification_report.txt
```

---

## 🎓 Learning Objectives

### By completing Phase 4, you'll understand:

1. **Transfer Learning**
   - How pretrained BERT works
   - Fine-tuning vs training from scratch
   - When to freeze layers

2. **Handling Class Imbalance**
   - Computing class weights
   - Weighted loss functions
   - Impact on minority class performance

3. **BERT Tokenization**
   - WordPiece tokenization
   - Special tokens ([CLS], [SEP], [PAD])
   - Attention masks

4. **Training Best Practices**
   - Learning rate scheduling
   - Gradient clipping
   - Early stopping
   - Checkpoint saving

5. **Evaluation Metrics**
   - Why accuracy alone isn't enough
   - Importance of F1 for imbalanced data
   - Interpreting confusion matrices

---

## ⚠️ Common Issues & Solutions

### Issue 1: Out of Memory (GPU)
**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size: `16 → 8 → 4`
2. Reduce max sequence length: `512 → 256`
3. Use gradient accumulation
4. Switch to CPU (slower but works)

### Issue 2: Slow Training (CPU)
**Problem:** Each epoch takes 3-4 hours

**Solutions:**
1. Use smaller subset for testing (10% of data)
2. Reduce epochs: `3 → 2`
3. Use Google Colab GPU (free)
4. Use cloud GPU (AWS, Azure)

### Issue 3: Low Neutral Class Performance
**Problem:** Neutral F1 < 0.60

**Solutions:**
1. Increase Neutral class weight: `3.165 → 4.0`
2. Train for more epochs: `3 → 5`
3. Adjust learning rate: `2e-5 → 3e-5`
4. Check data quality (might be truly ambiguous)

### Issue 4: Model Overfitting
**Problem:** Train accuracy >> Val accuracy

**Solutions:**
1. Increase dropout: `0.1 → 0.3`
2. Add weight decay: `0.01 → 0.05`
3. Reduce epochs
4. Use early stopping

---

## 🚦 Success Criteria

### Minimum Requirements (PASS)
- ✅ Model trains without errors
- ✅ Overall accuracy ≥ 75%
- ✅ All classes have F1 ≥ 0.50
- ✅ Model saved successfully
- ✅ Evaluation visualizations generated

### Target Performance (GOOD)
- ✅ Overall accuracy ≥ 80%
- ✅ Macro F1 ≥ 0.75
- ✅ Neutral F1 ≥ 0.65
- ✅ Training curves show convergence
- ✅ No significant overfitting

### Excellent Performance (EXCELLENT)
- ✅ Overall accuracy ≥ 85%
- ✅ Macro F1 ≥ 0.78
- ✅ Neutral F1 ≥ 0.70
- ✅ Smooth training curves
- ✅ Generalization gap < 5%

---

## 🔄 Next Steps After Phase 4

### Stage 2: Enhanced RoBERTa Model
1. **Continued Pretraining**
   - Train on 61K reviews (Masked Language Modeling)
   - Learn domain-specific vocabulary
   - Expected: +3-5% accuracy improvement

2. **Fine-tuning**
   - Use pretrained RoBERTa
   - Same architecture as baseline
   - Compare performance

3. **Analysis**
   - Side-by-side comparison
   - Where did RoBERTa improve?
   - Error analysis

### Expected Improvements (Stage 2 vs Stage 1)
```
Metric                Stage 1 (BERT)    Stage 2 (RoBERTa)    Improvement
─────────────────────────────────────────────────────────────────────────
Overall Accuracy        82-85%            87-90%              +5-7%
Positive F1             0.87-0.90         0.90-0.92           +3-5%
Negative F1             0.76-0.81         0.82-0.86           +6-8%
Neutral F1              0.62-0.72         0.75-0.82           +12-15%
Macro F1                0.75-0.81         0.82-0.87           +7-9%
```

---

## 📚 Additional Resources

### Papers to Read
1. **BERT:** "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
2. **RoBERTa:** "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
3. **ABSA:** "Aspect-Based Sentiment Analysis" survey papers

### Tutorials
- Hugging Face BERT Tutorial
- PyTorch Text Classification Guide
- Fine-tuning Transformers for Text Classification

---

## ⏱️ Total Time Estimate

| Step | Task | Time |
|------|------|------|
| 1 | Environment setup | 10-15 min |
| 2 | Model architecture | 30 min |
| 3 | Dataset class | 20 min |
| 4 | Training pipeline | 45 min |
| 5 | Metrics module | 15 min |
| 6 | Configuration file | 10 min |
| 7 | Training notebook | 30 min |
| 8 | **Model training** | **2-3 hours** |
| 9 | Evaluation & analysis | 30 min |
| **TOTAL** | | **~5-6 hours** |

**Recommendation:** Break this into 2-3 work sessions
- **Session 1:** Steps 1-7 (setup & code)
- **Session 2:** Step 8 (training - let it run)
- **Session 3:** Step 9 (evaluation & visualization)

---

## 🎯 Ready to Start?

**Next Action:** Let's begin with **Step 1** - creating the model architecture file!

Would you like me to create:
1. `src/models/bert_classifier.py` - Model definition
2. `src/utils/dataset.py` - Dataset class
3. `src/models/trainer.py` - Training logic
4. All of the above + configuration files

Let me know and we'll proceed step by step! 🚀
