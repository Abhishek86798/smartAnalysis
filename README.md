
# 🎯 SmartReview - Intelligent Product Review Analytics System

> **BE Project 2025 | Enhanced BERT for Aspect-Based Sentiment Analysis**

---

## 📖 Project Overview

An intelligent review analytics system that uses Enhanced BERT models to perform **Aspect-Based Sentiment Analysis (ABSA)** on smartphone and electronics reviews. The system extracts product aspects (battery, camera, screen, etc.) and determines sentiment for each aspect, providing actionable insights for consumers and manufacturers.

---

## 🎯 Objectives

1. ✅ Build a robust ABSA system for smartphone reviews
2. ✅ Extract key product aspects automatically
3. ✅ Analyze sentiment per aspect (positive/negative/neutral)
4. ✅ Compare baseline BERT vs enhanced models (RoBERTa/DeBERTa)
5. ✅ Create interactive visualizations for insights
6. ✅ (Optional) Deploy as web application

---

## 📊 Dataset

**Source:** Amazon Cell Phones Reviews (Kaggle - Option 1)

### Files:
- `20191226-items.csv` - 721 products with metadata
- `20191226-reviews.csv` - **67,987 reviews** with ratings (1-5 stars)

### Key Columns:
- `body` - Review text (main feature)
- `rating` - Star rating (1-5)
- `title` - Review summary
- `verified` - Purchase verification
- `asin` - Product ID

### Statistics:
- **Total Reviews:** 67,987
- **Average Rating:** ~3.5 stars (to be confirmed)
- **Date Range:** Pre-2019
- **Brands:** Samsung, Motorola, Nokia, Huawei, etc.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Review Text                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Pipeline                     │
│  • Text cleaning  • Tokenization  • Aspect labeling         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Enhanced BERT Model                         │
│  ┌──────────────────────────────────────────────┐          │
│  │  Pretrained BERT/RoBERTa/DeBERTa            │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                          │
│                   ▼                                          │
│  ┌──────────────────────────────────────────────┐          │
│  │  Continued Pretraining (Optional)            │          │
│  │  Domain Adaptation on Phone Reviews          │          │
│  └────────────────┬─────────────────────────────┘          │
│                   │                                          │
│                   ▼                                          │
│  ┌──────────────────────────────────────────────┐          │
│  │  Fine-tuning for ABSA                        │          │
│  │  • Aspect Extraction                         │          │
│  │  • Aspect Sentiment Classification           │          │
│  └────────────────┬─────────────────────────────┘          │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Output & Visualization                       │
│  • Per-aspect sentiment  • Importance-Performance Analysis  │
│  • Word clouds  • Interactive dashboards                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### Phase 1: Data Understanding & EDA
- [ ] Load and explore dataset
- [ ] Analyze rating distribution
- [ ] Check review length statistics
- [ ] Identify common aspects in reviews
- [ ] Visualize key patterns

### Phase 2: Data Preprocessing
- [ ] Text cleaning (lowercase, remove special chars)
- [ ] Handle missing values
- [ ] Create sentiment labels (1-2: Neg, 3: Neutral, 4-5: Pos)
- [ ] Train/Val/Test split (70/15/15)
- [ ] Tokenization for BERT

### Phase 3: Aspect Extraction
- [ ] Define aspect categories (battery, screen, camera, etc.)
- [ ] Create aspect keyword dictionary
- [ ] Rule-based aspect extraction
- [ ] (Advanced) Train aspect NER model

### Phase 4: Model Development

#### **Baseline Model**
- [ ] Load pretrained BERT-base-uncased
- [ ] Fine-tune for sentiment classification
- [ ] Evaluate on test set
- [ ] Establish baseline metrics

#### **Enhanced Model**
- [ ] Load RoBERTa-base or DeBERTa
- [ ] (Optional) Continued pretraining on phone reviews
- [ ] Fine-tune for ABSA
- [ ] Compare with baseline

### Phase 5: Aspect-Based Sentiment Analysis
- [ ] Extract aspects from reviews
- [ ] Predict sentiment per aspect
- [ ] Aggregate aspect-wise insights
- [ ] Generate aspect sentiment JSON

### Phase 6: Evaluation
- [ ] Metrics: Accuracy, Precision, Recall, F1
- [ ] Per-aspect performance analysis
- [ ] Confusion matrices
- [ ] Error analysis

### Phase 7: Visualization
- [ ] Aspect sentiment distribution charts
- [ ] Importance-Performance Analysis (IPA)
- [ ] Word clouds (positive/negative aspects)
- [ ] Interactive Plotly dashboards

### Phase 8: (Optional) Web Application
- [ ] Streamlit/FastAPI interface
- [ ] Real-time review analysis
- [ ] Dynamic visualizations
- [ ] User-friendly UI

---

## 🛠️ Technology Stack

### **Core Libraries**
```python
# Deep Learning
transformers==4.35.0      # BERT, RoBERTa, DeBERTa
torch==2.1.0              # PyTorch backend
datasets==2.14.0          # Dataset handling

# Data Processing
pandas==2.1.0             # Data manipulation
numpy==1.24.3             # Numerical operations
nltk==3.8.1               # Text processing
spacy==3.7.0              # NLP utilities

# Visualization
matplotlib==3.8.0         # Static plots
seaborn==0.13.0           # Statistical viz
plotly==5.17.0            # Interactive charts
wordcloud==1.9.2          # Word cloud generation

# Evaluation & Tracking
scikit-learn==1.3.0       # Metrics, preprocessing
evaluate==0.4.0           # Hugging Face metrics
wandb==0.16.0             # Experiment tracking

# Deployment (Optional)
streamlit==1.28.0         # Web app framework
fastapi==0.104.0          # REST API
gradio==4.4.0             # Quick UI
```

### **Hardware Requirements**
- **Minimum:** 16GB RAM, GPU with 8GB VRAM
- **Recommended:** 32GB RAM, GPU with 16GB+ VRAM
- **Cloud Options:** Google Colab Pro, Kaggle, AWS SageMaker

---

## 📁 Project Structure

```
smartReview/
│
├── Dataset/                          # Raw data
│   ├── 20191226-items.csv           # Product metadata
│   └── 20191226-reviews.csv         # Review data
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Data cleaning
│   ├── 03_aspect_extraction.ipynb   # Aspect identification
│   ├── 04_baseline_model.ipynb      # BERT baseline
│   ├── 05_enhanced_model.ipynb      # RoBERTa/DeBERTa
│   └── 06_visualization.ipynb       # Results visualization
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading utilities
│   │   ├── preprocessor.py          # Text preprocessing
│   │   └── aspect_extractor.py      # Aspect extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_baseline.py         # Baseline BERT
│   │   ├── enhanced_bert.py         # RoBERTa/DeBERTa
│   │   ├── absa_model.py            # ABSA implementation
│   │   └── trainer.py               # Training utilities
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── analyzer.py              # Error analysis
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                 # Plotting functions
│       └── dashboard.py             # Interactive dashboard
│
├── app/                              # Web application (optional)
│   ├── streamlit_app.py             # Streamlit interface
│   ├── api.py                       # FastAPI endpoints
│   └── utils.py                     # App utilities
│
├── models/                           # Saved models
│   ├── baseline_bert/               # Baseline model weights
│   ├── enhanced_model/              # Enhanced model weights
│   └── aspect_extractor/            # Aspect extraction model
│
├── outputs/                          # Results & visualizations
│   ├── figures/                     # Generated plots
│   ├── reports/                     # Analysis reports
│   └── predictions/                 # Model predictions
│
├── config/                           # Configuration files
│   ├── config.yaml                  # Main config
│   ├── aspects.json                 # Aspect definitions
│   └── hyperparameters.yaml         # Model hyperparameters
│
├── tests/                            # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── docs/                             # Documentation
│   ├── project_report.md            # Final report
│   ├── presentation.pptx            # Project presentation
│   └── references.md                # Research papers
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # This file
├── MODEL_RESEARCH.md                 # Model selection research
└── LICENSE                           # License file
```

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd smartReview
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train Models
```bash
# Baseline
python src/models/train_baseline.py

# Enhanced
python src/models/train_enhanced.py
```

### 5. (Optional) Launch Web App
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Key Aspects to Extract

Based on phone review analysis:

| Aspect | Keywords | Example |
|--------|----------|---------|
|  **Battery** | battery, charge, power, drain | "Battery life is amazing" |
|  **Screen** | screen, display, brightness | "Screen quality is excellent" |
|  **Camera** | camera, photo, picture, lens | "Camera is disappointing" |
|  **Signal** | signal, reception, network | "Great signal strength" |
|  **Audio** | speaker, sound, volume, ringer | "Loud and clear speaker" |
|  **Durability** | durable, drop, break, sturdy | "Phone is very fragile" |
|  **Performance** | fast, slow, lag, smooth | "Performance is sluggish" |
|  **Price** | price, value, cheap, expensive | "Worth every penny" |
|  **Design** | design, look, beautiful, ugly | "Sleek and modern design" |
|  **Buttons** | button, keypad, keyboard | "Buttons are too small" |

---

## 📈 Expected Results

### **Baseline BERT Model**
- **Aspect Extraction F1:** 0.72-0.75
- **Sentiment Accuracy:** 0.82-0.85
- **Training Time:** 3-4 hours

### **Enhanced Model (RoBERTa + Domain Adaptation)**
- **Aspect Extraction F1:** 0.78-0.82
- **Sentiment Accuracy:** 0.87-0.90
- **Training Time:** 8-12 hours

### **Improvement:**
- **+5-7%** in aspect extraction
- **+5%** in sentiment accuracy
- Better domain-specific understanding

---

## 📚 References

### **Research Papers**
1. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
2. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
3. "Understanding Pre-trained BERT for Aspect-based Sentiment Analysis"
4. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"

### **Datasets**
- Amazon Cell Phones Reviews (Kaggle)

### **Models**
- Hugging Face Transformers Library
- srimeenakshiks/BERT-ABSA

---

## 📝 License

This project is for educational purposes (BE Project 2025).

---

## 🎯 Project Status

**Current Phase:** Data Understanding & Model Research ✅

**Next Steps:** 
1. Complete EDA
2. Set up development environment
3. Implement preprocessing pipeline

---

**Last Updated:** June 12, 2026

