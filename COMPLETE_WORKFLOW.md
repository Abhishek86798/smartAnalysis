# 🚀 COMPLETE PROJECT WORKFLOW - SmartReview ABSA System

**Last Updated:** October 28, 2025  
**Your Current Status:** ✅ Python 3.13.8 installed, Virtual environment created

---

## 📍 **WHERE YOU ARE NOW**

✅ **COMPLETED:**
- Python 3.13.8 installed
- Virtual environment `venv` created and activated
- Dataset downloaded (68K reviews)
- Project structure initialized

🔄 **IN PROGRESS:**
- Installing required packages

⏳ **NEXT:**
- Complete environment setup
- Run first EDA

---

# 🎯 **YOUR COMPLETE STEP-BY-STEP FLOW**

---

## **PHASE 1: ENVIRONMENT SETUP** (TODAY - 1 hour)

### ✅ Step 1.1: Install Core Packages (15 mins)

**What:** Install essential Python libraries  
**Why:** Need these to work with data and models  
**How:**

```bash
# Make sure venv is activated (you'll see (venv) in prompt)
# If not: .\venv\Scripts\Activate.ps1

# Install basic packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook ipykernel tqdm

# Install PyTorch (for deep learning)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers (for BERT/RoBERTa)
pip install transformers datasets accelerate evaluate

# Install text processing
pip install nltk spacy wordcloud

# Install visualization
pip install plotly

# Optional: Experiment tracking
pip install wandb
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

**Expected Output:**
```
PyTorch: 2.x.x
CUDA Available: True (or False if no GPU)
Transformers: 4.35.x
Pandas: 2.x.x
```

---

### ✅ Step 1.2: Download NLTK & spaCy Data (10 mins)

**What:** Download language models for text processing  
**Why:** Needed for tokenization, stopwords, lemmatization  
**How:**

```python
# Create a setup script
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

### ✅ Step 1.3: Create Project Folders (5 mins)

**What:** Set up organized folder structure  
**Why:** Keep code, data, and outputs organized  
**How:**

```bash
# Create all necessary folders
mkdir notebooks
mkdir src
mkdir src\data
mkdir src\models
mkdir src\evaluation
mkdir src\visualization
mkdir models
mkdir outputs
mkdir outputs\figures
mkdir outputs\reports
mkdir outputs\predictions
mkdir tests
mkdir docs
```

**Verify:**
```bash
tree /F  # Shows folder structure
```

---

### ✅ Step 1.4: Test GPU Availability (5 mins)

**What:** Check if you have GPU access for faster training  
**Why:** GPU makes training 10-100x faster  
**How:**

```python
# Create test_gpu.py
import torch

print("=" * 50)
print("🔍 GPU CHECK")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print("✅ GPU is ready for training!")
else:
    print("⚠️ No GPU detected. Training will use CPU (slower)")
    print("💡 Consider using Google Colab for free GPU access")
print("=" * 50)
```

Run: `python test_gpu.py`

**If NO GPU:**
- ✅ Don't worry! You can use Google Colab (free GPU)
- ✅ Or cloud services: Kaggle, Paperspace
- ✅ CPU training works, just slower

---

## **PHASE 2: DATA EXPLORATION (EDA)** (Days 1-2)

### 🎯 Goal: Understand your dataset deeply before any modeling

---

### ✅ Step 2.1: Create EDA Notebook (1 hour)

**What:** Build a Jupyter notebook to explore data  
**Why:** Need to understand data quality, patterns, distributions  
**How:**

```bash
# Start Jupyter
jupyter notebook
```

**Create:** `notebooks/01_eda.ipynb`

**What to Include:**

#### 📊 **Part A: Load & Basic Info**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
reviews_df = pd.read_csv('../Dataset/20191226-reviews.csv')
items_df = pd.read_csv('../Dataset/20191226-items.csv')

# Basic info
print("📊 DATASET OVERVIEW")
print("=" * 50)
print(f"Total Reviews: {len(reviews_df):,}")
print(f"Total Products: {len(items_df):,}")
print(f"\nColumns in Reviews: {reviews_df.columns.tolist()}")
print(f"\nFirst few rows:")
display(reviews_df.head())

# Check missing values
print("\n🔍 MISSING VALUES")
print(reviews_df.isnull().sum())
```

#### 📊 **Part B: Rating Distribution**
```python
# Rating distribution
plt.figure(figsize=(10, 6))
reviews_df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Ratings', fontsize=16, fontweight='bold')
plt.xlabel('Rating (Stars)', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/figures/rating_distribution.png', dpi=300)
plt.show()

# Print statistics
print("📈 RATING STATISTICS")
print(reviews_df['rating'].value_counts().sort_index())
print(f"\nMean Rating: {reviews_df['rating'].mean():.2f}")
print(f"Median Rating: {reviews_df['rating'].median():.2f}")
```

#### 📊 **Part C: Review Length Analysis**
```python
# Calculate review lengths
reviews_df['review_length'] = reviews_df['body'].fillna('').apply(len)
reviews_df['word_count'] = reviews_df['body'].fillna('').apply(lambda x: len(x.split()))

# Statistics
print("📏 REVIEW LENGTH STATISTICS")
print(f"Average characters: {reviews_df['review_length'].mean():.0f}")
print(f"Average words: {reviews_df['word_count'].mean():.0f}")
print(f"Shortest review: {reviews_df['review_length'].min()} chars")
print(f"Longest review: {reviews_df['review_length'].max()} chars")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(reviews_df['word_count'], bins=50, color='coral', edgecolor='black')
axes[0].set_title('Distribution of Word Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Frequency')
axes[0].axvline(reviews_df['word_count'].mean(), color='red', linestyle='--', label='Mean')
axes[0].legend()

axes[1].boxplot(reviews_df['word_count'])
axes[1].set_title('Word Count Boxplot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Words')

plt.tight_layout()
plt.savefig('../outputs/figures/review_length_analysis.png', dpi=300)
plt.show()
```

#### 📊 **Part D: Brand Analysis**
```python
# Merge to get brand info
merged = reviews_df.merge(items_df[['asin', 'brand']], on='asin', how='left')

# Top brands
top_brands = merged['brand'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_brands.plot(kind='barh', color='lightgreen')
plt.title('Top 10 Brands by Review Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Reviews')
plt.ylabel('Brand')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/top_brands.png', dpi=300)
plt.show()
```

#### 📊 **Part E: Sentiment Distribution**
```python
# Create sentiment labels
def get_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

reviews_df['sentiment'] = reviews_df['rating'].apply(get_sentiment)

# Distribution
sentiment_counts = reviews_df['sentiment'].value_counts()

plt.figure(figsize=(8, 8))
colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
plt.savefig('../outputs/figures/sentiment_distribution.png', dpi=300)
plt.show()

print("😊 SENTIMENT BREAKDOWN")
print(sentiment_counts)
print(f"\nPercentages:")
print(sentiment_counts / len(reviews_df) * 100)
```

#### 📊 **Part F: Sample Reviews**
```python
# Show sample reviews from each sentiment
print("📝 SAMPLE REVIEWS")
print("\n" + "="*70)
print("😊 POSITIVE REVIEW:")
print("="*70)
pos_sample = reviews_df[reviews_df['sentiment']=='Positive'].sample(1).iloc[0]
print(f"Rating: {pos_sample['rating']} ⭐")
print(f"Title: {pos_sample['title']}")
print(f"Review: {pos_sample['body'][:500]}...")

print("\n" + "="*70)
print("😐 NEUTRAL REVIEW:")
print("="*70)
neu_sample = reviews_df[reviews_df['sentiment']=='Neutral'].sample(1).iloc[0]
print(f"Rating: {neu_sample['rating']} ⭐")
print(f"Title: {neu_sample['title']}")
print(f"Review: {neu_sample['body'][:500]}...")

print("\n" + "="*70)
print("😞 NEGATIVE REVIEW:")
print("="*70)
neg_sample = reviews_df[reviews_df['sentiment']=='Negative'].sample(1).iloc[0]
print(f"Rating: {neg_sample['rating']} ⭐")
print(f"Title: {neg_sample['title']}")
print(f"Review: {neg_sample['body'][:500]}...")
```

**✅ Deliverable:** 
- EDA notebook with visualizations
- 5-6 saved plots in `outputs/figures/`
- Understanding of data quality and distribution

---

### ✅ Step 2.2: Identify Common Aspects (1 hour)

**What:** Manually read reviews to find common phone aspects  
**Why:** Need to know what aspects to extract (battery, camera, etc.)  
**How:**

```python
# Sample random reviews and read them
import random

random_reviews = reviews_df.sample(50)

print("🔍 MANUAL ASPECT IDENTIFICATION")
print("Read these reviews and note common aspects mentioned:\n")

for idx, row in random_reviews.iterrows():
    print(f"\n--- Review {idx} ---")
    print(f"Rating: {row['rating']} ⭐")
    print(f"Review: {row['body'][:300]}...")
    print("-" * 70)
```

**Your Task:**
- Read 30-50 reviews
- Note down aspects mentioned (battery, screen, camera, etc.)
- This validates the aspects in `config/aspects.json`

**✅ Deliverable:**
- List of common aspects found
- Confidence that aspect dictionary is complete

---

## **PHASE 3: DATA PREPROCESSING** (Days 3-4)

### 🎯 Goal: Clean and prepare data for model training

---

### ✅ Step 3.1: Create Preprocessing Module (2 hours)

**What:** Build Python module to clean text data  
**Why:** BERT needs clean, properly formatted text  
**How:**

**Create:** `src/data/preprocessor.py`

```python
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ReviewPreprocessor:
    def __init__(self):
        # Download NLTK data if not already downloaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean review text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation (keep basic ones)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_sentiment_label(self, rating):
        """Convert rating to sentiment label"""
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def filter_short_reviews(self, df, min_words=10):
        """Remove very short reviews"""
        df['word_count'] = df['body'].fillna('').apply(lambda x: len(x.split()))
        return df[df['word_count'] >= min_words].copy()
    
    def preprocess_dataframe(self, df):
        """Complete preprocessing pipeline"""
        print("🔧 Starting preprocessing...")
        
        # Remove null reviews
        print(f"Removing null reviews...")
        df = df[df['body'].notna()].copy()
        
        # Clean text
        print(f"Cleaning text...")
        df['cleaned_text'] = df['body'].apply(self.clean_text)
        
        # Remove very short reviews
        print(f"Filtering short reviews...")
        df = self.filter_short_reviews(df, min_words=10)
        
        # Create sentiment labels
        print(f"Creating sentiment labels...")
        df['sentiment'] = df['rating'].apply(self.create_sentiment_label)
        
        # Remove duplicates
        print(f"Removing duplicates...")
        df = df.drop_duplicates(subset=['cleaned_text']).copy()
        
        print(f"✅ Preprocessing complete!")
        print(f"Final dataset size: {len(df):,} reviews")
        
        return df

# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load data
    df = pd.read_csv('../../Dataset/20191226-reviews.csv')
    
    # Preprocess
    preprocessor = ReviewPreprocessor()
    clean_df = preprocessor.preprocess_dataframe(df)
    
    # Save
    clean_df.to_csv('../../Dataset/cleaned_reviews.csv', index=False)
    print("💾 Saved to Dataset/cleaned_reviews.csv")
```

**Run:**
```bash
python src/data/preprocessor.py
```

**✅ Deliverable:**
- `src/data/preprocessor.py` module
- `Dataset/cleaned_reviews.csv` (cleaned data)

---

### ✅ Step 3.2: Train/Val/Test Split (30 mins)

**What:** Split data into training, validation, and test sets  
**Why:** Need separate data for training, tuning, and evaluation  
**How:**

**Create:** `src/data/split_data.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv('../../Dataset/cleaned_reviews.csv')

print(f"Total samples: {len(df):,}")

# Stratified split (maintains sentiment distribution)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.3, 
    random_state=42, 
    stratify=df['sentiment']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['sentiment']
)

print(f"\n📊 SPLIT SIZES:")
print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

# Check sentiment distribution
print(f"\n😊 SENTIMENT DISTRIBUTION:")
print(f"Train:\n{train_df['sentiment'].value_counts(normalize=True)}")
print(f"\nVal:\n{val_df['sentiment'].value_counts(normalize=True)}")
print(f"\nTest:\n{test_df['sentiment'].value_counts(normalize=True)}")

# Save splits
train_df.to_csv('../../Dataset/train.csv', index=False)
val_df.to_csv('../../Dataset/val.csv', index=False)
test_df.to_csv('../../Dataset/test.csv', index=False)

print(f"\n✅ Saved splits to Dataset/")
```

**Run:**
```bash
python src/data/split_data.py
```

**✅ Deliverable:**
- `Dataset/train.csv` (~70%)
- `Dataset/val.csv` (~15%)
- `Dataset/test.csv` (~15%)

---

## **PHASE 4: ASPECT EXTRACTION** (Days 5-6)

### 🎯 Goal: Extract product aspects from reviews

---

### ✅ Step 4.1: Rule-Based Aspect Extraction (2 hours)

**What:** Use keyword matching to find aspects in reviews  
**Why:** Need to identify which aspects are mentioned in each review  
**How:**

**Create:** `src/data/aspect_extractor.py`

```python
import json
import pandas as pd
import re

class AspectExtractor:
    def __init__(self, config_path='../../config/aspects.json'):
        """Load aspect keywords from config"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.aspects = config['aspects']
    
    def extract_aspects(self, text):
        """Extract all aspects mentioned in text"""
        if pd.isna(text):
            return []
        
        text = text.lower()
        found_aspects = []
        
        for aspect_name, aspect_info in self.aspects.items():
            keywords = aspect_info['keywords']
            
            # Check if any keyword is in the text
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    found_aspects.append(aspect_name)
                    break  # Don't count same aspect twice
        
        return found_aspects
    
    def extract_aspect_sentences(self, text, aspect):
        """Extract sentences that mention a specific aspect"""
        if pd.isna(text):
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        aspect_keywords = self.aspects[aspect]['keywords']
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            for keyword in aspect_keywords:
                if keyword in sentence:
                    relevant_sentences.append(sentence)
                    break
        
        return relevant_sentences
    
    def process_dataframe(self, df):
        """Add aspect information to dataframe"""
        print("🔍 Extracting aspects from reviews...")
        
        df['aspects'] = df['cleaned_text'].apply(self.extract_aspects)
        df['aspect_count'] = df['aspects'].apply(len)
        
        # Create binary columns for each aspect
        for aspect_name in self.aspects.keys():
            df[f'has_{aspect_name}'] = df['aspects'].apply(
                lambda x: 1 if aspect_name in x else 0
            )
        
        print(f"✅ Aspect extraction complete!")
        return df

# Example usage
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('../../Dataset/train.csv')
    
    # Extract aspects
    extractor = AspectExtractor()
    df_with_aspects = extractor.process_dataframe(df)
    
    # Statistics
    print(f"\n📊 ASPECT STATISTICS:")
    print(f"Reviews with at least 1 aspect: {(df_with_aspects['aspect_count'] > 0).sum():,}")
    print(f"Average aspects per review: {df_with_aspects['aspect_count'].mean():.2f}")
    
    print(f"\n🔝 TOP ASPECTS MENTIONED:")
    aspect_counts = {}
    for aspect_name in extractor.aspects.keys():
        count = df_with_aspects[f'has_{aspect_name}'].sum()
        aspect_counts[aspect_name] = count
    
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
    for aspect, count in sorted_aspects[:10]:
        print(f"{aspect:15s}: {count:6,} reviews ({count/len(df)*100:5.1f}%)")
    
    # Save
    df_with_aspects.to_csv('../../Dataset/train_with_aspects.csv', index=False)
    print(f"\n💾 Saved to Dataset/train_with_aspects.csv")
```

**Run:**
```bash
python src/data/aspect_extractor.py
```

**✅ Deliverable:**
- `src/data/aspect_extractor.py`
- `Dataset/train_with_aspects.csv`
- Statistics on aspect frequency

---

## **PHASE 5: BASELINE MODEL** (Week 2)

### 🎯 Goal: Train first working BERT model for sentiment

---

### ✅ Step 5.1: Test Loading Pretrained Model (30 mins)

**What:** Verify you can load BERT model  
**Why:** Make sure everything works before training  
**How:**

**Create:** `test_model_loading.py`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("🤖 Testing Model Loading...")

# Test 1: BERT-base
print("\n1️⃣ Loading BERT-base-uncased...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
print("✅ BERT loaded successfully!")

# Test 2: Test tokenization
sample_text = "The battery life is amazing but the screen is too small."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
print(f"\n📝 Sample tokenization:")
print(f"Input text: {sample_text}")
print(f"Token IDs shape: {inputs['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# Test 3: Check if model can run
with torch.no_grad():
    outputs = model(**inputs)
print(f"\n✅ Model forward pass successful!")
print(f"Output shape: {outputs.last_hidden_state.shape}")

print("\n🎉 All tests passed! Ready for training!")
```

**Run:**
```bash
python test_model_loading.py
```

---

### ✅ Step 5.2: Train Baseline BERT (4-6 hours)

**What:** Fine-tune BERT for sentiment classification  
**Why:** Establish baseline performance to beat later  
**How:**

**Create:** `src/models/train_baseline.py`

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return accuracy, f1, predictions, true_labels

def main():
    # Configuration
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv('../../Dataset/train.csv')
    val_df = pd.read_csv('../../Dataset/val.csv')
    
    # Convert sentiment to numeric labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['label'] = train_df['sentiment'].map(label_map)
    val_df['label'] = val_df['sentiment'].map(label_map)
    
    print(f"Train size: {len(train_df):,}")
    print(f"Val size: {len(val_df):,}")
    
    # Load tokenizer and model
    print(f"\n🤖 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    model.to(device)
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = ReviewDataset(
        train_df['cleaned_text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = ReviewDataset(
        val_df['cleaned_text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        accuracy, f1, _, _ = evaluate(model, val_loader, device)
        print(f"Val Accuracy: {accuracy:.4f}")
        print(f"Val F1 Score: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            print(f"💾 New best F1! Saving model...")
            model.save_pretrained('../../models/baseline_bert')
            tokenizer.save_pretrained('../../models/baseline_bert')
    
    print(f"\n✅ Training complete!")
    print(f"Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python src/models/train_baseline.py
```

**✅ Deliverable:**
- Trained BERT model in `models/baseline_bert/`
- Training logs with accuracy and F1 scores
- Baseline performance metrics

---

## **PHASE 6: EVALUATION & VISUALIZATION** (Days 9-10)

### ✅ Step 6.1: Evaluate on Test Set (1 hour)

**Create:** `src/evaluation/evaluate_model.py`

```python
# Load test data
# Run predictions
# Generate classification report
# Create confusion matrix
# Save results
```

### ✅ Step 6.2: Create Visualizations (2 hours)

**Create:** `src/visualization/plots.py`

```python
# Confusion matrix
# Aspect distribution charts
# Sentiment by aspect
# Word clouds
# Performance comparisons
```

---

## **PHASE 7: ENHANCED MODEL** (Week 3+)

### ✅ Step 7.1: Train RoBERTa

### ✅ Step 7.2: (Optional) Continued Pretraining

### ✅ Step 7.3: Compare Models

---

## **PHASE 8: (OPTIONAL) WEB APP** (Week 4+)

### ✅ Step 8.1: Create Streamlit App

### ✅ Step 8.2: Deploy

---

# 📋 **QUICK REFERENCE CHECKLIST**

Copy this to track your progress:

```
SETUP & ENVIRONMENT
[ ] Python 3.9+ installed
[ ] Virtual environment created
[ ] All packages installed
[ ] GPU tested
[ ] Folder structure created

DATA EXPLORATION
[ ] EDA notebook created
[ ] Rating distribution analyzed
[ ] Review lengths checked
[ ] Sample reviews read
[ ] Aspects identified

DATA PREPROCESSING
[ ] Preprocessor module created
[ ] Data cleaned
[ ] Train/val/test split done
[ ] Aspect extraction implemented

BASELINE MODEL
[ ] BERT loading tested
[ ] Training script created
[ ] Model trained (3 epochs)
[ ] Baseline metrics recorded

EVALUATION
[ ] Test set evaluation done
[ ] Confusion matrix created
[ ] Error analysis completed
[ ] Visualizations created

ENHANCED MODEL
[ ] RoBERTa fine-tuned
[ ] (Optional) Continued pretraining done
[ ] Models compared
[ ] Best model selected

DELIVERABLES
[ ] Project report written
[ ] Presentation created
[ ] Code documented
[ ] (Optional) Web app deployed
```

---

# 🆘 **TROUBLESHOOTING**

### Problem: CUDA out of memory
**Solution:** Reduce batch size to 8 or 4

### Problem: Training too slow
**Solution:** Use Google Colab with free GPU

### Problem: Low accuracy
**Solution:** Train for more epochs, tune hyperparameters

### Problem: Model not loading
**Solution:** Check internet connection, try different model

---

# 📞 **NEED HELP?**

1. Check error message carefully
2. Google the exact error
3. Check Hugging Face forums
4. Ask on Stack Overflow

---

**🎯 YOUR IMMEDIATE NEXT STEP:**

Run the package installation commands and create the EDA notebook!

```bash
# Install packages (if not done)
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook

# Start Jupyter
jupyter notebook

# Create 01_eda.ipynb in notebooks folder
```

**You're ready to start! 🚀**
