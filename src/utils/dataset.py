"""
PyTorch Dataset Class for SmartReview
======================================

Handles loading, tokenization, and batching of smartphone reviews
for BERT-based sentiment classification.

Author: SmartReview Team
Date: October 29, 2025
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from typing import Dict, Optional


class ReviewDataset(Dataset):
    """
    PyTorch Dataset for smartphone review sentiment classification.
    
    Tokenizes reviews using BERT tokenizer and prepares them for training.
    
    Features:
        - Automatic tokenization with padding/truncation
        - Label encoding (Positive=2, Negative=0, Neutral=1)
        - Attention mask generation
        - Efficient batching
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = 256,
        text_column: str = 'cleaned_text',
        label_column: str = 'sentiment'
    ):
        """
        Initialize Review Dataset.
        
        Args:
            dataframe: Pandas DataFrame with reviews and labels
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length (default: 256 for RTX 3050)
            text_column: Column name containing review text
            label_column: Column name containing sentiment labels
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Label encoding mapping
        self.label_map = {
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2
        }
        
        # Reverse mapping for predictions
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        print(f"✅ Dataset initialized:")
        print(f"   • Total samples: {len(self.dataframe):,}")
        print(f"   • Max sequence length: {max_length}")
        print(f"   • Text column: {text_column}")
        print(f"   • Label distribution:")
        for label, count in self.dataframe[label_column].value_counts().items():
            pct = count / len(self.dataframe) * 100
            print(f"      {label:8s}: {count:6,} ({pct:5.2f}%)")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
        
        Returns:
            Dictionary containing:
                - input_ids: Token IDs [max_length]
                - attention_mask: Attention mask [max_length]
                - labels: Label ID (0, 1, or 2)
        """
        # Get review text and label
        row = self.dataframe.iloc[idx]
        text = str(row[self.text_column])
        label = row[self.label_column]
        
        # Encode label to integer
        label_id = self.label_map[label]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_length,   # Truncate if longer
            padding='max_length',          # Pad if shorter
            truncation=True,               # Truncate long sequences
            return_attention_mask=True,    # Return attention mask
            return_tensors='pt'            # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
    
    def get_label_counts(self) -> Dict[str, int]:
        """Get count of each label in dataset."""
        return self.dataframe[self.label_column].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Uses inverse frequency: weight = total_samples / (num_classes * class_count)
        
        Returns:
            Tensor of class weights [num_classes]
        """
        label_counts = self.dataframe[self.label_column].value_counts()
        total = len(self.dataframe)
        num_classes = len(self.label_map)
        
        # Calculate weight for each class (in order: Negative, Neutral, Positive)
        weights = []
        for label in ['Negative', 'Neutral', 'Positive']:
            count = label_counts[label]
            weight = total / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: BertTokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    text_column: str = 'cleaned_text',
    label_column: str = 'sentiment',
    num_workers: int = 0
):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: BERT tokenizer
        batch_size: Batch size for DataLoader (default: 8 for RTX 3050)
        max_length: Maximum sequence length
        text_column: Name of text column
        label_column: Name of label column
        num_workers: Number of worker processes (0 = main process only)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    print("="*70)
    print("Creating PyTorch Datasets and DataLoaders")
    print("="*70)
    
    # Create datasets
    train_dataset = ReviewDataset(
        train_df, tokenizer, max_length, text_column, label_column
    )
    print()
    
    val_dataset = ReviewDataset(
        val_df, tokenizer, max_length, text_column, label_column
    )
    print()
    
    test_dataset = ReviewDataset(
        test_df, tokenizer, max_length, text_column, label_column
    )
    print()
    
    # Get class weights from training set
    class_weights = train_dataset.get_class_weights()
    print(f"📊 Class Weights for Training:")
    for label, weight in zip(['Negative', 'Neutral', 'Positive'], class_weights):
        print(f"   {label:8s}: {weight:.3f}")
    print()
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Shuffle training data
        num_workers=num_workers,
        pin_memory=True            # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,             # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,             # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   • Train batches: {len(train_loader):,}")
    print(f"   • Val batches: {len(val_loader):,}")
    print(f"   • Test batches: {len(test_loader):,}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Samples per batch: ~{batch_size}")
    print("="*70)
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test dataset creation
    print("="*70)
    print("Testing ReviewDataset")
    print("="*70)
    
    # Create dummy data
    import pandas as pd
    from transformers import BertTokenizer
    
    dummy_data = pd.DataFrame({
        'cleaned_text': [
            'this phone is amazing i love it',
            'terrible battery life worst purchase ever',
            'it is okay nothing special',
            'great camera quality and display',
        ],
        'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive']
    })
    
    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = ReviewDataset(
        dummy_data,
        tokenizer,
        max_length=128,
        text_column='cleaned_text',
        label_column='sentiment'
    )
    
    # Test __getitem__
    print("\nTesting __getitem__:")
    sample = dataset[0]
    print(f"   Input IDs shape: {sample['input_ids'].shape}")
    print(f"   Attention mask shape: {sample['attention_mask'].shape}")
    print(f"   Label: {sample['labels'].item()} (Positive)")
    
    # Test class weights
    print("\nClass weights:")
    weights = dataset.get_class_weights()
    print(f"   {weights}")
    
    # Test DataLoader
    print("\nTesting DataLoader:")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(loader))
    print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"   Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")
    
    print("\n✅ Dataset test passed!")
    print("="*70)
