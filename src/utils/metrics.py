"""
Evaluation Metrics for SmartReview
===================================

Functions for computing and visualizing model performance metrics.

Author: SmartReview Team
Date: October 29, 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple
import pandas as pd


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive']
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class labels [num_samples]
        labels: True class labels [num_samples]
        label_names: Names of classes
    
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - macro_f1: Macro-averaged F1 score
            - weighted_f1: Weighted F1 score
            - per_class metrics: Precision, recall, F1 for each class
    """
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro-averaged metrics (equal weight for each class)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Weighted metrics (weighted by class frequency)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Build results dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # Add per-class metrics
    for idx, label_name in enumerate(label_names):
        metrics[f'{label_name.lower()}_precision'] = precision[idx]
        metrics[f'{label_name.lower()}_recall'] = recall[idx]
        metrics[f'{label_name.lower()}_f1'] = f1[idx]
        metrics[f'{label_name.lower()}_support'] = support[idx]
    
    return metrics


def print_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    label_names: List[str] = ['Negative', 'Neutral', 'Positive']
):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics()
        prefix: Prefix for print statements (e.g., "Train" or "Val")
        label_names: Names of classes
    """
    print(f"\n{'='*70}")
    print(f"{prefix} Metrics:")
    print(f"{'='*70}")
    
    # Overall metrics
    print(f"\n📊 Overall Performance:")
    print(f"   Accuracy:         {metrics['accuracy']:.4f}")
    print(f"   Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"   Weighted F1:      {metrics['weighted_f1']:.4f}")
    
    # Per-class metrics
    print(f"\n📈 Per-Class Performance:")
    print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"   {'-'*60}")
    
    for label_name in label_names:
        label_lower = label_name.lower()
        precision = metrics[f'{label_lower}_precision']
        recall = metrics[f'{label_lower}_recall']
        f1 = metrics[f'{label_lower}_f1']
        support = int(metrics[f'{label_lower}_support'])
        
        print(f"   {label_name:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10,}")
    
    print(f"{'='*70}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    save_path: str = None,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        label_names: Names of classes
        save_path: Path to save figure (optional)
        title: Title for the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Normalize by row (true labels) to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot absolute counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    
    # Plot normalized percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax2,
        cbar_kws={'label': 'Percentage'}
    )
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title} (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to {save_path}")
    
    plt.show()


def save_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    save_path: str = None
) -> str:
    """
    Generate and save detailed classification report.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        label_names: Names of classes
        save_path: Path to save report (optional)
    
    Returns:
        Classification report as string
    """
    # Generate report
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=4
    )
    
    # Print to console
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(report)
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("="*70 + "\n")
            f.write(report)
        print(f"💾 Classification report saved to {save_path}")
    
    return report


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: str = None
):
    """
    Plot training history curves.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accuracies: Training accuracy per epoch
        val_accuracies: Validation accuracy per epoch
        save_path: Path to save figure (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('📉 Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('📈 Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training curves saved to {save_path}")
    
    plt.show()


def plot_per_class_f1(
    metrics: Dict[str, float],
    label_names: List[str] = ['Negative', 'Neutral', 'Positive'],
    save_path: str = None
):
    """
    Plot per-class F1 scores as bar chart.
    
    Args:
        metrics: Metrics dictionary from compute_metrics()
        label_names: Names of classes
        save_path: Path to save figure (optional)
    """
    # Extract F1 scores
    f1_scores = [metrics[f'{label.lower()}_f1'] for label in label_names]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Red, Yellow, Green
    bars = ax.bar(label_names, f1_scores, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('📊 Per-Class F1 Scores', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line for macro average
    macro_f1 = metrics['macro_f1']
    ax.axhline(y=macro_f1, color='gray', linestyle='--', linewidth=2, label=f'Macro F1: {macro_f1:.3f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 F1 scores chart saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test metrics functions
    print("="*70)
    print("Testing Metrics Module")
    print("="*70)
    
    # Create dummy predictions and labels
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=1000, p=[0.25, 0.07, 0.68])
    predictions = labels.copy()
    
    # Add some errors
    error_indices = np.random.choice(len(labels), size=200, replace=False)
    predictions[error_indices] = np.random.choice([0, 1, 2], size=200)
    
    # Test compute_metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels)
    print_metrics(metrics, prefix="Test")
    
    # Test confusion matrix plot
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(predictions, labels, title='Test Confusion Matrix')
    
    # Test classification report
    print("\nGenerating classification report...")
    save_classification_report(predictions, labels)
    
    print("\n✅ Metrics module test passed!")
    print("="*70)
