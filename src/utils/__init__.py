"""
Utilities Module for SmartReview
=================================

Contains utility functions, dataset classes, and evaluation metrics.
"""

from .dataset import ReviewDataset
from .metrics import compute_metrics, plot_confusion_matrix, save_classification_report

__all__ = ['ReviewDataset', 'compute_metrics', 'plot_confusion_matrix', 'save_classification_report']
