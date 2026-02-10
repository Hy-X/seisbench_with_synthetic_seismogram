"""
Visualization functions for seismic data, labels, and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from evaluation import extract_picks_from_prediction


def plot_waveform_with_predictions(data: np.ndarray,
                                   label: np.ndarray,
                                   prediction: np.ndarray,
                                   channel_names: Optional[List[str]] = None,
                                   sample_rate: float = 100.0,
                                   threshold: float = 0.5,
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None):
    """
    Plot 3-channel seismic data with labels and predictions.
    
    Args:
        data: Waveform data of shape (n_channels, n_samples)
        label: True labels of shape (n_samples,)
        prediction: Model predictions of shape (n_samples,)
        channel_names: Names of channels (default: ['Channel 0', 'Channel 1', 'Channel 2'])
        sample_rate: Sampling rate in Hz
        threshold: Threshold for pick detection
        figsize: Figure size
        save_path: Path to save figure (if None, displays instead)
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / sample_rate
    
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(n_channels)]
    
    # Extract picks
    true_picks = extract_picks_from_prediction(label, threshold=0.5)
    pred_picks = extract_picks_from_prediction(prediction, threshold=threshold)
    
    # Create figure
    fig, axes = plt.subplots(n_channels + 2, 1, figsize=figsize, sharex=True)
    
    # Plot waveforms
    for i in range(n_channels):
        axes[i].plot(time, data[i], 'k-', linewidth=0.5)
        axes[i].set_ylabel(channel_names[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Mark true picks
        for pick in true_picks:
            axes[i].axvline(pick / sample_rate, color='green', linestyle='--', 
                          linewidth=1.5, alpha=0.7, label='True Pick' if i == 0 else '')
        
        # Mark predicted picks
        for pick in pred_picks:
            axes[i].axvline(pick / sample_rate, color='red', linestyle='-', 
                          linewidth=1.5, alpha=0.7, label='Predicted Pick' if i == 0 else '')
        
        if i == 0 and (len(true_picks) > 0 or len(pred_picks) > 0):
            axes[i].legend(loc='upper right', fontsize=8)
    
    # Plot true label
    axes[n_channels].plot(time, label, 'g-', linewidth=1.5, label='True Label')
    axes[n_channels].fill_between(time, 0, label, color='green', alpha=0.3)
    axes[n_channels].set_ylabel('True\nProbability', fontsize=10)
    axes[n_channels].set_ylim([-0.05, 1.05])
    axes[n_channels].grid(True, alpha=0.3)
    axes[n_channels].legend(loc='upper right', fontsize=8)
    
    # Plot prediction
    axes[n_channels + 1].plot(time, prediction, 'r-', linewidth=1.5, label='Prediction')
    axes[n_channels + 1].fill_between(time, 0, prediction, color='red', alpha=0.3)
    axes[n_channels + 1].axhline(threshold, color='orange', linestyle=':', 
                                 linewidth=1, label=f'Threshold ({threshold})')
    axes[n_channels + 1].set_ylabel('Predicted\nProbability', fontsize=10)
    axes[n_channels + 1].set_ylim([-0.05, 1.05])
    axes[n_channels + 1].set_xlabel('Time (s)', fontsize=10)
    axes[n_channels + 1].grid(True, alpha=0.3)
    axes[n_channels + 1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_examples(data_list: List[np.ndarray],
                          label_list: List[np.ndarray],
                          prediction_list: List[np.ndarray],
                          n_examples: int = 3,
                          sample_rate: float = 100.0,
                          threshold: float = 0.5,
                          save_prefix: Optional[str] = None):
    """
    Plot multiple examples of waveforms with predictions.
    
    Args:
        data_list: List of waveform arrays
        label_list: List of label arrays
        prediction_list: List of prediction arrays
        n_examples: Number of examples to plot
        sample_rate: Sampling rate in Hz
        threshold: Detection threshold
        save_prefix: Prefix for saved figure filenames
    """
    n_examples = min(n_examples, len(data_list))
    
    for i in range(n_examples):
        save_path = None
        if save_prefix:
            save_path = f"{save_prefix}_example_{i+1}.png"
        
        plot_waveform_with_predictions(
            data_list[i],
            label_list[i],
            prediction_list[i],
            sample_rate=sample_rate,
            threshold=threshold,
            save_path=save_path
        )


def plot_statistics(metrics: dict, save_path: Optional[str] = None):
    """
    Plot statistics and performance metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot metrics bar chart
    metric_names = ['Precision', 'Recall', 'F1 Score']
    metric_values = [
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0)
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = axes[0].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot confusion matrix style
    tp = metrics.get('true_positives', 0)
    fp = metrics.get('false_positives', 0)
    fn = metrics.get('false_negatives', 0)
    
    categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
    counts = [tp, fp, fn]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = axes[1].bar(categories, counts, color=colors, alpha=0.7)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Detection Statistics', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}',
                    ha='center', va='bottom', fontsize=10)
    
    # Add tolerance info
    tolerance = metrics.get('tolerance_seconds', 1.0)
    fig.suptitle(f'Model Performance (Tolerance: {tolerance:.1f}s)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_statistics(metrics: dict):
    """
    Print statistics in a formatted way.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL PERFORMANCE STATISTICS")
    print("="*50)
    print(f"Tolerance: {metrics.get('tolerance_seconds', 1.0):.1f} seconds")
    print("-"*50)
    print(f"True Positives:  {metrics.get('true_positives', 0):>6d}")
    print(f"False Positives: {metrics.get('false_positives', 0):>6d}")
    print(f"False Negatives: {metrics.get('false_negatives', 0):>6d}")
    print("-"*50)
    print(f"Precision:       {metrics.get('precision', 0):>6.3f}")
    print(f"Recall:          {metrics.get('recall', 0):>6.3f}")
    print(f"F1 Score:        {metrics.get('f1_score', 0):>6.3f}")
    print("="*50 + "\n")
