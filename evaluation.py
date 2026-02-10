"""
Evaluation metrics for seismic phase picking.
"""

import numpy as np
from typing import Tuple, List, Dict
from scipy.signal import find_peaks


def extract_picks_from_prediction(prediction: np.ndarray, 
                                  threshold: float = 0.5,
                                  min_distance: int = 50) -> np.ndarray:
    """
    Extract pick locations from probabilistic prediction.
    
    Args:
        prediction: Probabilistic prediction array
        threshold: Threshold for peak detection
        min_distance: Minimum distance between peaks in samples
        
    Returns:
        pick_samples: Array of pick sample indices
    """
    # Find peaks above threshold
    peaks, properties = find_peaks(prediction, height=threshold, distance=min_distance)
    
    return peaks


def calculate_true_positives(pred_picks: np.ndarray,
                             true_picks: np.ndarray,
                             tolerance_samples: int = 100) -> Tuple[int, int, int]:
    """
    Calculate true positives, false positives, and false negatives.
    
    Args:
        pred_picks: Predicted pick sample indices
        true_picks: True pick sample indices
        tolerance_samples: Tolerance window in samples (default: 100 for 1 second at 100 Hz)
        
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if len(true_picks) == 0:
        return 0, len(pred_picks), 0
    
    if len(pred_picks) == 0:
        return 0, 0, len(true_picks)
    
    # Match predicted picks to true picks
    matched_true = set()
    matched_pred = set()
    
    for i, pred_pick in enumerate(pred_picks):
        for j, true_pick in enumerate(true_picks):
            if j not in matched_true:
                if abs(pred_pick - true_pick) <= tolerance_samples:
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
    
    true_positives = len(matched_true)
    false_positives = len(pred_picks) - len(matched_pred)
    false_negatives = len(true_picks) - len(matched_true)
    
    return true_positives, false_positives, false_negatives


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Dictionary with precision, recall, and f1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def evaluate_predictions(predictions: np.ndarray,
                        labels: np.ndarray,
                        threshold: float = 0.5,
                        tolerance_samples: int = 100,
                        sample_rate: float = 100.0) -> Dict[str, float]:
    """
    Evaluate predictions against labels.
    
    Args:
        predictions: Predicted probability array of shape (n_samples,) or (batch, n_samples)
        labels: True label array of same shape as predictions
        threshold: Detection threshold
        tolerance_samples: Tolerance for true positive in samples
        sample_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Handle batch predictions
    if predictions.ndim == 2:
        all_tp, all_fp, all_fn = 0, 0, 0
        
        for pred, label in zip(predictions, labels):
            # Extract picks
            pred_picks = extract_picks_from_prediction(pred, threshold)
            true_picks = extract_picks_from_prediction(label, threshold=0.5)
            
            tp, fp, fn = calculate_true_positives(pred_picks, true_picks, tolerance_samples)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        
        metrics = calculate_metrics(all_tp, all_fp, all_fn)
        metrics['tolerance_seconds'] = tolerance_samples / sample_rate
        
        return metrics
    else:
        # Single prediction
        pred_picks = extract_picks_from_prediction(predictions, threshold)
        true_picks = extract_picks_from_prediction(labels, threshold=0.5)
        
        tp, fp, fn = calculate_true_positives(pred_picks, true_picks, tolerance_samples)
        metrics = calculate_metrics(tp, fp, fn)
        metrics['tolerance_seconds'] = tolerance_samples / sample_rate
        
        return metrics


def evaluate_batch(predictions: List[np.ndarray],
                  labels: List[np.ndarray],
                  threshold: float = 0.5,
                  tolerance_samples: int = 100) -> Dict[str, float]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of prediction arrays
        labels: List of label arrays
        threshold: Detection threshold
        tolerance_samples: Tolerance for true positive in samples
        
    Returns:
        Dictionary with aggregated metrics
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for pred, label in zip(predictions, labels):
        pred_picks = extract_picks_from_prediction(pred, threshold)
        true_picks = extract_picks_from_prediction(label, threshold=0.5)
        
        tp, fp, fn = calculate_true_positives(pred_picks, true_picks, tolerance_samples)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    return calculate_metrics(all_tp, all_fp, all_fn)
