"""
Label generation using Gaussian probabilistic labels.
Compatible with SeisBench labeling approach.
"""

import numpy as np
from typing import List, Optional, Union


def generate_gaussian_label(length: int, 
                            pick_samples: Union[int, List[int]], 
                            sigma: float = 50.0) -> np.ndarray:
    """
    Generate Gaussian probabilistic label for phase picks.
    
    Args:
        length: Length of the time series in samples
        pick_samples: Sample index(es) of phase arrival(s)
        sigma: Standard deviation of Gaussian in samples (default: 50)
        
    Returns:
        label: Probabilistic label array of shape (length,)
    """
    label = np.zeros(length, dtype=np.float32)
    
    # Handle single pick or list of picks
    if isinstance(pick_samples, (int, float)):
        pick_samples = [int(pick_samples)]
    
    # Generate Gaussian for each pick
    for pick in pick_samples:
        if 0 <= pick < length:
            # Create Gaussian centered at pick location
            x = np.arange(length)
            gaussian = np.exp(-0.5 * ((x - pick) / sigma) ** 2)
            label = np.maximum(label, gaussian)
    
    return label


def generate_labels_from_catalog(length: int,
                                 picks_dict: dict,
                                 sigma: float = 50.0,
                                 sample_rate: float = 100.0) -> dict:
    """
    Generate labels from a catalog of picks.
    
    Args:
        length: Length of the time series in samples
        picks_dict: Dictionary with phase types as keys and pick times (in seconds) as values
                   Example: {'P': [5.2, 8.7], 'S': [7.8, 12.3]}
        sigma: Standard deviation of Gaussian in samples
        sample_rate: Sampling rate in Hz
        
    Returns:
        labels_dict: Dictionary with phase types as keys and label arrays as values
    """
    labels_dict = {}
    
    for phase, pick_times in picks_dict.items():
        # Convert times to samples
        pick_samples = [int(t * sample_rate) for t in pick_times]
        labels_dict[phase] = generate_gaussian_label(length, pick_samples, sigma)
    
    return labels_dict


def generate_detection_label(length: int,
                             pick_samples: Union[int, List[int]],
                             sigma: float = 50.0) -> np.ndarray:
    """
    Generate detection label (any phase arrival).
    This is a simplified version that generates a single label for detection.
    
    Args:
        length: Length of the time series in samples
        pick_samples: Sample index(es) of any phase arrival(s)
        sigma: Standard deviation of Gaussian in samples
        
    Returns:
        label: Detection label array of shape (length,)
    """
    return generate_gaussian_label(length, pick_samples, sigma)


class GaussianLabelGenerator:
    """
    SeisBench-compatible Gaussian label generator.
    """
    
    def __init__(self, sigma: float = 50.0, sample_rate: float = 100.0):
        """
        Initialize label generator.
        
        Args:
            sigma: Standard deviation of Gaussian in samples
            sample_rate: Sampling rate in Hz
        """
        self.sigma = sigma
        self.sample_rate = sample_rate
    
    def __call__(self, length: int, picks: Union[List[float], dict]) -> np.ndarray:
        """
        Generate labels from picks.
        
        Args:
            length: Length of the time series in samples
            picks: Either list of pick times in seconds, or dict of phase picks
            
        Returns:
            labels: Label array
        """
        if isinstance(picks, dict):
            # Multiple phases - combine all picks
            all_picks = []
            for phase_picks in picks.values():
                all_picks.extend(phase_picks)
            pick_samples = [int(t * self.sample_rate) for t in all_picks]
        else:
            # Single phase or list of times
            pick_samples = [int(t * self.sample_rate) for t in picks]
        
        return generate_gaussian_label(length, pick_samples, self.sigma)
