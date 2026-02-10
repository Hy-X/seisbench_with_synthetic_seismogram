"""
Data loader for miniSEED files.
Handles reading 20s sequences at 100 Hz and preprocessing.
"""

import numpy as np
from obspy import read
from typing import Tuple, Optional, List
import warnings


class MiniSEEDLoader:
    """Loader for miniSEED seismic data files."""
    
    def __init__(self, target_sample_rate: int = 100, window_length: float = 20.0):
        """
        Initialize the miniSEED loader.
        
        Args:
            target_sample_rate: Target sampling rate in Hz (default: 100)
            window_length: Window length in seconds (default: 20.0)
        """
        self.target_sample_rate = target_sample_rate
        self.window_length = window_length
        self.n_samples = int(target_sample_rate * window_length)
    
    def read_mseed(self, filepath: str, normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Read miniSEED file and return 3-component seismic data.
        
        Args:
            filepath: Path to miniSEED file
            normalize: Whether to normalize each channel
            
        Returns:
            data: Array of shape (n_channels, n_samples)
            channel_names: List of channel names
        """
        try:
            stream = read(filepath)
            
            # Resample if needed
            for trace in stream:
                if trace.stats.sampling_rate != self.target_sample_rate:
                    trace.resample(self.target_sample_rate)
            
            # Extract data from each channel
            data_list = []
            channel_names = []
            
            for trace in stream:
                # Trim or pad to target length
                trace_data = trace.data[:self.n_samples]
                if len(trace_data) < self.n_samples:
                    trace_data = np.pad(trace_data, (0, self.n_samples - len(trace_data)), 
                                       mode='constant', constant_values=0)
                
                if normalize:
                    # Normalize to zero mean and unit std
                    mean = np.mean(trace_data)
                    std = np.std(trace_data)
                    if std > 0:
                        trace_data = (trace_data - mean) / std
                
                data_list.append(trace_data)
                channel_names.append(f"{trace.stats.station}.{trace.stats.channel}")
            
            # Stack to shape (n_channels, n_samples)
            data = np.stack(data_list, axis=0)
            
            return data, channel_names
            
        except Exception as e:
            warnings.warn(f"Error reading {filepath}: {str(e)}")
            raise
    
    def create_windows(self, data: np.ndarray, stride: Optional[int] = None) -> np.ndarray:
        """
        Create overlapping windows from continuous data.
        
        Args:
            data: Array of shape (n_channels, n_samples_total)
            stride: Stride for sliding window (default: n_samples // 2)
            
        Returns:
            windows: Array of shape (n_windows, n_channels, n_samples)
        """
        if stride is None:
            stride = self.n_samples // 2
        
        n_channels, n_samples_total = data.shape
        windows = []
        
        for start in range(0, n_samples_total - self.n_samples + 1, stride):
            window = data[:, start:start + self.n_samples]
            windows.append(window)
        
        return np.array(windows)
