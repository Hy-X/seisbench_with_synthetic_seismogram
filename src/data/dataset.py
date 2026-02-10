"""
Data loading and preprocessing utilities for seismic data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Optional, Tuple, List


class SeismicDataset(Dataset):
    """
    Dataset class for seismic waveform data.
    
    Args:
        data: Numpy array of shape (n_samples, n_channels, n_timesteps)
        labels: Numpy array of shape (n_samples, n_classes, n_timesteps) or None
        transform: Optional transform to apply to the data
    """
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, 
                 transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        sample = torch.from_numpy(sample).float()
        
        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()
            return sample, label
        
        return sample


class SeismicDataLoader:
    """
    Data loader for seismic data from various formats.
    """
    
    @staticmethod
    def load_from_hdf5(filepath: str, data_key: str = 'waveforms', 
                       label_key: str = 'labels') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load seismic data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            data_key: Key for waveform data in HDF5 file
            label_key: Key for labels in HDF5 file
            
        Returns:
            Tuple of (waveforms, labels)
        """
        with h5py.File(filepath, 'r') as f:
            data = f[data_key][:]
            labels = f[label_key][:] if label_key in f else None
        return data, labels
    
    @staticmethod
    def load_from_numpy(data_path: str, label_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load seismic data from numpy files.
        
        Args:
            data_path: Path to numpy file containing waveforms
            label_path: Optional path to numpy file containing labels
            
        Returns:
            Tuple of (waveforms, labels)
        """
        data = np.load(data_path)
        labels = np.load(label_path) if label_path else None
        return data, labels


def normalize_waveform(waveform: np.ndarray, method: str = 'std') -> np.ndarray:
    """
    Normalize seismic waveform.
    
    Args:
        waveform: Input waveform of shape (n_channels, n_timesteps)
        method: Normalization method ('std', 'minmax', or 'max')
        
    Returns:
        Normalized waveform
    """
    if method == 'std':
        mean = np.mean(waveform, axis=-1, keepdims=True)
        std = np.std(waveform, axis=-1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (waveform - mean) / std
    elif method == 'minmax':
        min_val = np.min(waveform, axis=-1, keepdims=True)
        max_val = np.max(waveform, axis=-1, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (waveform - min_val) / range_val
    elif method == 'max':
        max_val = np.max(np.abs(waveform), axis=-1, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)
        return waveform / max_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_synthetic_data(n_samples: int = 100, n_channels: int = 3, 
                         n_timesteps: int = 3000, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic seismic data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of channels (e.g., 3 for E, N, Z components)
        n_timesteps: Number of time steps
        n_classes: Number of output classes
        
    Returns:
        Tuple of (data, labels)
    """
    # Generate random waveforms
    data = np.random.randn(n_samples, n_channels, n_timesteps).astype(np.float32)
    
    # Generate random phase labels (simplified)
    labels = np.zeros((n_samples, n_classes, n_timesteps), dtype=np.float32)
    
    for i in range(n_samples):
        # Simulate P-wave arrival
        p_arrival = np.random.randint(500, 1500)
        labels[i, 1, p_arrival:p_arrival+200] = 1.0
        
        # Simulate S-wave arrival (after P-wave)
        s_arrival = p_arrival + np.random.randint(300, 800)
        if s_arrival + 200 < n_timesteps:
            labels[i, 2, s_arrival:s_arrival+200] = 1.0
        
        # Rest is noise
        labels[i, 0, :] = 1.0 - labels[i, 1, :] - labels[i, 2, :]
    
    return data, labels


def get_dataloader(dataset: Dataset, batch_size: int = 32, 
                   shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
