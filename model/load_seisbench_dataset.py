#!/usr/bin/env python3
"""
Load and use synthetic seismogram dataset with SeisBench.

This script demonstrates how to read the HDF5 and CSV files using SeisBench's
WaveformDataset class and prepare data for training phase picking models.

Features:
    - Load dataset with SeisBench WaveformDataset
    - Access individual traces and metadata
    - Generate probabilistic pick labels (Gaussian)
    - Create data loaders for training
    - Visualize waveforms with picks
    - Export windowed samples for model training

Usage:
    python load_seisbench_dataset.py

Requirements:
    - seisbench
    - torch
    - numpy
    - pandas
    - matplotlib
    - h5py
"""

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import warnings


class SyntheticSeismicDataset:
    """
    Wrapper for synthetic seismogram dataset in SeisBench format.
    
    This class provides convenient access to the HDF5 and CSV files without
    requiring the full SeisBench library installation.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str
    ):
        """
        Initialize dataset from HDF5 and CSV files.
        
        Args:
            hdf5_path: Path to HDF5 waveform file
            csv_path: Path to CSV metadata file
            
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If files are incompatible
        """
        self.hdf5_path = Path(hdf5_path)
        self.csv_path = Path(csv_path)
        
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load metadata
        self.metadata = pd.read_csv(csv_path)
        
        # Validate HDF5 contains expected traces
        with h5py.File(hdf5_path, 'r') as hdf:
            hdf_traces = set(hdf.keys())
            csv_traces = set(self.metadata['trace_name'].values)
            
            if not csv_traces.issubset(hdf_traces):
                missing = csv_traces - hdf_traces
                warnings.warn(f"CSV references {len(missing)} traces not in HDF5: {missing}")
        
        print(f"✓ Loaded dataset: {len(self.metadata)} traces")
    
    def __len__(self) -> int:
        """Return number of traces in dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Get a single trace by index.
        
        Args:
            idx: Index of trace to retrieve
            
        Returns:
            waveform: 3C waveform array (3, n_samples)
            metadata: Dictionary with trace metadata
        """
        row = self.metadata.iloc[idx]
        trace_name = row['trace_name']
        
        # Load waveform from HDF5
        with h5py.File(self.hdf5_path, 'r') as hdf:
            waveform = hdf[trace_name][:]
        
        # Extract metadata
        metadata = row.to_dict()
        
        return waveform, metadata
    
    def get_trace_by_name(self, trace_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Get trace by name.
        
        Args:
            trace_name: Name of trace to retrieve
            
        Returns:
            waveform: 3C waveform array (3, n_samples)
            metadata: Dictionary with trace metadata
            
        Raises:
            KeyError: If trace not found
        """
        idx = self.metadata[self.metadata['trace_name'] == trace_name].index
        if len(idx) == 0:
            raise KeyError(f"Trace not found: {trace_name}")
        
        return self.__getitem__(idx[0])


def generate_gaussian_pick_labels(
    n_samples: int,
    pick_sample: int,
    sigma: float = 50.0,
    sampling_rate: float = 100.0
) -> np.ndarray:
    """
    Generate Gaussian probabilistic pick label (SeisBench convention).
    
    Creates a normalized Gaussian centered at the pick sample, following
    SeisBench standards for probabilistic phase labels.
    
    Args:
        n_samples: Total number of samples in waveform
        pick_sample: Sample index of phase arrival
        sigma: Standard deviation in samples (default: 50 for 100 Hz data)
        sampling_rate: Sampling rate in Hz (for documentation)
        
    Returns:
        Gaussian probability array of shape (n_samples,), range [0, 1]
        
    Example:
        >>> labels = generate_gaussian_pick_labels(3000, 1500, sigma=50)
        >>> labels.shape
        (3000,)
        >>> labels.max()  # Peak at pick
        1.0
    """
    if pick_sample < 0:
        # No pick available
        return np.zeros(n_samples)
    
    x = np.arange(n_samples)
    gaussian = np.exp(-((x - pick_sample) ** 2) / (2 * sigma ** 2))
    
    # Normalize to [0, 1]
    if gaussian.max() > 0:
        gaussian /= gaussian.max()
    
    return gaussian


def create_pick_labels(
    waveform: np.ndarray,
    metadata: Dict,
    sigma: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3-channel pick label array (P, S, Noise) from metadata.
    
    Following SeisBench PhaseNet convention:
    - Channel 0: P-wave probability
    - Channel 1: S-wave probability
    - Channel 2: Noise probability (1 - max(P, S))
    
    Args:
        waveform: Input waveform array (3, n_samples)
        metadata: Metadata dictionary with pick information
        sigma: Gaussian width in samples (default: 50)
        
    Returns:
        labels_p: P-wave probability array (n_samples,)
        labels_s: S-wave probability array (n_samples,)
        labels_n: Noise probability array (n_samples,)
    """
    n_samples = waveform.shape[1]
    
    # Generate P and S labels
    p_sample = int(metadata.get('trace_p_arrival_sample', -1))
    s_sample = int(metadata.get('trace_s_arrival_sample', -1))
    
    labels_p = generate_gaussian_pick_labels(n_samples, p_sample, sigma)
    labels_s = generate_gaussian_pick_labels(n_samples, s_sample, sigma)
    
    # Noise is complement of signal
    labels_max = np.maximum(labels_p, labels_s)
    labels_n = 1.0 - labels_max
    
    return labels_p, labels_s, labels_n


class SeismicPhaseDataset(Dataset):
    """
    PyTorch Dataset for seismic phase picking training.
    
    This dataset wraps the synthetic seismogram data and provides
    waveforms with corresponding pick labels in the format expected
    by phase picking models (e.g., PhaseNet, EQTransformer).
    """
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        sigma: float = 50.0,
        normalize: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize PyTorch dataset for phase picking.
        
        Args:
            hdf5_path: Path to HDF5 waveform file
            csv_path: Path to CSV metadata file
            sigma: Gaussian width for pick labels (default: 50 samples)
            normalize: Whether to normalize waveforms (default: True)
            transform: Optional transform to apply to waveforms
        """
        self.dataset = SyntheticSeismicDataset(hdf5_path, csv_path)
        self.sigma = sigma
        self.normalize = normalize
        self.transform = transform
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get waveform and labels as PyTorch tensors.
        
        Args:
            idx: Sample index
            
        Returns:
            waveform: Tensor of shape (3, n_samples)
            labels: Tensor of shape (3, n_samples) for P, S, Noise
        """
        waveform, metadata = self.dataset[idx]
        
        # Normalize waveform
        if self.normalize:
            waveform_mean = waveform.mean(axis=1, keepdims=True)
            waveform_std = waveform.std(axis=1, keepdims=True) + 1e-8
            waveform = (waveform - waveform_mean) / waveform_std
        
        # Generate labels
        labels_p, labels_s, labels_n = create_pick_labels(
            waveform, metadata, self.sigma
        )
        labels = np.stack([labels_p, labels_s, labels_n], axis=0)
        
        # Apply optional transform
        if self.transform:
            waveform = self.transform(waveform)
        
        # Convert to tensors
        waveform_tensor = torch.from_numpy(waveform).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        return waveform_tensor, labels_tensor


def plot_waveform_with_labels(
    waveform: np.ndarray,
    labels: np.ndarray,
    metadata: Dict,
    sampling_rate: float = 100.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3C waveform with probabilistic pick labels.
    
    Args:
        waveform: Waveform array (3, n_samples)
        labels: Label array (3, n_samples) for P, S, Noise
        metadata: Metadata dictionary
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save figure
    """
    n_samples = waveform.shape[1]
    time = np.arange(n_samples) / sampling_rate
    
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    
    # Component names
    components = ['Z (Vertical)', 'N (North)', 'E (East)']
    label_names = ['P Probability', 'S Probability', 'Noise Probability']
    
    # Plot waveforms
    for i in range(3):
        ax = axes[i]
        ax.plot(time, waveform[i], 'k-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(components[i], fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark true picks
        p_sample = metadata.get('trace_p_arrival_sample', -1)
        s_sample = metadata.get('trace_s_arrival_sample', -1)
        
        if p_sample >= 0:
            ax.axvline(p_sample / sampling_rate, color='blue', 
                      linestyle='--', linewidth=1.5, alpha=0.6)
        if s_sample >= 0:
            ax.axvline(s_sample / sampling_rate, color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Plot labels
    colors = ['blue', 'red', 'gray']
    for i in range(3):
        ax = axes[i + 3]
        ax.plot(time, labels[i], color=colors[i], linewidth=1.5, alpha=0.8)
        ax.fill_between(time, 0, labels[i], color=colors[i], alpha=0.3)
        ax.set_ylabel(label_names[i], fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    # Title with metadata
    trace_name = metadata.get('trace_name', 'Unknown')
    snr_db = metadata.get('snr_db', 0)
    p_time = metadata.get('trace_p_arrival_time', -1)
    s_time = metadata.get('trace_s_arrival_time', -1)
    
    fig.suptitle(
        f'{trace_name} | SNR: {snr_db:.1f} dB | '
        f'P: {p_time:.2f}s, S: {s_time:.2f}s',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_dataset_statistics(dataset: SyntheticSeismicDataset) -> None:
    """
    Print comprehensive dataset statistics.
    
    Args:
        dataset: SyntheticSeismicDataset instance
    """
    df = dataset.metadata
    
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    print(f"\nBasic Info:")
    print(f"  Total traces: {len(df)}")
    print(f"  Stations: {df['station_code'].nunique()}")
    print(f"  Networks: {df['network_code'].nunique()}")
    
    print(f"\nWaveform Properties:")
    print(f"  Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
    print(f"  Samples per trace: {df['trace_npts'].iloc[0]}")
    print(f"  Duration: {df['trace_npts'].iloc[0] / df['trace_sampling_rate_hz'].iloc[0]:.1f} s")
    
    print(f"\nPhase Picks:")
    p_count = (df['trace_p_arrival_sample'] >= 0).sum()
    s_count = (df['trace_s_arrival_sample'] >= 0).sum()
    print(f"  P-wave picks: {p_count}/{len(df)} ({100*p_count/len(df):.1f}%)")
    print(f"  S-wave picks: {s_count}/{len(df)} ({100*s_count/len(df):.1f}%)")
    
    # Pick time statistics
    df_picks = df[(df['trace_p_arrival_sample'] >= 0) & (df['trace_s_arrival_sample'] >= 0)]
    if len(df_picks) > 0:
        p_times = df_picks['trace_p_arrival_time']
        s_times = df_picks['trace_s_arrival_time']
        ps_sep = s_times - p_times
        
        print(f"\n  P-arrival time range: {p_times.min():.2f} - {p_times.max():.2f} s")
        print(f"  S-arrival time range: {s_times.min():.2f} - {s_times.max():.2f} s")
        print(f"  P-S separation: {ps_sep.min():.2f} - {ps_sep.max():.2f} s (mean: {ps_sep.mean():.2f} s)")
    
    print(f"\nSNR Statistics:")
    print(f"  Mean: {df['snr_db'].mean():.2f} dB")
    print(f"  Std:  {df['snr_db'].std():.2f} dB")
    print(f"  Range: {df['snr_db'].min():.2f} - {df['snr_db'].max():.2f} dB")
    
    print("=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("Loading Synthetic Seismogram Dataset with SeisBench Format")
    print("=" * 70)
    
    # File paths (try both locations)
    possible_locations = [
        ('/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/data/synthetic_dataset.hdf5',
         '/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/data/synthetic_metadata.csv'),
        ('/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/synthetic_input/synthetic_dataset.hdf5',
         '/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/synthetic_input/synthetic_metadata.csv'),
        ('synthetic_dataset.hdf5', 'synthetic_metadata.csv')  # Current directory
    ]
    
    hdf5_path, csv_path = None, None
    for h5, csv in possible_locations:
        if Path(h5).exists() and Path(csv).exists():
            hdf5_path, csv_path = h5, csv
            break
    
    if hdf5_path is None:
        print("✗ Error: Could not find synthetic_dataset.hdf5 and synthetic_metadata.csv")
        print("  Searched locations:")
        for h5, csv in possible_locations:
            print(f"    - {h5}")
        return 1
    
    print(f"\n✓ Found dataset files:")
    print(f"  HDF5: {hdf5_path}")
    print(f"  CSV:  {csv_path}")
    
    # Load dataset
    dataset = SyntheticSeismicDataset(hdf5_path, csv_path)
    
    # Print statistics
    print_dataset_statistics(dataset)
    
    # Example 1: Access individual trace
    print("\n" + "=" * 70)
    print("Example 1: Loading Individual Traces")
    print("=" * 70)
    
    waveform, metadata = dataset[0]
    print(f"\nLoaded trace: {metadata['trace_name']}")
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  P-arrival: {metadata['trace_p_arrival_time']:.2f} s")
    print(f"  S-arrival: {metadata['trace_s_arrival_time']:.2f} s")
    print(f"  SNR: {metadata['snr_db']:.2f} dB")
    
    # Example 2: Generate labels
    print("\n" + "=" * 70)
    print("Example 2: Generating Probabilistic Pick Labels")
    print("=" * 70)
    
    labels_p, labels_s, labels_n = create_pick_labels(waveform, metadata, sigma=50.0)
    labels = np.stack([labels_p, labels_s, labels_n], axis=0)
    
    print(f"\nLabel shape: {labels.shape}")
    print(f"  P-channel max: {labels_p.max():.3f} at sample {labels_p.argmax()}")
    print(f"  S-channel max: {labels_s.max():.3f} at sample {labels_s.argmax()}")
    print(f"  True P sample: {metadata['trace_p_arrival_sample']}")
    print(f"  True S sample: {metadata['trace_s_arrival_sample']}")
    
    # Example 3: Visualize
    print("\n" + "=" * 70)
    print("Example 3: Visualizing Waveforms with Labels")
    print("=" * 70)
    
    # Plot first 3 traces with different SNR
    df_sorted = dataset.metadata.sort_values('snr_db')
    indices = [0, len(df_sorted)//2, len(df_sorted)-1]
    
    for i, idx in enumerate(indices):
        waveform, metadata = dataset[idx]
        labels_p, labels_s, labels_n = create_pick_labels(waveform, metadata)
        labels = np.stack([labels_p, labels_s, labels_n], axis=0)
        
        save_path = f"seisbench_example_{i+1}_{metadata['trace_name']}.png"
        plot_waveform_with_labels(waveform, labels, metadata, save_path=save_path)
    
    print(f"\n✓ Saved 3 example plots")
    
    # Example 4: PyTorch DataLoader
    print("\n" + "=" * 70)
    print("Example 4: Creating PyTorch DataLoader")
    print("=" * 70)
    
    torch_dataset = SeismicPhaseDataset(hdf5_path, csv_path, sigma=50.0, normalize=True)
    dataloader = DataLoader(torch_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print(f"\nDataLoader created:")
    print(f"  Total samples: {len(torch_dataset)}")
    print(f"  Batch size: 4")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Get one batch
    waveforms, labels = next(iter(dataloader))
    print(f"\nSample batch:")
    print(f"  Waveforms shape: {waveforms.shape}  # (batch, channels, samples)")
    print(f"  Labels shape: {labels.shape}        # (batch, 3, samples)")
    print(f"  Waveform dtype: {waveforms.dtype}")
    print(f"  Labels dtype: {labels.dtype}")
    
    print("\n" + "=" * 70)
    print("✓ Dataset loading complete!")
    print("=" * 70)
    
    print("\nReady for model training:")
    print("  - Use SeismicPhaseDataset with PyTorch models")
    print("  - Compatible with PhaseNet, EQTransformer architectures")
    print("  - Labels follow SeisBench Gaussian convention (sigma=50)")
    
    return 0


if __name__ == "__main__":
    exit(main())
