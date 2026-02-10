#!/usr/bin/env python3
"""
Test loading and visualizing SeisBench format synthetic dataset.

This script demonstrates how to load the packed synthetic dataset
and visualize sample waveforms with phase picks.

Usage:
    python test_seisbench_dataset.py
"""

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def load_trace_from_hdf5(
    hdf5_file: str,
    trace_name: str
) -> Tuple[np.ndarray, dict]:
    """
    Load a single trace from HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        trace_name: Name of trace to load
        
    Returns:
        data: Waveform array (3, n_samples)
        attrs: Dictionary of trace attributes
    """
    with h5py.File(hdf5_file, 'r') as hdf:
        data = hdf[trace_name][:]
        attrs = dict(hdf[trace_name].attrs)
    
    return data, attrs


def plot_trace_with_picks(
    data: np.ndarray,
    metadata: dict,
    sampling_rate: float = 100.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3-component waveform with P and S picks.
    
    Args:
        data: Waveform array (3, n_samples)
        metadata: Dictionary with trace metadata
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save figure
    """
    n_samples = data.shape[1]
    time = np.arange(n_samples) / sampling_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    components = ['Vertical (Z)', 'North (N)', 'East (E)']
    colors = ['k', 'k', 'k']
    
    p_sample = metadata.get('trace_p_arrival_sample', -1)
    s_sample = metadata.get('trace_s_arrival_sample', -1)
    
    for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
        # Plot waveform
        ax.plot(time, data[i], color=color, linewidth=0.6, alpha=0.8)
        
        # Mark P and S arrivals
        if p_sample >= 0:
            p_time = p_sample / sampling_rate
            ax.axvline(p_time, color='blue', linestyle='--', linewidth=2, 
                      alpha=0.7, label='P-arrival')
        
        if s_sample >= 0:
            s_time = s_sample / sampling_rate
            ax.axvline(s_time, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label='S-arrival')
        
        # Formatting
        ax.set_ylabel(comp, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1])
        
        if i == 0:
            ax.legend(loc='upper right')
    
    # Title with metadata
    snr_db = metadata.get('snr_db', 0)
    trace_name = metadata.get('trace_name', 'Unknown')
    
    fig.suptitle(
        f'{trace_name} | SNR: {snr_db:.1f} dB | '
        f'P: {p_sample/sampling_rate:.2f}s, S: {s_sample/sampling_rate:.2f}s',
        fontsize=13, fontweight='bold'
    )
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Test loading and visualizing the dataset."""
    print("=" * 70)
    print("Testing SeisBench Format Dataset")
    print("=" * 70)
    
    hdf5_file = 'synthetic_dataset.hdf5'
    csv_file = 'synthetic_metadata.csv'
    
    # Load metadata
    df = pd.read_csv(csv_file)
    print(f"\n✓ Loaded metadata: {len(df)} traces")
    
    # Show summary
    print(f"\nDataset Overview:")
    print(f"  Traces: {len(df)}")
    print(f"  Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
    print(f"  Duration: {df['trace_npts'].iloc[0] / df['trace_sampling_rate_hz'].iloc[0]:.1f} s")
    print(f"  SNR range: {df['snr_db'].min():.1f} - {df['snr_db'].max():.1f} dB")
    
    # Select a few traces to visualize
    # Pick low, medium, high SNR examples
    df_sorted = df.sort_values('snr_db')
    examples = [
        df_sorted.iloc[0],      # Low SNR
        df_sorted.iloc[len(df)//2],  # Medium SNR
        df_sorted.iloc[-1]      # High SNR
    ]
    
    print(f"\nVisualizing 3 example traces...")
    
    for i, row in enumerate(examples):
        trace_name = row['trace_name']
        
        # Load trace from HDF5
        data, attrs = load_trace_from_hdf5(hdf5_file, trace_name)
        
        # Combine metadata
        metadata = {
            'trace_name': trace_name,
            'trace_p_arrival_sample': int(row['trace_p_arrival_sample']),
            'trace_s_arrival_sample': int(row['trace_s_arrival_sample']),
            'snr_db': row['snr_db']
        }
        
        # Plot
        save_path = f'test_trace_{i+1}_{trace_name}.png'
        plot_trace_with_picks(
            data, 
            metadata, 
            sampling_rate=row['trace_sampling_rate_hz'],
            save_path=save_path
        )
        
        print(f"  [{i+1}] {trace_name}: SNR={row['snr_db']:.1f} dB")
    
    print("\n" + "=" * 70)
    print("✓ Dataset test completed successfully!")
    print("=" * 70)
    print("\nTo load with SeisBench:")
    print("  import seisbench.data")
    print(f"  data = seisbench.data.WaveformDataset('{hdf5_file}', '{csv_file}')")
    print("  waveforms, metadata = data.get_idx(0)")


if __name__ == "__main__":
    main()
