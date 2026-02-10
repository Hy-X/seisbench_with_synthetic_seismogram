#!/usr/bin/env python3
"""
Test loading and visualizing SeisBench format synthetic dataset.

This script demonstrates how to load the packed synthetic dataset
and visualize sample waveforms with phase picks.

Usage:
    python T001_test_seisbench_dataset.py

Requirements:
    - seisbench
    - pandas
    - numpy
    - matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import seisbench.data as sbd


def load_dataset(data_dir: str = '../data') -> sbd.WaveformDataset:
    """
    Load SeisBench dataset using the official API.
    
    Args:
        data_dir: Path to directory containing metadata.csv and waveforms.hdf5
        
    Returns:
        SeisBench WaveformDataset object
        
    Raises:
        FileNotFoundError: If dataset files don't exist
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    metadata_file = data_path / 'metadata.csv'
    waveforms_file = data_path / 'waveforms.hdf5'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    if not waveforms_file.exists():
        raise FileNotFoundError(f"Waveforms file not found: {waveforms_file}")
    
    # Load using SeisBench API
    dataset = sbd.WaveformDataset(data_path, sampling_rate=100)
    
    return dataset


def plot_trace_with_picks(
    data: np.ndarray,
    metadata: pd.Series,
    sampling_rate: float = 100.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3-component waveform with P and S picks.
    
    Args:
        data: Waveform array (3, n_samples)
        metadata: Pandas Series with trace metadata
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save figure
    """
    n_samples = data.shape[1]
    time = np.arange(n_samples) / sampling_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    components = ['Vertical (Z)', 'North (N)', 'East (E)']
    colors = ['k', 'k', 'k']
    
    # Handle both dict and Series access
    p_sample = metadata.get('trace_p_arrival_sample', -1)
    s_sample = metadata.get('trace_s_arrival_sample', -1)
    
    # Convert to int if not NaN
    if pd.notna(p_sample):
        p_sample = int(p_sample)
    else:
        p_sample = -1
        
    if pd.notna(s_sample):
        s_sample = int(s_sample)
    else:
        s_sample = -1
    
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
    snr_db = metadata.get('trace_snr_db', metadata.get('snr_db', 0))
    trace_name = metadata.get('trace_name', metadata.name if hasattr(metadata, 'name') else 'Unknown')
    
    p_time_str = f'{p_sample/sampling_rate:.2f}s' if p_sample >= 0 else 'N/A'
    s_time_str = f'{s_sample/sampling_rate:.2f}s' if s_sample >= 0 else 'N/A'
    
    fig.suptitle(
        f'{trace_name} | SNR: {snr_db:.1f} dB | P: {p_time_str}, S: {s_time_str}',
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


def print_dataset_stats(dataset: sbd.WaveformDataset) -> None:
    """
    Print detailed statistics about the dataset.
    
    Args:
        dataset: SeisBench WaveformDataset object
    """
    df = dataset.metadata
    
    print(f"\nDataset Overview:")
    print(f"  Total traces: {len(df)}")
    print(f"  Stations: {df['station_code'].nunique()}")
    print(f"  Networks: {df['station_network_code'].nunique()}")
    
    if 'trace_sampling_rate_hz' in df.columns:
        print(f"  Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
    
    if 'trace_npts' in df.columns:
        duration = df['trace_npts'].iloc[0] / df['trace_sampling_rate_hz'].iloc[0]
        print(f"  Trace duration: {duration:.1f} s")
    
    # Phase arrival statistics
    if 'trace_p_arrival_sample' in df.columns:
        p_count = df['trace_p_arrival_sample'].notna().sum()
        print(f"\n  P-wave picks: {p_count}/{len(df)}")
        
    if 'trace_s_arrival_sample' in df.columns:
        s_count = df['trace_s_arrival_sample'].notna().sum()
        print(f"  S-wave picks: {s_count}/{len(df)}")
    
    # SNR statistics
    snr_col = 'trace_snr_db' if 'trace_snr_db' in df.columns else 'snr_db'
    if snr_col in df.columns:
        print(f"\n  SNR statistics:")
        print(f"    Mean: {df[snr_col].mean():.2f} dB")
        print(f"    Std:  {df[snr_col].std():.2f} dB")
        print(f"    Range: {df[snr_col].min():.2f} - {df[snr_col].max():.2f} dB")
    
    # Split information
    if 'split' in df.columns:
        splits = df['split'].value_counts()
        if len(splits) > 0:
            print(f"\n  Dataset splits:")
            for split_name, count in splits.items():
                print(f"    {split_name}: {count} traces")


def main():
    """Test loading and visualizing the dataset."""
    print("=" * 70)
    print("Testing SeisBench Format Dataset")
    print("=" * 70)
    
    # Load dataset using SeisBench API
    try:
        dataset = load_dataset('../data')
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Location: ../data")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run P003_pack_to_seisbench.py first to create the dataset.")
        return 1
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return 1
    
    # Print statistics
    print_dataset_stats(dataset)
    
    # Get metadata
    df = dataset.metadata
    
    # Select traces to visualize
    # Pick low, medium, high SNR examples if SNR column exists
    snr_col = 'trace_snr_db' if 'trace_snr_db' in df.columns else 'snr_db'
    
    if snr_col in df.columns and len(df) >= 3:
        df_sorted = df.sort_values(snr_col)
        example_indices = [
            0,                    # Low SNR
            len(df) // 2,        # Medium SNR
            len(df) - 1          # High SNR
        ]
    else:
        # Just take first 3 traces
        example_indices = list(range(min(3, len(df))))
    
    print(f"\n\nVisualizing {len(example_indices)} example traces...")
    
    for idx, trace_idx in enumerate(example_indices):
        try:
            # Load waveform using SeisBench
            waveform = dataset.get_waveforms(trace_idx)
            metadata = df.iloc[trace_idx]
            
            trace_name = metadata.name if hasattr(metadata, 'name') else f"trace_{trace_idx}"
            snr_value = metadata.get(snr_col, 0)
            
            # Get sampling rate
            sampling_rate = metadata.get('trace_sampling_rate_hz', 100.0)
            
            # Plot
            output_dir = Path('../output')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f'test_trace_{idx+1}_{trace_name}.png'
            
            plot_trace_with_picks(
                waveform,
                metadata,
                sampling_rate=sampling_rate,
                save_path=str(save_path)
            )
            
            print(f"  [{idx+1}] {trace_name}: SNR={snr_value:.1f} dB")
            
        except Exception as e:
            print(f"  Warning: Failed to plot trace {trace_idx}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("✓ Dataset test completed successfully!")
    print("=" * 70)
    print("\nOutput:")
    print(f"  Figures saved to: ../output/")
    print("\nUsage with SeisBench:")
    print("  import seisbench.data as sbd")
    print("  dataset = sbd.WaveformDataset('../data', sampling_rate=100)")
    print("  waveform = dataset.get_waveforms(0)")
    print("  metadata = dataset.metadata.iloc[0]")
    
    return 0


if __name__ == "__main__":
    exit(main())
