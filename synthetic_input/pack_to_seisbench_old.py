#!/usr/bin/env python3
"""
Pack synthetic seismograms into SeisBench HDF5 and CSV format.

This script converts generated synthetic 3-component seismograms into the
SeisBench data format for easy integration with SeisBench models and workflows.

Uses the official SeisBench WaveformDataWriter API for proper dataset creation.

Output:
    - waveforms.hdf5: Waveform data in HDF5 format
    - metadata.csv: Metadata with phase picks in CSV format

The format follows SeisBench conventions for compatibility with existing
SeisBench models (e.g., PhaseNet, EQTransformer).

Usage:
    python pack_to_seisbench.py

Requirements:
    - seisbench
    - pandas
    - numpy
    - obspy
"""

import numpy as np
import pandas as pd
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from obspy import read
from datetime import datetime

import seisbench.data as sbd
import seisbench.util as sbu


def discover_synthetic_data() -> List[Dict]:
    """
    Discover all synthetic seismogram files and load their metadata.
    
    Returns:
        List of dictionaries containing event information and file paths
        
    Raises:
        FileNotFoundError: If no synthetic data files are found
    """
    metadata_files = sorted(glob.glob('SYNTHETIC_*_metadata.json'))
    
    if len(metadata_files) == 0:
        raise FileNotFoundError(
            "No synthetic seismogram metadata files found (SYNTHETIC_*_metadata.json). "
            "Please run batch_generate_synthetic_3c.py first."
        )
    
    events = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract base name for file discovery
        base_name = meta_file.replace('_metadata.json', '')
        
        # Check for required files
        npy_file = f"{base_name}_3C.npy"
        mseed_files = {
            'Z': f"{base_name}_HHZ.mseed",
            'N': f"{base_name}_HHN.mseed",
            'E': f"{base_name}_HHE.mseed"
        }
        
        # Verify files exist
        if not os.path.exists(npy_file):
            print(f"Warning: NPY file not found for {metadata['event_id']}, skipping...")
            continue
        
        missing_mseed = [ch for ch, path in mseed_files.items() if not os.path.exists(path)]
        if missing_mseed:
            print(f"Warning: Missing MSEED files for {metadata['event_id']} channels: {missing_mseed}")
            # Continue if NPY file exists as fallback
        
        events.append({
            'event_id': metadata['event_id'],
            'metadata': metadata,
            'npy_file': npy_file,
            'mseed_files': mseed_files,
            'base_name': base_name
        })
    
    return events


def load_waveform_data(event: Dict, use_mseed: bool = True) -> Tuple[np.ndarray, float]:
    """
    Load 3-component waveform data from MSEED or NPY files.
    
    Args:
        event: Event dictionary from discover_synthetic_data()
        use_mseed: If True, load from MSEED files; otherwise use NPY (default: True)
        
    Returns:
        data_3c: Waveform array of shape (3, n_samples) for Z, N, E
        sampling_rate: Sampling rate in Hz
        
    Raises:
        ValueError: If waveform data cannot be loaded
    """
    if use_mseed and all(os.path.exists(f) for f in event['mseed_files'].values()):
        # Load from MSEED files
        channels = ['Z', 'N', 'E']
        data_list = []
        
        for ch in channels:
            st = read(event['mseed_files'][ch])
            tr = st[0]
            data_list.append(tr.data)
            sampling_rate = tr.stats.sampling_rate
        
        data_3c = np.vstack(data_list)
        
    else:
        # Load from NPY file
        data_3c = np.load(event['npy_file'])
        sampling_rate = event['metadata'].get('sample_rate', 100.0)
    
    # Validate shape
    if data_3c.shape[0] != 3:
        raise ValueError(
            f"Invalid waveform shape {data_3c.shape} for {event['event_id']}. "
            f"Expected (3, n_samples)."
        )
    
    return data_3c, sampling_rate


def build_trace_metadata(event: Dict, data_3c: np.ndarray, sampling_rate: float) -> Dict:
    """
    Build metadata dictionary for a single trace following SeisBench conventions.
    
    Args:
        event: Event dictionary from discover_synthetic_data()
        data_3c: Waveform array of shape (3, n_samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with trace metadata following SeisBench naming conventions
    """
    metadata = event['metadata']
    event_id = event['event_id']
    
    # Station/Network information
    station_code = metadata.get('station', 'SYN')
    network_code = metadata.get('network', 'XX')
    
    # Phase arrivals
    p_arrival_sample = metadata.get('p_arrival_sample', -1)
    s_arrival_sample = metadata.get('s_arrival_sample', -1)
    
    # Build metadata dictionary with SeisBench conventions
    trace_metadata = {
        # Station information
        'station_network_code': network_code,
        'station_code': station_code,
        'station_location_code': '',
        
        # Trace properties
        'trace_channel': 'HH',  # High-gain, high sample rate
        'trace_sampling_rate_hz': sampling_rate,
        'trace_npts': data_3c.shape[1],
        'trace_start_time': metadata.get('start_time', '1970-01-01T00:00:00'),
        
        # Phase arrivals (P-wave)
        'trace_p_arrival_sample': p_arrival_sample if p_arrival_sample >= 0 else None,
        'trace_p_status': 'manual' if p_arrival_sample >= 0 else None,
        'trace_p_weight': 1.0 if p_arrival_sample >= 0 else None,
        
        # Phase arrivals (S-wave)
        'trace_s_arrival_sample': s_arrival_sample if s_arrival_sample >= 0 else None,
        'trace_s_status': 'manual' if s_arrival_sample >= 0 else None,
        'trace_s_weight': 1.0 if s_arrival_sample >= 0 else None,
        
        # Source information
        'source_id': event_id,
        'source_origin_time': metadata.get('start_time', '1970-01-01T00:00:00'),
        'source_type': 'earthquake',
        'source_magnitude': 2.0,
        'source_magnitude_type': 'ML',
        'source_magnitude_author': 'synthetic',
        
        # Quality metrics
        'trace_snr_db': metadata.get('snr_db', 0.0),
        
        # Dataset split
        'split': 'train',  # Default to training set
    }
    
    return trace_metadata


def create_seisbench_dataset(
    events: List[Dict],
    output_dir: str = '../data'
) -> Path:
    
    Args:
        df: Metadata DataFrame
        hdf5_file: Path to HDF5 file
    """
    print("\n" + "=" * 70)dataset_path: Path) -> None:
    """
    Load and print summary statistics of the created SeisBench dataset.
    
    Args:
        dataset_path: Path to the dataset directory
    """
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    
    metadata_path = dataset_path / 'metadata.csv'
    waveforms_path = dataset_path / 'waveforms.hdf5'
    
    # File sizes
    if waveforms_path.exists():
        size_mb = waveforms_path.stat().st_size / (1024 * 1024)
        print(f"Waveforms HDF5 size: {size_mb:.2f} MB")
    
    if metadata_path.exists():
        size_kb = metadata_path.stat().st_size / 1024
        print(f"Metadata CSV size: {size_kb:.2f} KB")
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
        
        print(f"\nDataset statistics:")
        print(f"  Total traces: {len(df)}")
        print(f"  Stations: {df['station_code'].nunique()}")
        print(f"  Networks: {df['station_network_code'].nunique()}")
        
        # Sampling rate
        if 'trace_sampling_rate_hz' in df.columns:
            print(f"  Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
        
        # Pick statistics
        print(f"\nPhase arrival statistics:")
        p_col = 'trace_p_arrival_sample'
        s_col = 'trace_s_arrival_sample'
        
        if p_col in df.columns:
            p_count = df[p_col].notna().sum()
            print(f"  P-arrivals: {p_count}/{len(df)}")
            if p_count > 0:
                p_samples = df[df[p_col].notna()][p_col]
                print(f"    Sample range: {p_samples.min():.0f} - {p_samples.max():.0f}")
        
        if s_col in df.columns:
            s_count = df[s_col].notna().sum()
            print(f"  S-arrivals: {s_count}/{len(df)}")
            if s_codataset_path: Path        
        # SNR statistics
        if 'trace_snr_db' in df.columns:
            snr_data = df['trace_snr_db']
            print(f"\nSNR statistics:")
            print(f"  Mean: {snr_data.mean():.2f} dB")
            print(f"  Std:  {snr_data.std():.2f} dB")
            print(f"  Range: {snr_data.min():.2f} - {snr_data.max():.2f} dB")
        
        # Split information
        if 'split' in df.columns:
            print(f"\nDataset splits:")
            for split_name in ['train', 'dev', 'test']:
                count = (df['split'] == split_name).sum()
                if count > 0:
                    print(f"  {split_name}: {count} traces")
        
    except Exception as e:
        print(f"\n  Warning: Could not load metadata for summary: {e}
    Args:
        hdf5_file: Path to HDF5 file
        csv_file: Path to CSV file
        
    Returns:
        True if dataset is valid, False otherwise
    """
    print("\nVerifying dataset integrity...")
    
    try:
        # Check files exist
        if not os.path.exists(hdf5_file):
            print(f"  ✗ HDF5 file not found: {hdf5_file}")
            return False
        
        if not os.path.exists(csv_file):
            print(f"  ✗ CSV file not found: {csv_file}")
            return False
        
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"  ✓ CSV loaded: {len(df)} rows")
        
        # Check CSV has 3 components per event
        num_events = len(df) // 3
        if len(df) % 3 != 0:
            print(f"  ⚠ Warning: CSV has {len(df)} rows, not divisible by 3 (expected 3 components per event)")
        
        # Open HDF5 and check traces
        with h5py.File(hdf5_file, 'r') as hdf:
            # Count actual trace datasets (exclude data_format group)
            trace_keys = [k for k in hdf.keys() if k != 'data_format']
            num_traces = len(trace_keys)
            print(f"  ✓ HDF5 loaded: {num_traces} traces (events)")
            
            # Check consistency: CSV should have 3 rows per HDF5 trace (3 components)
            if len(df) != num_traces * 3:
                print(f"  ✗ Mismatch: CSV has {len(df)} rows but HDF5 has {num_traces} traces")
                print(f"     Expected {num_traces * 3} CSV rows (3 per trace)")
                return False
            
            # Check a few traces
            for trace_name in trace_keys[:3]:
                data = hdf[trace_name][:]
                if data.shape[0] != 3:
                    print(f"  ✗ Invalid shape for {trace_name}: {data.shape}")
                    return False
            
            print(f"  ✓ Trace shapes validated (3, n_samples)")
            print(f"  ✓ CSV format validated (3 component rows per event)")
        
        # Verify required columns exist
        required_cols = [
            'network_code', 'receiver_code', 'receiver_type',
            'p_arrival_sample', 's_arrival_sample',
            'trace_name', 'trace_category'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing required columns: {missing_cols}")
            return False
        
        print(f"  ✓ All required columns present")
        priUsing Official WaveformDataWriter API")
    print("=" * 70)
    
    # Discover synthetic data
    try:
        events = discover_synthetic_data()
        print(f"\n✓ Found {len(events)} synthetic seismograms")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    # Create SeisBench dataset using WaveformDataWriter
    try:
        dataset_path = create_seisbench_dataset(events, output_dir='../data')
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print_dataset_summary(dataset_path)
    
    # Verify dataset by loading it with SeisBench
    if verify_dataset(dataset_path):
        print("\n" + "=" * 70)
        print("✓ Dataset creation completed successfully!")
        print("=" * 70)
        print(f"\nOutput directory: {dataset_path}")
        print(f"  - metadata.csv (trace metadata with picks)")
        print(f"  - waveforms.hdf5 (3-component waveform data)")
        print("\nUsage with SeisBench:")
        print("  import seisbench.data as sbd")
        print(f"  dataset = sbd.WaveformDataset('{dataset_path}', sampling_rate=100)")
        print(f"  waveform = dataset.get_waveforms(0)")
        print(f"  metadata = dataset.metadata
    
    # Verify dataset
    if verify_dataset(hdf5_file, csv_file):
        print("\n" + "=" * 70)
        print("✓ Dataset creation completed successfully!")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {hdf5_file} (waveform data)")
        print(f"  - {csv_file} (metadata with picks)")
        print("\nUsage with SeisBench:")
        print("  import seisbench.data")
        print(f"  data = seisbench.data.WaveformDataset('{hdf5_file}', '{csv_file}')")
        return 0
    else:
        print("\n✗ Dataset verification failed!")
        return 1


if __name__ == "__main__":
    exit(main())
