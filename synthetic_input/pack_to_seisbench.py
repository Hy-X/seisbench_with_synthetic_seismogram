#!/usr/bin/env python3
"""
Pack synthetic seismograms into SeisBench HDF5 and CSV format.

This script converts generated synthetic 3-component seismograms into the
SeisBench data format for easy integration with SeisBench models and workflows.

Output:
    - synthetic_dataset.hdf5: Waveform data in HDF5 format
    - synthetic_metadata.csv: Metadata with phase picks in CSV format

The format follows SeisBench conventions for compatibility with existing
SeisBench models (e.g., PhaseNet, EQTransformer).

Usage:
    python pack_to_seisbench.py

Requirements:
    - h5py
    - pandas
    - numpy
    - obspy
"""

import numpy as np
import pandas as pd
import h5py
import json
import glob
import os
from typing import Dict, List, Tuple, Optional
from obspy import read
from datetime import datetime


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


def create_hdf5_dataset(
    events: List[Dict],
    output_file: str = 'synthetic_dataset.hdf5',
    compression: str = 'gzip',
    compression_opts: int = 4
) -> None:
    """
    Create HDF5 file with waveform data in SeisBench format.
    
    The HDF5 structure follows SeisBench conventions:
    - Each trace is stored as a dataset named by trace_name
    - Dataset shape: (3, n_samples) for Z, N, E components
    - Attributes: sampling_rate, station, network, event_id
    
    Args:
        events: List of event dictionaries
        output_file: Output HDF5 filename (default: 'synthetic_dataset.hdf5')
        compression: HDF5 compression algorithm (default: 'gzip')
        compression_opts: Compression level 0-9 (default: 4)
        
    Raises:
        IOError: If HDF5 file cannot be created
    """
    print(f"\nCreating HDF5 dataset: {output_file}")
    print(f"  Number of traces: {len(events)}")
    
    with h5py.File(output_file, 'w') as hdf:
        # Store dataset-level metadata
        hdf.attrs['format_version'] = '1.0'
        hdf.attrs['dataset_name'] = 'synthetic_seismograms'
        hdf.attrs['creation_time'] = datetime.now().isoformat()
        hdf.attrs['num_traces'] = len(events)
        
        # Create data_format group with SeisBench specifications
        data_format = hdf.create_group('data_format')
        data_format.attrs['dimension_order'] = 'CW'  # Channel, Width (samples)
        data_format.attrs['component_order'] = 'ZNE'
        
        for i, event in enumerate(events, 1):
            event_id = event['event_id']
            metadata = event['metadata']
            
            # Load waveform data
            try:
                data_3c, sampling_rate = load_waveform_data(event)
            except Exception as e:
                print(f"  Error loading {event_id}: {e}")
                continue
            
            # Create trace name (SeisBench convention)
            # Format: network.station.location.channel or simply event_id
            trace_name = event_id
            
            # Create a group for this trace (SeisBench format)
            trace_group = hdf.create_group(trace_name)
            
            # Create dataset for waveform data within the trace group
            dset = trace_group.create_dataset(
                'data',
                data=data_3c,
                dtype='float32',
                compression=compression,
                compression_opts=compression_opts
            )
            
            # Store trace-level metadata as attributes on the group
            trace_group.attrs['sampling_rate'] = sampling_rate
            trace_group.attrs['station'] = metadata.get('station', 'SYN')
            trace_group.attrs['network'] = metadata.get('network', 'XX')
            trace_group.attrs['event_id'] = event_id
            trace_group.attrs['p_arrival_sample'] = metadata.get('p_arrival_sample', -1)
            trace_group.attrs['s_arrival_sample'] = metadata.get('s_arrival_sample', -1)
            trace_group.attrs['p_arrival_time'] = metadata.get('p_arrival_time', -1.0)
            trace_group.attrs['s_arrival_time'] = metadata.get('s_arrival_time', -1.0)
            trace_group.attrs['snr_db'] = metadata.get('snr_db', 0.0)
            
            if i % 10 == 0 or i == len(events):
                print(f"  Progress: {i}/{len(events)} traces written")
    
    print(f"✓ HDF5 dataset created: {output_file}")


def create_csv_metadata(
    events: List[Dict],
    output_file: str = 'synthetic_metadata.csv'
) -> pd.DataFrame:
    """
    Create CSV metadata file with phase picks in SeisBench format.
    
    The CSV follows SeisBench conventions with columns:
    - trace_name: Unique identifier for each trace
    - station_code: Station code
    - network_code: Network code
    - trace_sampling_rate_hz: Sampling rate in Hz
    - trace_npts: Number of samples
    - trace_start_time: Start time (ISO format or relative)
    - trace_p_arrival_sample: P-wave arrival sample index
    - trace_s_arrival_sample: S-wave arrival sample index
    - trace_p_status: P-pick status ('manual', 'automatic', etc.)
    - trace_s_status: S-pick status
    - trace_p_weight: Pick weight/uncertainty for P
    - trace_s_weight: Pick weight/uncertainty for S
    - snr_db: Signal-to-noise ratio in dB
    
    Args:
        events: List of event dictionaries
        output_file: Output CSV filename (default: 'synthetic_metadata.csv')
        
    Returns:
        DataFrame with metadata
        
    Raises:
        IOError: If CSV file cannot be written
    """
    print(f"\nCreating CSV metadata: {output_file}")
    
    metadata_list = []
    
    for event in events:
        event_id = event['event_id']
        metadata = event['metadata']
        
        # Load data to get number of samples
        try:
            data_3c, sampling_rate = load_waveform_data(event)
            npts = data_3c.shape[1]
        except Exception as e:
            print(f"  Warning: Could not load {event_id} for metadata: {e}")
            continue
        
        # Build metadata row following SeisBench conventions
        row = {
            'trace_name': event_id,
            'station_code': metadata.get('station', 'SYN'),
            'network_code': metadata.get('network', 'XX'),
            'trace_sampling_rate_hz': sampling_rate,
            'trace_npts': npts,
            'trace_start_time': metadata.get('start_time', '1970-01-01T00:00:00'),
            'trace_p_arrival_sample': metadata.get('p_arrival_sample', -1),
            'trace_s_arrival_sample': metadata.get('s_arrival_sample', -1),
            'trace_p_arrival_time': metadata.get('p_arrival_time', -1.0),
            'trace_s_arrival_time': metadata.get('s_arrival_time', -1.0),
            'trace_p_status': 'manual',  # Synthetic = known ground truth
            'trace_s_status': 'manual',
            'trace_p_weight': 1.0,  # High confidence (synthetic)
            'trace_s_weight': 1.0,
            'snr_db': metadata.get('snr_db', 0.0),
            'source_type': 'synthetic',
            'source_id': event_id
        }
        
        metadata_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✓ CSV metadata created: {output_file}")
    print(f"  Total traces: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return df


def print_dataset_summary(
    df: pd.DataFrame,
    hdf5_file: str = 'synthetic_dataset.hdf5'
) -> None:
    """
    Print summary statistics of the created dataset.
    
    Args:
        df: Metadata DataFrame
        hdf5_file: Path to HDF5 file
    """
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    
    # HDF5 file size
    if os.path.exists(hdf5_file):
        size_mb = os.path.getsize(hdf5_file) / (1024 * 1024)
        print(f"HDF5 file size: {size_mb:.2f} MB")
    
    # Basic statistics
    print(f"Total traces: {len(df)}")
    print(f"Stations: {df['station_code'].nunique()}")
    print(f"Networks: {df['network_code'].nunique()}")
    
    # Sampling rate
    print(f"Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
    print(f"Trace length: {df['trace_npts'].iloc[0]} samples ({df['trace_npts'].iloc[0] / df['trace_sampling_rate_hz'].iloc[0]:.1f} s)")
    
    # Pick statistics
    print(f"\nPhase arrival statistics:")
    print(f"  P-arrivals: {(df['trace_p_arrival_sample'] >= 0).sum()}/{len(df)}")
    print(f"  S-arrivals: {(df['trace_s_arrival_sample'] >= 0).sum()}/{len(df)}")
    
    # P and S sample ranges
    p_samples = df[df['trace_p_arrival_sample'] >= 0]['trace_p_arrival_sample']
    s_samples = df[df['trace_s_arrival_sample'] >= 0]['trace_s_arrival_sample']
    
    if len(p_samples) > 0:
        print(f"  P-sample range: {p_samples.min():.0f} - {p_samples.max():.0f}")
        print(f"  P-time range: {p_samples.min()/df['trace_sampling_rate_hz'].iloc[0]:.2f} - {p_samples.max()/df['trace_sampling_rate_hz'].iloc[0]:.2f} s")
    
    if len(s_samples) > 0:
        print(f"  S-sample range: {s_samples.min():.0f} - {s_samples.max():.0f}")
        print(f"  S-time range: {s_samples.min()/df['trace_sampling_rate_hz'].iloc[0]:.2f} - {s_samples.max()/df['trace_sampling_rate_hz'].iloc[0]:.2f} s")
    
    # SNR statistics
    if 'snr_db' in df.columns:
        print(f"\nSNR statistics:")
        print(f"  Mean: {df['snr_db'].mean():.2f} dB")
        print(f"  Std:  {df['snr_db'].std():.2f} dB")
        print(f"  Min:  {df['snr_db'].min():.2f} dB")
        print(f"  Max:  {df['snr_db'].max():.2f} dB")
    
    print("=" * 70)


def verify_dataset(
    hdf5_file: str = 'synthetic_dataset.hdf5',
    csv_file: str = 'synthetic_metadata.csv'
) -> bool:
    """
    Verify integrity of created dataset.
    
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
        
        # Open HDF5 and check traces
        with h5py.File(hdf5_file, 'r') as hdf:
            num_traces = len(hdf.keys())
            print(f"  ✓ HDF5 loaded: {num_traces} traces")
            
            # Check consistency
            if len(df) != num_traces:
                print(f"  ✗ Mismatch: CSV has {len(df)} rows but HDF5 has {num_traces} traces")
                return False
            
            # Check a few traces
            for trace_name in list(hdf.keys())[:3]:
                data = hdf[trace_name][:]
                if data.shape[0] != 3:
                    print(f"  ✗ Invalid shape for {trace_name}: {data.shape}")
                    return False
            
            print(f"  ✓ Trace shapes validated (3, n_samples)")
        
        print("  ✓ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 70)
    print("Packing Synthetic Seismograms into SeisBench Format")
    print("=" * 70)
    
    # Discover synthetic data
    try:
        events = discover_synthetic_data()
        print(f"\n✓ Found {len(events)} synthetic seismograms")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    # Define output files
    hdf5_file = 'synthetic_dataset.hdf5'
    csv_file = 'synthetic_metadata.csv'
    
    # Create HDF5 dataset
    try:
        create_hdf5_dataset(events, hdf5_file)
    except Exception as e:
        print(f"\n✗ Error creating HDF5: {e}")
        return 1
    
    # Create CSV metadata
    try:
        df = create_csv_metadata(events, csv_file)
    except Exception as e:
        print(f"\n✗ Error creating CSV: {e}")
        return 1
    
    # Print summary
    print_dataset_summary(df, hdf5_file)
    
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
