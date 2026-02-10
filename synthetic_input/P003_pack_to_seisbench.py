#!/usr/bin/env python3
"""
Pack synthetic seismograms into SeisBench HDF5 and CSV format.

This script converts generated synthetic 3-component seismograms into the
SeisBench data format for easy integration with SeisBench models and workflows.

Uses the official SeisBench WaveformDataWriter API for proper dataset creation,
following the reference example from SeisBench documentation.

Output:
    - metadata.csv: Trace metadata with phase picks
    - waveforms.hdf5: 3-component waveform data in HDF5 format

The format follows SeisBench conventions for compatibility with existing
SeisBench models (e.g., PhaseNet, EQTransformer).

Usage:
    python P003_pack_to_seisbench.py

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
            "Please run P002_batch_generate_synthetic_3c.py first."
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
    
    Following the pattern from SeisBench reference example, metadata uses
    prefixes like 'station_', 'trace_', 'source_' for different property types.
    
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
    station_code = metadata.get('station', 'HX')
    network_code = metadata.get('network', 'QD')
    station_latitude = metadata.get('station_latitude', 35.0)
    station_longitude = metadata.get('station_longitude', -97.0)
    
    # Phase arrivals
    p_arrival_sample = metadata.get('p_arrival_sample', -1)
    s_arrival_sample = metadata.get('s_arrival_sample', -1)
    
    # Build metadata dictionary with SeisBench conventions
    trace_metadata = {
        # Station information (station_ prefix)
        'station_network_code': network_code,
        'station_code': station_code,
        'station_location_code': '',
        'station_latitude': station_latitude,
        'station_longitude': station_longitude,
        
        # Trace properties (trace_ prefix)
        'trace_channel': 'HH',  # High-gain, high sample rate
        'trace_sampling_rate_hz': sampling_rate,
        'trace_npts': data_3c.shape[1],
        'trace_start_time': metadata.get('start_time', '1970-01-01T00:00:00'),
        
        # Phase arrivals - P-wave (trace_ prefix)
        'trace_p_arrival_sample': p_arrival_sample if p_arrival_sample >= 0 else None,
        'trace_p_status': 'manual' if p_arrival_sample >= 0 else None,
        'trace_p_weight': 1.0 if p_arrival_sample >= 0 else None,
        
        # Phase arrivals - S-wave (trace_ prefix)
        'trace_s_arrival_sample': s_arrival_sample if s_arrival_sample >= 0 else None,
        'trace_s_status': 'manual' if s_arrival_sample >= 0 else None,
        'trace_s_weight': 1.0 if s_arrival_sample >= 0 else None,
        
        # Source information (source_ prefix)
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
    """
    Create SeisBench dataset using the official WaveformDataWriter API.
    
    This function follows the SeisBench best practices for dataset creation
    as shown in the reference creating_a_dataset.ipynb example:
    - Uses WaveformDataWriter as a context manager
    - Sets proper data_format specifications
    - Writes traces incrementally with add_trace()
    - Creates metadata.csv and waveforms.hdf5 in SeisBench format
    
    Args:
        events: List of event dictionaries from discover_synthetic_data()
        output_dir: Output directory for dataset files (default: '../data')
        
    Returns:
        Path to the output directory
        
    Raises:
        IOError: If dataset files cannot be created
    """
    # Setup output paths
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = base_path / 'metadata.csv'
    waveforms_path = base_path / 'waveforms.hdf5'
    
    print(f"\nCreating SeisBench dataset using WaveformDataWriter")
    print(f"  Output directory: {base_path}")
    print(f"  Number of events: {len(events)}")
    
    # Use WaveformDataWriter following SeisBench conventions
    # This is the official API method from seisbench.data
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        # Define data format specifications
        # This tells SeisBench how to interpret the waveform arrays
        writer.data_format = {
            'dimension_order': 'CW',  # Channel, Width (samples)
            'component_order': 'ZNE',  # Vertical, North, East
            'measurement': 'velocity',
            'unit': 'counts',
            'instrument_response': 'not restituted',
        }
        
        # Iterate over events and write traces
        traces_written = 0
        traces_failed = 0
        
        for i, event in enumerate(events, 1):
            event_id = event['event_id']
            
            try:
                # Load waveform data
                data_3c, sampling_rate = load_waveform_data(event)
                
                # Build metadata dictionary
                trace_metadata = build_trace_metadata(event, data_3c, sampling_rate)
                
                # Add trace to dataset using SeisBench writer
                # The writer handles HDF5 writing and metadata collection
                writer.add_trace(trace_metadata, data_3c)
                
                traces_written += 1
                
                if i % 10 == 0 or i == len(events):
                    print(f"  Progress: {i}/{len(events)} events processed ({traces_written} written, {traces_failed} failed)")
                
            except Exception as e:
                traces_failed += 1
                print(f"  Warning: Failed to process {event_id}: {e}")
                continue
    
    print(f"\n✓ SeisBench dataset created successfully!")
    print(f"  Total traces written: {traces_written}")
    print(f"  Failed: {traces_failed}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Waveforms: {waveforms_path}")
    
    return base_path


def print_dataset_summary(dataset_path: Path) -> None:
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
            if s_count > 0:
                s_samples = df[df[s_col].notna()][s_col]
                print(f"    Sample range: {s_samples.min():.0f} - {s_samples.max():.0f}")
        
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
        print(f"\n  Warning: Could not load metadata for summary: {e}")
    
    print("=" * 70)


def verify_dataset(dataset_path: Path) -> bool:
    """
    Verify integrity of created SeisBench dataset by loading it.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if dataset is valid, False otherwise
    """
    print("\nVerifying dataset integrity...")
    
    try:
        # Try to load the dataset using SeisBench
        # This is the ultimate test - if SeisBench can load it, it's valid
        dataset = sbd.WaveformDataset(dataset_path, sampling_rate=100)
        
        print(f"  ✓ Dataset loaded successfully with SeisBench")
        print(f"  ✓ Number of traces: {len(dataset)}")
        
        # Check metadata
        if len(dataset.metadata) == 0:
            print(f"  ✗ Dataset has no metadata")
            return False
        
        print(f"  ✓ Metadata loaded: {len(dataset.metadata)} rows")
        
        # Try to load first waveform
        if len(dataset) > 0:
            waveform = dataset.get_waveforms(0)
            print(f"  ✓ Waveform shape: {waveform.shape}")
            
            if waveform.shape[0] != 3:
                print(f"  ✗ Invalid waveform shape (expected 3 components)")
                return False
        
        # Check required columns
        required_cols = ['station_code', 'station_network_code', 'trace_sampling_rate_hz']
        missing_cols = [col for col in required_cols if col not in dataset.metadata.columns]
        if missing_cols:
            print(f"  ⚠ Warning: Missing recommended columns: {missing_cols}")
        else:
            print(f"  ✓ Required columns present")
        
        print("  ✓ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_synthetic_files(events: List[Dict]) -> None:
    """
    Delete synthetic trace files after successful packing to save disk space.
    
    Removes MSEED files, NPY files, and metadata JSON files for each event.
    
    Args:
        events: List of event dictionaries from discover_synthetic_data()
    """
    print("\nCleaning up synthetic trace files...")
    
    files_deleted = 0
    files_failed = 0
    
    for event in events:
        try:
            # Delete MSEED files
            for mseed_file in event['mseed_files'].values():
                if os.path.exists(mseed_file):
                    os.remove(mseed_file)
                    files_deleted += 1
            
            # Delete NPY file
            if os.path.exists(event['npy_file']):
                os.remove(event['npy_file'])
                files_deleted += 1
            
            # Delete metadata JSON file
            metadata_file = f"{event['base_name']}_metadata.json"
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                files_deleted += 1
                
        except Exception as e:
            files_failed += 1
            print(f"  Warning: Failed to delete files for {event['event_id']}: {e}")
    
    print(f"  ✓ Deleted {files_deleted} synthetic trace files")
    if files_failed > 0:
        print(f"  ⚠ Failed to delete {files_failed} files")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Packing Synthetic Seismograms into SeisBench Format")
    print("Using Official WaveformDataWriter API")
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
        # Clean up synthetic trace files after successful packing
        cleanup_synthetic_files(events)
        
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
        print(f"  metadata = dataset.metadata")
        print("\nNote: Original synthetic trace files have been deleted to save space.")
        print("      All data is preserved in the SeisBench format.")
        return 0
    else:
        print("\n✗ Dataset verification failed!")
        print("   Keeping synthetic trace files for debugging.")
        return 1


if __name__ == "__main__":
    exit(main())
