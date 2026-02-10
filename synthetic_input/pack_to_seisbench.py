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
        # Create data_format group with SeisBench specifications
        # This tells SeisBench how to interpret the data
        data_format = hdf.create_group('data_format')
        data_format.attrs['dimension_order'] = 'CW'  # Channel, Width (samples)
        data_format.attrs['component_order'] = 'ZNE'  # Vertical, North, East
        
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
            # Use simple event_id as dataset name at root level
            trace_name = event_id
            
            # Create dataset directly at root level (SeisBench simple format)
            # This is simpler than bucketed format used by large datasets
            dset = hdf.create_dataset(
                trace_name,
                data=data_3c,
                dtype='float32',
                compression=compression,
                compression_opts=compression_opts
            )
            
            # Store trace-level metadata as dataset attributes
            dset.attrs['sampling_rate'] = sampling_rate
            dset.attrs['station'] = metadata.get('station', 'SYN')
            dset.attrs['network'] = metadata.get('network', 'XX')
            dset.attrs['event_id'] = event_id
            dset.attrs['p_arrival_sample'] = metadata.get('p_arrival_sample', -1)
            dset.attrs['s_arrival_sample'] = metadata.get('s_arrival_sample', -1)
            dset.attrs['p_arrival_time'] = metadata.get('p_arrival_time', -1.0)
            dset.attrs['s_arrival_time'] = metadata.get('s_arrival_time', -1.0)
            dset.attrs['snr_db'] = metadata.get('snr_db', 0.0)
            
            if i % 10 == 0 or i == len(events):
                print(f"  Progress: {i}/{len(events)} traces written")
    
    print(f"✓ HDF5 dataset created: {output_file}")


def create_csv_metadata(
    events: List[Dict],
    output_file: str = 'synthetic_metadata.csv'
) -> pd.DataFrame:
    """
    Create CSV metadata file with phase picks in SeisBench format.
    
    The CSV follows SeisBench conventions matching real datasets with columns:
    - network_code: Network code
    - receiver_code: Station/receiver code
    - receiver_type: Component channel (HHZ, HHN, HHE, etc.)
    - receiver_latitude: Station latitude (degrees)
    - receiver_longitude: Station longitude (degrees)
    - receiver_elevation_m: Station elevation (meters)
    - p_arrival_sample: P-wave arrival sample index
    - p_status: P-pick status ('manual', 'automatic')
    - p_weight: Pick weight/confidence for P (0-1)
    - p_travel_sec: P-wave travel time in seconds
    - s_arrival_sample: S-wave arrival sample index
    - s_status: S-pick status
    - s_weight: Pick weight/confidence for S (0-1)
    - source_id: Unique event identifier
    - source_origin_time: Event origin time (ISO format)
    - source_latitude: Event latitude (degrees)
    - source_longitude: Event longitude (degrees)
    - source_depth_km: Event depth (kilometers)
    - source_magnitude: Event magnitude
    - source_magnitude_type: Magnitude type (ML, Mw, etc.)
    - snr_db: Signal-to-noise ratio in dB
    - trace_start_time: Trace start time (ISO format)
    - trace_category: Category (earthquake_local, noise, etc.)
    - trace_name: Unique trace identifier
    
    Note: Creates 3 rows per event (one per component: Z, N, E)
    
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
    
    # Component mapping for receiver_type
    components = ['HHZ', 'HHN', 'HHE']
    component_suffixes = ['Z', 'N', 'E']
    
    for event in events:
        event_id = event['event_id']
        metadata = event['metadata']
        
        # Load data to get sampling info
        try:
            data_3c, sampling_rate = load_waveform_data(event)
            npts = data_3c.shape[1]
        except Exception as e:
            print(f"  Warning: Could not load {event_id} for metadata: {e}")
            continue
        
        station_code = metadata.get('station', 'SYN')
        network_code = metadata.get('network', 'XX')
        start_time = metadata.get('start_time', '1970-01-01T00:00:00')
        
        # Get phase arrivals
        p_arrival_sample = metadata.get('p_arrival_sample', -1)
        s_arrival_sample = metadata.get('s_arrival_sample', -1)
        p_arrival_time = metadata.get('p_arrival_time', -1.0)
        s_arrival_time = metadata.get('s_arrival_time', -1.0)
        snr_db = metadata.get('snr_db', 0.0)
        
        # Calculate travel times (from trace start)
        p_travel_sec = p_arrival_time if p_arrival_sample >= 0 else -1.0
        s_travel_sec = s_arrival_time if s_arrival_sample >= 0 else -1.0
        
        # Default synthetic source parameters
        # Use reasonable values for synthetic events
        receiver_lat = 35.0  # Default receiver location
        receiver_lon = -97.0
        receiver_elev = 500.0
        
        source_lat = 36.0  # Default source location
        source_lon = -98.0
        source_depth_km = 5.0
        source_magnitude = 2.0
        source_magnitude_type = 'ML'
        
        # Create one row per component (3C data)
        for comp_idx, (component, suffix) in enumerate(zip(components, component_suffixes)):
            # Build trace name following SeisBench convention
            # Format: network.station.component_type.component_order.start_time
            trace_name = f"{network_code}.{station_code}.{component}.{start_time.replace(':', '').replace('-', '').replace('T', 'T')}"
            
            row = {
                'network_code': network_code,
                'receiver_code': station_code,
                'receiver_type': component,
                'receiver_latitude': receiver_lat,
                'receiver_longitude': receiver_lon,
                'receiver_elevation_m': receiver_elev,
                'p_arrival_sample': p_arrival_sample,
                'p_status': 'manual' if p_arrival_sample >= 0 else '',
                'p_weight': 1.0 if p_arrival_sample >= 0 else 0.0,
                'p_travel_sec': p_travel_sec,
                's_arrival_sample': s_arrival_sample,
                's_status': 'manual' if s_arrival_sample >= 0 else '',
                's_weight': 1.0 if s_arrival_sample >= 0 else 0.0,
                'source_id': event_id,
                'source_origin_time': start_time,
                'source_origin_uncertainty_sec': '',
                'source_latitude': source_lat,
                'source_longitude': source_lon,
                'source_error_sec': '',
                'source_gap_deg': '',
                'source_horizontal_uncertainty_km': '',
                'source_depth_km': source_depth_km,
                'source_depth_uncertainty_km': '',
                'source_magnitude': source_magnitude,
                'source_magnitude_type': source_magnitude_type,
                'source_magnitude_author': 'synthetic',
                'source_mechanism_strike_dip_rake': '',
                'source_distance_deg': '',
                'source_distance_km': '',
                'back_azimuth_deg': '',
                'snr_db': snr_db,
                'coda_end_sample': '',
                'trace_start_time': start_time,
                'trace_category': 'earthquake_local',
                'trace_name': trace_name
            }
            
            metadata_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✓ CSV metadata created: {output_file}")
    print(f"  Total trace components: {len(df)} (3 per event)")
    print(f"  Events: {len(df) // 3}")
    print(f"  Columns: {len(df.columns)}")
    
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
    num_events = len(df) // 3  # 3 components per event
    print(f"Total trace components: {len(df)} (3 per event)")
    print(f"Total events: {num_events}")
    print(f"Stations: {df['receiver_code'].nunique()}")
    print(f"Networks: {df['network_code'].nunique()}")
    
    # Pick statistics (check first row of each event)
    event_rows = df[::3]  # Every 3rd row (one per event)
    print(f"\nPhase arrival statistics (per event):")
    print(f"  P-arrivals: {(event_rows['p_arrival_sample'] >= 0).sum()}/{num_events}")
    print(f"  S-arrivals: {(event_rows['s_arrival_sample'] >= 0).sum()}/{num_events}")
    
    # P and S sample ranges
    p_samples = event_rows[event_rows['p_arrival_sample'] >= 0]['p_arrival_sample']
    s_samples = event_rows[event_rows['s_arrival_sample'] >= 0]['s_arrival_sample']
    
    if len(p_samples) > 0:
        print(f"  P-sample range: {p_samples.min():.0f} - {p_samples.max():.0f}")
        p_travel = event_rows[event_rows['p_arrival_sample'] >= 0]['p_travel_sec']
        print(f"  P-time range: {p_travel.min():.2f} - {p_travel.max():.2f} s")
    
    if len(s_samples) > 0:
        print(f"  S-sample range: {s_samples.min():.0f} - {s_samples.max():.0f}")
    
    # SNR statistics
    if 'snr_db' in df.columns:
        print(f"\nSNR statistics:")
        print(f"  Mean: {event_rows['snr_db'].mean():.2f} dB")
        print(f"  Std:  {event_rows['snr_db'].std():.2f} dB")
        print(f"  Min:  {event_rows['snr_db'].min():.2f} dB")
        print(f"  Max:  {event_rows['snr_db'].max():.2f} dB")
    
    # Component breakdown
    print(f"\nComponent breakdown:")
    for comp in df['receiver_type'].unique():
        count = (df['receiver_type'] == comp).sum()
        print(f"  {comp}: {count} traces")
    
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
