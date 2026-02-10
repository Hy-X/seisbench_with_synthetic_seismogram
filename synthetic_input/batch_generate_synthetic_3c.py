#!/usr/bin/env python3
"""
Batch generation of realistic 3-component synthetic seismograms.

This script generates multiple synthetic 3C seismograms with varying arrival times,
saving each as individual MiniSEED files, NumPy arrays, and metadata JSON files.
The number of seismograms to generate is specified in the configuration file.
"""

import numpy as np
import json
import os
from obspy import Trace, Stream, UTCDateTime
from typing import Tuple, List, Dict
from datetime import datetime

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback progress indicator when tqdm not available."""
        return iterable


def load_config(config_path: str = 'Syn_Config.json') -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file (relative or absolute)
        
    Returns:
        Configuration dictionary containing generation parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_random_times(
    duration: float,
    pre_time: float,
    window_length: float,
    min_sep: float = 1.0
) -> Tuple[float, float]:
    """
    Generate random P and S arrival times ensuring minimum separation.
    
    Args:
        duration: Total seismogram duration in seconds
        pre_time: Pre-arrival time window in seconds
        window_length: Phase window length in seconds
        min_sep: Minimum separation between P and S in seconds
        
    Returns:
        p_time: P-wave arrival time in seconds
        s_time: S-wave arrival time in seconds
    """
    post_time = window_length - pre_time
    start_margin = pre_time + 0.1
    end_margin = post_time + 0.1
    
    latest_p = duration - end_margin - min_sep
    if latest_p <= start_margin:
        return duration * 0.3, duration * 0.6
        
    p_time = np.random.uniform(start_margin, latest_p)
    
    latest_s = duration - end_margin
    earliest_s = p_time + min_sep
    
    if latest_s <= earliest_s:
        s_time = latest_s
    else:
        s_time = np.random.uniform(earliest_s, latest_s)
         
    return p_time, s_time


def generate_realistic_wave_packet(
    n_samples: int,
    arrival_sample: int,
    dominant_freq: float,
    sample_rate: float,
    amplitude: float = 1.0,
    decay_rate: float = 0.3
) -> np.ndarray:
    """
    Generate realistic seismic wave packet with exponential decay envelope.
    
    Args:
        n_samples: Total number of samples in output array
        arrival_sample: Sample index of phase arrival
        dominant_freq: Dominant frequency of wave packet in Hz
        sample_rate: Sampling rate in Hz
        amplitude: Peak amplitude of wave packet
        decay_rate: Exponential decay rate (larger = faster decay)
        
    Returns:
        Wave packet array of shape (n_samples,)
    """
    wave = np.zeros(n_samples)
    duration_after = (n_samples - arrival_sample) / sample_rate
    t = np.linspace(0, duration_after, n_samples - arrival_sample)
    
    envelope = np.exp(-decay_rate * t)
    phase = 2 * np.pi * dominant_freq * t
    
    # Smooth onset ramp
    onset_samples = int(0.5 * sample_rate / dominant_freq)
    if len(t) > onset_samples:
        onset_ramp = np.linspace(0, 1, onset_samples)
        envelope[:onset_samples] *= onset_ramp
    
    wave[arrival_sample:] = amplitude * envelope * np.sin(phase)
    return wave


def add_coda_waves(
    waveform: np.ndarray,
    s_arrival_sample: int,
    sample_rate: float,
    amplitude: float
) -> np.ndarray:
    """
    Add realistic coda waves following S-wave arrival.
    
    Coda consists of multiple frequency components with exponential decay,
    simulating scattered energy from heterogeneous subsurface.
    
    Args:
        waveform: Input waveform array to modify in-place
        s_arrival_sample: Sample index of S-wave arrival
        sample_rate: Sampling rate in Hz
        amplitude: Peak amplitude of coda waves
        
    Returns:
        Modified waveform with added coda waves
    """
    n_samples = len(waveform)
    coda_start = s_arrival_sample + int(2 * sample_rate)
    
    if coda_start < n_samples:
        coda_length = n_samples - coda_start
        t = np.arange(coda_length) / sample_rate
        
        coda = np.zeros(coda_length)
        for freq in [5, 8, 12, 15, 20]:
            phase = 2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)
            decay = np.exp(-0.3 * t)
            coda += np.sin(phase) * decay
        
        coda *= amplitude / 5
        waveform[coda_start:] += coda
    
    return waveform


def generate_3c_event(
    duration: float,
    sample_rate: float,
    p_time: float,
    s_time: float,
    noise_level: float,
    p_frequency: float,
    s_frequency: float,
    coda_amplitude: float,
    station: str = 'SYN',
    network: str = 'XX'
) -> Tuple[Stream, np.ndarray]:
    """
    Generate 3-component (Z, N, E) synthetic seismic event.
    
    Produces realistic waveforms with:
    - Coherent background noise across components
    - P-wave with vertical polarization dominance
    - S-wave with horizontal polarization dominance
    - Randomized but physically realistic polarizations
    - Coda waves following S-arrival
    
    Args:
        duration: Total duration in seconds
        sample_rate: Sampling rate in Hz
        p_time: P-wave arrival time in seconds
        s_time: S-wave arrival time in seconds
        noise_level: Background noise amplitude (relative to signal)
        p_frequency: P-wave dominant frequency in Hz
        s_frequency: S-wave dominant frequency in Hz
        coda_amplitude: Coda wave amplitude (relative to S-wave)
        station: Station code (default: 'SYN')
        network: Network code (default: 'XX')
        
    Returns:
        stream: ObsPy Stream object with 3 traces (Z, N, E)
        time: Time vector array in seconds
    """
    n_samples = int(duration * sample_rate)
    time = np.arange(n_samples) / sample_rate
    
    # Initialize channels
    channels = ['HHZ', 'HHN', 'HHE']
    data = {}
    
    # Generate coherent noise sources (simulating ground motion coupling)
    noise_master = np.random.randn(n_samples)
    window_size = 5
    noise_master = np.convolve(noise_master, np.ones(window_size)/window_size, mode='same')
    
    # Random projection of noise onto components (ensures coherence)
    noise_pol_z = np.random.uniform(0.5, 1.0)
    noise_pol_n = np.random.uniform(0.3, 0.7) * np.random.choice([1, -1])
    noise_pol_e = np.random.uniform(0.3, 0.7) * np.random.choice([1, -1])
    
    # Add independent component (scattering/instrument noise)
    noise_indep_scale = 0.3
    
    data['HHZ'] = noise_level * (noise_master * noise_pol_z + 
                                  np.random.randn(n_samples) * noise_indep_scale * 0.1)
    data['HHN'] = noise_level * (noise_master * noise_pol_n + 
                                  np.random.randn(n_samples) * noise_indep_scale * 0.1)
    data['HHE'] = noise_level * (noise_master * noise_pol_e + 
                                  np.random.randn(n_samples) * noise_indep_scale * 0.1)

    # Define polarization factors (randomized for variety)
    # P-wave: Strong on Z, weaker on H
    p_pol_z = np.random.uniform(0.7, 1.0)
    p_pol_n = np.random.uniform(0.1, 0.4) * np.random.choice([1, -1])
    p_pol_e = np.random.uniform(0.1, 0.4) * np.random.choice([1, -1])
    
    # S-wave: Strong on H, weaker on Z
    s_pol_z = np.random.uniform(0.1, 0.3) * np.random.choice([1, -1])
    s_pol_n = np.random.uniform(0.6, 1.0) * np.random.choice([1, -1])
    s_pol_e = np.random.uniform(0.6, 1.0) * np.random.choice([1, -1])
    
    p_sample = int(p_time * sample_rate)
    s_sample = int(s_time * sample_rate)

    # Generate source wavelets
    p_source = generate_realistic_wave_packet(
        n_samples, p_sample, p_frequency, sample_rate, 
        amplitude=0.5, decay_rate=3.0
    )
    
    s_source = generate_realistic_wave_packet(
        n_samples, s_sample, s_frequency, sample_rate,
        amplitude=1.0, decay_rate=1.5
    )

    # Project onto components
    data['HHZ'] += p_source * p_pol_z + s_source * s_pol_z
    data['HHN'] += p_source * p_pol_n + s_source * s_pol_n
    data['HHE'] += p_source * p_pol_e + s_source * s_pol_e
    
    # Add coda waves
    data['HHZ'] = add_coda_waves(data['HHZ'], s_sample, sample_rate, coda_amplitude * 0.5)
    data['HHN'] = add_coda_waves(data['HHN'], s_sample, sample_rate, coda_amplitude)
    data['HHE'] = add_coda_waves(data['HHE'], s_sample, sample_rate, coda_amplitude)

    # Create Stream
    stream = Stream()
    start_time = UTCDateTime(0)
    
    for ch in channels:
        tr = Trace(data=data[ch])
        tr.stats.station = station
        tr.stats.channel = ch
        tr.stats.network = network
        tr.stats.sampling_rate = sample_rate
        tr.stats.starttime = start_time
        stream.append(tr)
        
    return stream, time


def save_seismogram(
    stream: Stream,
    event_id: str,
    station: str,
    network: str,
    p_time: float,
    s_time: float,
    sample_rate: float,
    duration: float,
    noise_level: float,
    output_dir: str = '.'
) -> Dict[str, str]:
    """
    Save 3C seismogram to MiniSEED, NumPy array, and JSON metadata.
    
    Args:
        stream: ObsPy Stream object with 3 traces
        event_id: Unique event identifier
        station: Station code
        network: Network code
        p_time: P-wave arrival time in seconds
        s_time: S-wave arrival time in seconds
        sample_rate: Sampling rate in Hz
        duration: Total duration in seconds
        noise_level: Noise level for SNR calculation
        output_dir: Output directory path (default: current directory)
        
    Returns:
        Dictionary containing paths to saved files
    """
    base_name = f"{event_id}_synthetic_{station}_{network}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = {}
    
    # 1. Save MiniSEED files (one per channel)
    mseed_files = []
    for tr in stream:
        channel = tr.stats.channel
        filename = f"{base_name}_{channel}.mseed"
        filepath = os.path.join(output_dir, filename)
        tr.write(filepath, format="MSEED")
        mseed_files.append(filename)
    
    file_paths['mseed'] = mseed_files
    
    # 2. Save NumPy array (stacked 3C data)
    data_3c = np.vstack([tr.data for tr in stream])
    npy_filename = f"{base_name}_3C.npy"
    npy_filepath = os.path.join(output_dir, npy_filename)
    np.save(npy_filepath, data_3c)
    file_paths['npy'] = npy_filename
    
    # 3. Save metadata JSON
    metadata = {
        "event_id": event_id,
        "station": station,
        "network": network,
        "start_time": stream[0].stats.starttime.isoformat(),
        "sample_rate": sample_rate,
        "duration": duration,
        "p_arrival_sample": int(p_time * sample_rate),
        "s_arrival_sample": int(s_time * sample_rate),
        "p_arrival_time": float(p_time),
        "s_arrival_time": float(s_time),
        "snr_db": float(10 * np.log10(1.0 / noise_level)) if noise_level > 0 else float('inf'),
        "channels": ["HHZ", "HHN", "HHE"],
        "files": {
            "mseed": mseed_files,
            "npy": npy_filename
        }
    }
    
    json_filename = f"{base_name}_metadata.json"
    json_filepath = os.path.join(output_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    file_paths['metadata'] = json_filename
    
    return file_paths


def generate_batch_summary(
    metadata_list: List[Dict],
    output_dir: str = '.',
    summary_filename: str = 'batch_summary.json'
) -> None:
    """
    Generate summary file for batch of synthetic seismograms.
    
    Args:
        metadata_list: List of metadata dictionaries for all generated events
        output_dir: Output directory path
        summary_filename: Name of summary file
    """
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "num_seismograms": len(metadata_list),
        "events": metadata_list
    }
    
    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n[✓] Batch summary saved to: {summary_filename}")


def main():
    """Generate batch of 3C synthetic seismograms based on configuration."""
    print("=" * 70)
    print("Batch Generation of 3C Synthetic Seismograms")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Extract parameters
    num_seismograms = config.get('num_seismograms', 10)
    sample_rate = config.get('sample_rate', 100)
    duration = config.get('duration', 20.0)
    p_frequency = config.get('p_frequency', 20.0)
    s_frequency = config.get('s_frequency', 10.0)
    noise_level = config.get('noise_level', 0.15)
    randomize_noise = config.get('randomize_noise_level', False)
    noise_level_range = config.get('noise_level_range', [0.1, 0.4])
    window_length = config.get('window_length', 6.0)
    pre_arrival_time = config.get('pre_arrival_time', 2.0)
    coda_amplitude = config.get('coda_amplitude', 0.3)
    
    # Fixed naming parameters
    station = 'SYN'
    network = 'XX'
    output_dir = '.'
    
    print(f"\nConfiguration:")
    print(f"  Number of seismograms: {num_seismograms}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  P-wave frequency: {p_frequency} Hz")
    print(f"  S-wave frequency: {s_frequency} Hz")
    if randomize_noise:
        print(f"  Noise level: RANDOMIZED [{noise_level_range[0]:.2f} - {noise_level_range[1]:.2f}]")
        print(f"  SNR range: {10 * np.log10(1.0 / noise_level_range[1]):.1f} - {10 * np.log10(1.0 / noise_level_range[0]):.1f} dB")
    else:
        print(f"  Noise level: {noise_level}")
        print(f"  SNR: {10 * np.log10(1.0 / noise_level):.1f} dB")
    print("\n" + "=" * 70)
    
    # Generate seismograms
    metadata_list = []
    
    iterator = range(1, num_seismograms + 1)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Generating seismograms")
    
    for i in iterator:
        # Create unique event ID
        event_id = f"SYNTHETIC_{i:03d}"
        
        # Print progress if tqdm not available
        if not HAS_TQDM and i % max(1, num_seismograms // 10) == 0:
            print(f"  Progress: {i}/{num_seismograms} seismograms generated...")
        
        # Generate random arrival times
        p_time, s_time = get_random_times(duration, pre_arrival_time, window_length)
        
        # Randomize noise level if enabled
        if randomize_noise:
            event_noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])
        else:
            event_noise_level = noise_level
        
        # Generate 3C event
        stream, time = generate_3c_event(
            duration=duration,
            sample_rate=sample_rate,
            p_time=p_time,
            s_time=s_time,
            noise_level=event_noise_level,
            p_frequency=p_frequency,
            s_frequency=s_frequency,
            coda_amplitude=coda_amplitude,
            station=station,
            network=network
        )
        
        # Save to files
        file_paths = save_seismogram(
            stream=stream,
            event_id=event_id,
            station=station,
            network=network,
            p_time=p_time,
            s_time=s_time,
            sample_rate=sample_rate,
            duration=duration,
            noise_level=event_noise_level,
            output_dir=output_dir
        )
        
        # Store metadata for summary
        metadata = {
            "event_id": event_id,
            "p_arrival_time": float(p_time),
            "s_arrival_time": float(s_time),
            "noise_level": float(event_noise_level),
            "snr_db": float(10 * np.log10(1.0 / event_noise_level)) if event_noise_level > 0 else float('inf'),
            "files": file_paths
        }
        metadata_list.append(metadata)
    
    # Generate batch summary
    generate_batch_summary(metadata_list, output_dir)
    
    print("\n" + "=" * 70)
    print(f"✓ Successfully generated {num_seismograms} 3C seismograms")
    print(f"  Total files created: {num_seismograms * 5} (3 MSEED + 1 NPY + 1 JSON per event)")
    print("=" * 70)


if __name__ == "__main__":
    main()
