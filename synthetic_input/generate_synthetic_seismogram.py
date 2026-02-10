#!/usr/bin/env python3
"""
Generate realistic synthetic seismogram with P and S wave arrivals using ObsPy.

This script creates a 20-second synthetic waveform at 100 Hz sampling rate
with simulated P and S phase arrivals using realistic seismic wave modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from scipy.signal import gausspulse
from typing import Tuple, Optional


# Load Configuration
def load_config(config_path: str = 'Syn_Config.json') -> dict:
    """Load configuration from JSON file."""
    if not os.path.isabs(config_path):
        # Resolve relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Extract constants for default values
SAMPLE_RATE = CONFIG.get('sample_rate', 100)
DURATION = CONFIG.get('duration', 20.0)
P_ARRIVAL_TIME = CONFIG.get('p_arrival_time', 5.0)
S_ARRIVAL_TIME = CONFIG.get('s_arrival_time', 10.0)
P_FREQUENCY = CONFIG.get('p_frequency', 20.0)
S_FREQUENCY = CONFIG.get('s_frequency', 10.0)
NOISE_LEVEL = CONFIG.get('noise_level', 0.15)
WINDOW_LENGTH = CONFIG.get('window_length', 6.0)
PRE_ARRIVAL_TIME = CONFIG.get('pre_arrival_time', 2.0)


def generate_realistic_wave_packet(
    n_samples: int,
    arrival_sample: int,
    dominant_freq: float,
    sample_rate: float,
    amplitude: float = 1.0,
    decay_rate: float = 0.3
) -> np.ndarray:
    """
    Generate realistic seismic wave packet with exponential decay.
    
    Uses a Gaussian-modulated sinusoid (Gabor wavelet) with exponential
    decay to simulate realistic seismic wave arrival and attenuation.
    
    Args:
        n_samples: Total number of samples
        arrival_sample: Sample index of wave arrival
        dominant_freq: Dominant frequency in Hz
        sample_rate: Sampling rate in Hz
        amplitude: Peak amplitude of the wave packet
        decay_rate: Exponential decay rate
        
    Returns:
        Wave packet array of shape (n_samples,)
    """
    wave = np.zeros(n_samples)
    
    # Generate wave packet starting at arrival
    duration_after = (n_samples - arrival_sample) / sample_rate
    t = np.linspace(0, duration_after, n_samples - arrival_sample)
    
    # Gaussian-modulated sinusoid with exponential decay
    # This simulates realistic seismic wave attenuation
    envelope = np.exp(-decay_rate * t)
    phase = 2 * np.pi * dominant_freq * t
    
    # Add onset ramp (not instantaneous)
    onset_samples = int(0.5 * sample_rate / dominant_freq)  # ~half period
    if len(t) > onset_samples:
        onset_ramp = np.linspace(0, 1, onset_samples)
        envelope[:onset_samples] *= onset_ramp
    
    wave[arrival_sample:] = amplitude * envelope * np.sin(phase)
    
    return wave


def add_coda_waves(
    waveform: np.ndarray,
    s_arrival_sample: int,
    sample_rate: float,
    amplitude: float = CONFIG.get('coda_amplitude', 0.3)
) -> np.ndarray:
    """
    Add realistic coda waves (scattered waves after S arrival).
    
    Coda waves are the scattered and reflected waves that arrive after
    the main P and S phases, creating a gradual decay in amplitude.
    
    Args:
        waveform: Input waveform array
        s_arrival_sample: Sample index of S-wave arrival
        sample_rate: Sampling rate in Hz
        amplitude: Coda amplitude relative to main signal
        
    Returns:
        Waveform with added coda waves
    """
    n_samples = len(waveform)
    coda_start = s_arrival_sample + int(2 * sample_rate)  # 2 sec after S
    
    if coda_start < n_samples:
        coda_length = n_samples - coda_start
        t = np.arange(coda_length) / sample_rate
        
        # Multiple frequency components with decay (higher freq for small events)
        coda = np.zeros(coda_length)
        for freq in [5, 8, 12, 15, 20]:  # Higher frequency components
            phase = 2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)
            decay = np.exp(-0.3 * t)  # Faster decay for small events
            coda += np.sin(phase) * decay
        
        coda *= amplitude / 5  # Normalize by number of components
        waveform[coda_start:] += coda
    
    return waveform


def generate_synthetic_seismogram(
    duration: float = DURATION,
    sample_rate: float = SAMPLE_RATE,
    p_time: float = P_ARRIVAL_TIME,
    s_time: float = S_ARRIVAL_TIME,
    p_freq: float = P_FREQUENCY,
    s_freq: float = S_FREQUENCY,
    noise_level: float = NOISE_LEVEL,
    add_coda: bool = True
) -> Tuple[Trace, np.ndarray]:
    """
    Generate realistic synthetic seismogram with P and S wave arrivals.
    
    Creates a realistic synthetic waveform using ObsPy Trace object with:
    - Background seismic noise (colored noise, not white)
    - P-wave arrival with realistic onset and decay
    - S-wave arrival with larger amplitude and lower frequency
    - Optional coda waves (scattered energy)
    - Proper metadata (station, channel, sampling rate, etc.)
    
    Args:
        duration: Total duration in seconds
        sample_rate: Sampling rate in Hz
        p_time: P-wave arrival time in seconds
        s_time: S-wave arrival time in seconds
        p_freq: P-wave dominant frequency in Hz
        s_freq: S-wave dominant frequency in Hz
        noise_level: Background noise level relative to signal
        add_coda: Whether to add realistic coda waves
        
    Returns:
        trace: ObsPy Trace object with waveform and metadata
        time: Time array in seconds, shape (n_samples,)
        
    Raises:
        ValueError: If S-wave arrives before P-wave
    """
    if s_time <= p_time:
        raise ValueError(f"S-wave time ({s_time}s) must be after P-wave time ({p_time}s)")
    
    # Generate time array
    n_samples = int(duration * sample_rate)
    time = np.arange(n_samples) / sample_rate
    
    # Generate realistic background noise (low-pass filtered for seismic band)
    noise = np.random.randn(n_samples)
    # Apply simple low-pass filter (moving average) for realistic seismic noise
    window_size = 5
    noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    waveform = noise_level * noise
    
    # Calculate arrival samples
    p_sample = int(p_time * sample_rate)
    s_sample = int(s_time * sample_rate)
    
    # Add P-wave arrival (smaller amplitude, higher frequency)
    p_wave = generate_realistic_wave_packet(
        n_samples, p_sample, p_freq, sample_rate, 
        amplitude=0.5, decay_rate=3.0  # Fast decay for impulsive source
    )
    waveform += p_wave
    
    # Add S-wave arrival (larger amplitude, lower frequency, slower decay)
    s_wave = generate_realistic_wave_packet(
        n_samples, s_sample, s_freq, sample_rate,
        amplitude=1.0, decay_rate=1.5  # Fast decay for impulsive source
    )
    waveform += s_wave
    
    # Add coda waves for realism
    if add_coda:
        waveform = add_coda_waves(waveform, s_sample, sample_rate)
    
    # Create ObsPy Trace with proper metadata
    trace = Trace(data=waveform)
    trace.stats.sampling_rate = sample_rate
    trace.stats.station = 'SYN'
    trace.stats.channel = 'HHZ'  # High-gain, High-broadband, Vertical
    trace.stats.network = 'XX'
    trace.stats.starttime = UTCDateTime(0)
    
    return trace, time


def extract_phase_window(
    trace: Trace,
    arrival_time: float,
    window_length: float = WINDOW_LENGTH,
    pre_arrival_time: float = PRE_ARRIVAL_TIME,
    normalize: str = 'max'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a short window centered around a phase arrival with normalization.
    
    Args:
        trace: Full waveform trace
        arrival_time: Phase arrival time in seconds
        window_length: Length of window in seconds
        pre_arrival_time: Time to include before the arrival
        normalize: Normalization strategy ('max', 'zscore', 'none')
        
    Returns:
        t_window: Relative time array (0 at arrival)
        data_window: Waveform slice (normalized)
    """
    dt = trace.stats.sampling_rate
    sr = trace.stats.sampling_rate
    
    start_time = arrival_time - pre_arrival_time
    end_time = start_time + window_length
    
    # Slice using ObsPy's time handling (handles sub-sample precision)
    slice_trace = trace.slice(
        starttime=trace.stats.starttime + start_time,
        endtime=trace.stats.starttime + end_time
    )
    
    # Ensure exact sample count
    target_samples = int(window_length * sr)
    data = slice_trace.data.copy()  # Copy to avoid modifying original trace
    
    if len(data) != target_samples:
        if len(data) > target_samples:
            data = data[:target_samples]
        else:
            data = np.pad(data, (0, target_samples - len(data)), 'constant')
            
    # Apply normalization relative to this specific window
    if normalize == 'max':
        peak_amp = np.max(np.abs(data))
        if peak_amp > 0:
            data = data / peak_amp
    elif normalize == 'zscore':
        std_val = np.std(data)
        if std_val > 0:
            data = (data - np.mean(data)) / std_val
            
    # Create relative time vector (0 is the arrival)
    t_vec = np.linspace(-pre_arrival_time, window_length - pre_arrival_time, len(data))
    
    return t_vec, data


def plot_seismogram(
    trace: Trace,
    time: np.ndarray,
    p_time: float = P_ARRIVAL_TIME,
    s_time: float = S_ARRIVAL_TIME,
    save_path: Optional[str] = None
) -> None:
    """
    Plot full seismogram and focused phase windows (normalized independently).
    """
    # Create figure with 3 panels: Full trace, P-window, S-window
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    ax_full = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    
    # 1. Full Waveform (Raw Amplitude)
    ax_full.plot(time, trace.data, 'k-', linewidth=0.5, label='Full Trace (Raw)')
    ax_full.axvline(p_time, color='blue', linestyle='--', alpha=0.7, label='P')
    ax_full.axvline(s_time, color='red', linestyle='--', alpha=0.7, label='S')
    ax_full.set_title(f'Full Synthetic Event ({DURATION}s)', fontweight='bold')
    ax_full.set_ylabel('Raw Amplitude')
    ax_full.legend(loc='upper right')
    ax_full.set_xlim(0, time[-1])
    
    # 2. P-Wave Window (Independently Normalized)
    t_p, data_p = extract_phase_window(trace, p_time, normalize='max')
    ax_p.plot(t_p, data_p, 'b-', linewidth=1)
    ax_p.axvline(0, color='k', linestyle=':', label='Arrival')
    ax_p.set_title('P-Wave Window (Max Normalized)', color='blue', fontweight='bold')
    ax_p.set_xlabel('Time relative to P (s)')
    ax_p.set_ylabel('Norm. Amplitude')
    ax_p.set_ylim(-1.1, 1.1)
    ax_p.grid(True, alpha=0.3)
    
    # 3. S-Wave Window (Independently Normalized)
    t_s, data_s = extract_phase_window(trace, s_time, normalize='max')
    ax_s.plot(t_s, data_s, 'r-', linewidth=1)
    ax_s.axvline(0, color='k', linestyle=':', label='Arrival')
    
    # Check if P-wave is visible in S-window (if S-P < 2s)
    p_relative = p_time - s_time
    if t_s[0] < p_relative < t_s[-1]:
         ax_s.axvline(p_relative, color='blue', linestyle='--', alpha=0.5, label='P-Leakage')

    ax_s.set_title('S-Wave Window (Max Normalized)', color='red', fontweight='bold')
    ax_s.set_xlabel('Time relative to S (s)')
    ax_s.set_ylabel('Norm. Amplitude')
    ax_s.set_ylim(-1.1, 1.1)
    ax_s.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def get_random_times(
    duration: float, 
    pre_time: float, 
    window_length: float,
    min_sep: float = 1.0
) -> Tuple[float, float]:
    """
    Generate random P and S times within safe margins for window extraction.
    
    Args:
        duration: Total signal duration
        pre_time: Pre-arrival time required in window
        window_length: Total length of phase window
        min_sep: Minimum separation between P and S
        
    Returns:
        p_time, s_time
    """
    # Safety margin: ensure we can extract full window at start and end
    # We need: time >= pre_time
    # We need: time + (window_length - pre_time) <= duration
    
    post_time = window_length - pre_time
    start_margin = pre_time + 0.1  # small buffer
    end_margin = post_time + 0.1   # small buffer
    
    # P Range
    # Earliest P: start_margin
    # Latest P: duration - end_margin - min_sep (leaving room for S)
    latest_p = duration - end_margin - min_sep
    
    if latest_p <= start_margin:
        print("Warning: Duration too tight for random margins. Using center.")
        return duration * 0.3, duration * 0.6
        
    p_time = np.random.uniform(start_margin, latest_p)
    
    # S Range
    # Earliest S: p_time + min_sep
    # Latest S: duration - end_margin
    latest_s = duration - end_margin
    earliest_s = p_time + min_sep
    
    if latest_s <= earliest_s:
         s_time = latest_s
    else:
         s_time = np.random.uniform(earliest_s, latest_s)
         
    return p_time, s_time


def main():
    """
    Main function to generate and visualize realistic synthetic seismogram.
    """
    print("=" * 60)
    print("Generating Realistic Synthetic Seismogram with ObsPy")
    print("=" * 60)
    
    # Randomize timestamps
    p_time, s_time = get_random_times(DURATION, PRE_ARRIVAL_TIME, WINDOW_LENGTH)
    
    print(f"Duration: {DURATION} s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Randomized P-wave arrival: {p_time:.2f} s")
    print(f"Randomized S-wave arrival: {s_time:.2f} s")
    print(f"P-S Separation: {s_time - p_time:.2f} s")
    print(f"Total samples: {int(DURATION * SAMPLE_RATE)}")
    
    # Generate synthetic data
    trace, time = generate_synthetic_seismogram(
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        p_time=p_time,
        s_time=s_time,
        p_freq=P_FREQUENCY,
        s_freq=S_FREQUENCY,
        noise_level=NOISE_LEVEL
    )
    
    # Create stream (container for traces)
    stream = Stream(traces=[trace])
    
    # Save waveform in multiple formats
    
    # 1. Save as NumPy array
    np.save('synthetic_seismogram.npy', trace.data)
    print(f"\n[✓] NumPy array saved: synthetic_seismogram.npy")
    
    # 2. Save as MiniSEED (standard seismological format)
    stream.write('synthetic_seismogram.mseed', format='MSEED')
    print(f"[✓] MiniSEED saved: synthetic_seismogram.mseed")
    
    # 3. Save as SAC (Seismic Analysis Code format)
    stream.write('synthetic_seismogram.sac', format='SAC')
    print(f"[✓] SAC format saved: synthetic_seismogram.sac")
    
    # Print trace information
    print(f"\n{trace}")
    print(f"\nTrace stats:")
    print(f"  Station: {trace.stats.station}")
    print(f"  Channel: {trace.stats.channel}")
    print(f"  Sampling rate: {trace.stats.sampling_rate} Hz")
    print(f"  Number of samples: {trace.stats.npts}")
    print(f"  Duration: {trace.stats.npts / trace.stats.sampling_rate:.1f} s")
    
    # Print waveform statistics
    print(f"\nWaveform statistics:")
    print(f"  Mean: {np.mean(trace.data):.4f}")
    print(f"  Std: {np.std(trace.data):.4f}")
    print(f"  Min: {np.min(trace.data):.4f}")
    print(f"  Max: {np.max(trace.data):.4f}")
    print(f"  Peak-to-peak: {np.ptp(trace.data):.4f}")
    
    # Plot the seismogram
    print(f"\nGenerating plot...")
    plot_seismogram(
        trace, time, 
        p_time=p_time, 
        s_time=s_time, 
        save_path='synthetic_seismogram.png'
    )
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
