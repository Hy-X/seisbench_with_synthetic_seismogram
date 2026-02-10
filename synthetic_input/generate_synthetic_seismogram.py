#!/usr/bin/env python3
"""
Generate realistic synthetic seismogram with P and S wave arrivals using ObsPy.

This script creates a 20-second synthetic waveform at 100 Hz sampling rate
with simulated P and S phase arrivals using realistic seismic wave modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from scipy.signal import gausspulse
from typing import Tuple, Optional


# Configuration constants
SAMPLE_RATE = 100  # Hz
DURATION = 20.0  # seconds
P_ARRIVAL_TIME = 5.0  # seconds
S_ARRIVAL_TIME = 10.0  # seconds
P_FREQUENCY = 8.0  # Hz (typical for local earthquakes)
S_FREQUENCY = 4.0  # Hz (typically lower than P)
NOISE_LEVEL = 0.05  # Background noise amplitude relative to signal


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
    amplitude: float = 0.3
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
        
        # Multiple frequency components with decay
        coda = np.zeros(coda_length)
        for freq in [2, 3, 5, 7, 10]:  # Multiple frequency components
            phase = 2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)
            decay = np.exp(-0.2 * t)
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
        amplitude=0.4, decay_rate=0.5
    )
    waveform += p_wave
    
    # Add S-wave arrival (larger amplitude, lower frequency, slower decay)
    s_wave = generate_realistic_wave_packet(
        n_samples, s_sample, s_freq, sample_rate,
        amplitude=1.0, decay_rate=0.3
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


def plot_seismogram(
    trace: Trace,
    time: np.ndarray,
    p_time: float = P_ARRIVAL_TIME,
    s_time: float = S_ARRIVAL_TIME,
    save_path: Optional[str] = None
) -> None:
    """
    Plot realistic synthetic seismogram with phase markers and STA/LTA.
    
    Creates a publication-quality plot showing:
    - Raw waveform
    - P and S arrival markers
    - Optional STA/LTA characteristic function
    
    Args:
        trace: ObsPy Trace object
        time: Time array in seconds
        p_time: P-wave arrival time in seconds
        s_time: S-wave arrival time in seconds
        save_path: Optional path to save figure (e.g., 'seismogram.png')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), 
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    waveform = trace.data
    
    # Plot waveform
    ax1.plot(time, waveform, 'k-', linewidth=0.7, label='Waveform')
    
    # Mark P and S arrivals
    ax1.axvline(p_time, color='blue', linestyle='--', linewidth=2, 
                alpha=0.7, label='P-wave arrival')
    ax1.axvline(s_time, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='S-wave arrival')
    
    # Labels and formatting
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Synthetic Seismogram - {trace.stats.station}.{trace.stats.channel} | '
                  f'{trace.stats.sampling_rate} Hz', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0, time[-1])
    
    # Plot STA/LTA characteristic function
    sta = 0.5  # seconds
    lta = 5.0  # seconds
    cft = recursive_sta_lta(waveform, int(sta * trace.stats.sampling_rate), 
                            int(lta * trace.stats.sampling_rate))
    
    ax2.plot(time, cft, 'g-', linewidth=0.7, label='STA/LTA')
    ax2.axvline(p_time, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(s_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(2.0, color='orange', linestyle=':', linewidth=1.5, 
                label='Trigger threshold', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('STA/LTA Ratio', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(0, time[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close()  # Close figure to prevent blocking
    else:
        plt.show()


def main():
    """
    Main function to generate and visualize realistic synthetic seismogram.
    """
    print("=" * 60)
    print("Generating Realistic Synthetic Seismogram with ObsPy")
    print("=" * 60)
    print(f"Duration: {DURATION} s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"P-wave arrival: {P_ARRIVAL_TIME} s @ {P_FREQUENCY} Hz")
    print(f"S-wave arrival: {S_ARRIVAL_TIME} s @ {S_FREQUENCY} Hz")
    print(f"Total samples: {int(DURATION * SAMPLE_RATE)}")
    
    # Generate synthetic data
    trace, time = generate_synthetic_seismogram()
    
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
    plot_seismogram(trace, time, save_path='synthetic_seismogram.png')
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
