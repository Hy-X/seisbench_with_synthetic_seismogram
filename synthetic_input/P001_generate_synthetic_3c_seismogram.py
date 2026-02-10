#!/usr/bin/env python3
"""
Generate realistic 3-component (Z, N, E) synthetic seismogram.

This script expands the single-channel synthesis to 3 components (Vertical, North, East),
simulating realistic polarization for P and S waves.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from typing import Tuple, List, Optional

# Re-use the existing configuration file
def load_config(config_path: str = 'Syn_Config.json') -> dict:
    """Load configuration from JSON file."""
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Extract constants
SAMPLE_RATE = CONFIG.get('sample_rate', 100)
DURATION = CONFIG.get('duration', 20.0)
P_FREQUENCY = CONFIG.get('p_frequency', 20.0)
S_FREQUENCY = CONFIG.get('s_frequency', 10.0)
NOISE_LEVEL = CONFIG.get('noise_level', 0.15)
WINDOW_LENGTH = CONFIG.get('window_length', 6.0)
PRE_ARRIVAL_TIME = CONFIG.get('pre_arrival_time', 2.0)
CODA_AMP = CONFIG.get('coda_amplitude', 0.3)


def get_random_times(duration: float, pre_time: float, window_length: float, min_sep: float = 1.0) -> Tuple[float, float]:
    """Generate random P and S times (reused logic)."""
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
    """Generate basic wave packet (reused logic)."""
    wave = np.zeros(n_samples)
    duration_after = (n_samples - arrival_sample) / sample_rate
    t = np.linspace(0, duration_after, n_samples - arrival_sample)
    
    envelope = np.exp(-decay_rate * t)
    phase = 2 * np.pi * dominant_freq * t
    
    onset_samples = int(0.5 * sample_rate / dominant_freq)
    if len(t) > onset_samples:
        onset_ramp = np.linspace(0, 1, onset_samples)
        envelope[:onset_samples] *= onset_ramp
    
    wave[arrival_sample:] = amplitude * envelope * np.sin(phase)
    return wave


def add_coda_waves(waveform: np.ndarray, s_arrival_sample: int, sample_rate: float, amplitude: float) -> np.ndarray:
    """Add coda noise to a waveform."""
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
    station: str = 'SYN',
    network: str = 'XX'
) -> Tuple[Stream, np.ndarray]:
    """
    Generate 3-component (Z, N, E) synthetic event.
    """
    n_samples = int(duration * sample_rate)
    time = np.arange(n_samples) / sample_rate
    
    # 1. Initialize Channels with consistent (correlated) noise
    channels = ['HHZ', 'HHN', 'HHE']
    data = {}
    
    # Generate coherent noise sources (simulating real ground motion)
    # Instead of independent white noise on each channel, we generate 
    # vector noise sources (e.g., Rayleigh/Love waves) acting on all channels.
    
    # Source 1: Low-frequency background (e.g., Microseisms)
    noise_master = np.random.randn(n_samples)
    window_size = 5
    noise_master = np.convolve(noise_master, np.ones(window_size)/window_size, mode='same')
    
    # Random projection of noise onto components (Coherence)
    # This ensures that if Z goes up, N/E move in a related pattern
    noise_pol_z = np.random.uniform(0.5, 1.0)
    noise_pol_n = np.random.uniform(0.3, 0.7) * np.random.choice([1, -1])
    noise_pol_e = np.random.uniform(0.3, 0.7) * np.random.choice([1, -1])
    
    # Add a smaller independent component to each (scattering/instrument noise)
    noise_indep_scale = 0.3 # 30% independent, 70% coherent
    
    data['HHZ'] = NOISE_LEVEL * (noise_master * noise_pol_z + np.random.randn(n_samples) * noise_indep_scale * 0.1)
    data['HHN'] = NOISE_LEVEL * (noise_master * noise_pol_n + np.random.randn(n_samples) * noise_indep_scale * 0.1)
    data['HHE'] = NOISE_LEVEL * (noise_master * noise_pol_e + np.random.randn(n_samples) * noise_indep_scale * 0.1)

    # 2. Define Polarization Factors (Randomized for variety)
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

    # 3. Generate Source Wavelets
    # P-Source
    p_source = generate_realistic_wave_packet(
        n_samples, p_sample, P_FREQUENCY, sample_rate, 
        amplitude=0.5, decay_rate=3.0
    )
    
    # S-Source
    s_source = generate_realistic_wave_packet(
        n_samples, s_sample, S_FREQUENCY, sample_rate,
        amplitude=1.0, decay_rate=1.5
    )

    # 4. Project onto Components
    # Z Channel
    data['HHZ'] += p_source * p_pol_z
    data['HHZ'] += s_source * s_pol_z
    
    # N Channel
    data['HHN'] += p_source * p_pol_n
    data['HHN'] += s_source * s_pol_n
    
    # E Channel
    data['HHE'] += p_source * p_pol_e
    data['HHE'] += s_source * s_pol_e
    
    # 5. Add Coda (to all channels, but mostly Horizontal)
    data['HHZ'] = add_coda_waves(data['HHZ'], s_sample, sample_rate, CODA_AMP * 0.5)
    data['HHN'] = add_coda_waves(data['HHN'], s_sample, sample_rate, CODA_AMP)
    data['HHE'] = add_coda_waves(data['HHE'], s_sample, sample_rate, CODA_AMP)

    # 6. Create Stream
    stream = Stream()
    
    # Random start time (e.g., from year 2000 to present) or fixed
    # For synthetic dataset, simpler to keep 0 or randomize. 
    # Let's keep 0 but allows main to override if needed.
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


def extract_3c_phase_window(
    stream: Stream,
    arrival_time: float,
    window_length: float,
    pre_arrival: float,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3C window with preserved component ratios.
    
    Returns:
        time_vec: Relative time array
        data_3c: Shape (3, n_samples) corresponds to Z, N, E
    """
    start_t = stream[0].stats.starttime + arrival_time - pre_arrival
    end_t = start_t + window_length
    
    # Slice stream
    st_slice = stream.slice(start_t, end_t)
    
    # Ensure correct order Z, N, E
    comps = ['HHZ', 'HHN', 'HHE']
    data_list = []
    
    sr = stream[0].stats.sampling_rate
    target_len = int(window_length * sr)
    
    for ch in comps:
        tr = st_slice.select(channel=ch)[0]
        d = tr.data.copy()
        
        # Pad/Trim
        if len(d) > target_len:
            d = d[:target_len]
        elif len(d) < target_len:
            d = np.pad(d, (0, target_len - len(d)), 'constant')
            
        data_list.append(d)
        
    data_3c = np.vstack(data_list) # Shape (3, N)
    
    # Global Max Normalization (Preserves relative amplitudes between channels)
    if normalize:
        peak = np.max(np.abs(data_3c))
        if peak > 0:
            data_3c /= peak
            
    t_vec = np.linspace(-pre_arrival, window_length - pre_arrival, target_len)
    return t_vec, data_3c


def plot_3c_seismogram(
    stream: Stream,
    time: np.ndarray,
    p_time: float,
    s_time: float,
    save_path: str = "synthetic_seismogram_3c.png"
):
    """Plot 3-component seismogram with zoomed P and S windows."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3) 
    # Schematic:
    # [ Full Z ] [ Full N ] [ Full E ]  <- Row 0
    # [  P Z   ] [  P N   ] [  P E   ]  <- Row 1 (Zoomed & Norm)
    # [  S Z   ] [  S N   ] [  S E   ]  <- Row 2 (Zoomed & Norm)
    
    comps = ['HHZ', 'HHN', 'HHE']
    colors = ['k', 'k', 'k']
    
    # 1. Full Waveforms
    for i, ch in enumerate(comps):
        ax = fig.add_subplot(gs[0, i])
        tr = stream.select(channel=ch)[0]
        ax.plot(time, tr.data, color=colors[i], linewidth=0.5)
        
        # Markers
        ax.axvline(p_time, color='b', linestyle='--', alpha=0.9, label='P')
        ax.axvline(s_time, color='r', linestyle='--', alpha=0.9, label='S')
        
        ax.set_title(f'Full Channel {ch} (Raw)', fontweight='bold')
        ax.set_xlim(0, time[-1])
        if i == 0: ax.set_ylabel('Amplitude')
        if i == 2: ax.legend(loc='upper right')

    # 2. P-Wave Window (Global Norm over 3C window)
    t_p, data_p = extract_3c_phase_window(stream, p_time, WINDOW_LENGTH, PRE_ARRIVAL_TIME)
    
    for i, ch in enumerate(comps):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t_p, data_p[i], color='b', linewidth=1.0)
        ax.axvline(0, color='k', linestyle=':', linewidth=1.5)
        
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'{ch} | P-Window ', color='blue', fontsize=10)
        if i == 0: ax.set_ylabel('Norm. Amplitude')
    
    # 3. S-Wave Window (Global Norm over 3C window)
    t_s, data_s = extract_3c_phase_window(stream, s_time, WINDOW_LENGTH, PRE_ARRIVAL_TIME)
    
    for i, ch in enumerate(comps):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(t_s, data_s[i], color='r', linewidth=1.0)
        ax.axvline(0, color='k', linestyle=':', linewidth=1.5)
        
        # Show ghost P if in window
        p_rel = p_time - s_time
        if t_s[0] < p_rel < t_s[-1]:
             ax.axvline(p_rel, color='b', linestyle='--', alpha=0.3)
        
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'{ch} | S-Window', color='red', fontsize=10)
        ax.set_xlabel('Time (s)')
        if i == 0: ax.set_ylabel('Norm. Amplitude')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating 3C Synthetic Seismogram (Z, N, E)")
    print("=" * 60)

    # Naming Parameters
    EVENT_ID = "SYNTHETIC_001"
    STATION = "HX"
    NETWORK = "QD"
    SYN_TAG = "synthetic"
    
    # Random location in Oklahoma
    # Oklahoma boundaries: Lat 33.6° to 37.0° N, Lon -103.0° to -94.4° W
    station_latitude = np.random.uniform(33.6, 37.0)
    station_longitude = np.random.uniform(-103.0, -94.4)
    
    p_time, s_time = get_random_times(DURATION, PRE_ARRIVAL_TIME, WINDOW_LENGTH)
    
    print(f"P-Arrival: {p_time:.2f} s")
    print(f"S-Arrival: {s_time:.2f} s")
    
    stream, time = generate_3c_event(
        DURATION, 
        SAMPLE_RATE, 
        p_time, 
        s_time, 
        station=STATION, 
        network=NETWORK
    )
    
    # Save Data
    base_name = f"{EVENT_ID}_{SYN_TAG}_{STATION}_{NETWORK}"

    # 1. MiniSEED (Separate files per channel)
    # Naming: [eventid]_[syn]_[station]_[network]_[channel].mseed
    for tr in stream:
        channel = tr.stats.channel
        filename = f"{base_name}_{channel}.mseed"
        tr.write(filename, format="MSEED")
        print(f"[✓] Saved {filename}")
    
    # 2. NumPy (Stacked)
    # Naming: [base_name]_3C.npy
    data_3c = np.vstack([tr.data for tr in stream])
    npy_filename = f"{base_name}_3C.npy"
    np.save(npy_filename, data_3c)
    print(f"[✓] Saved {npy_filename} (Shape: {data_3c.shape})")
    
    # 3. Metadata (Labels JSON)
    metadata = {
        "event_id": EVENT_ID,
        "station": STATION,
        "network": NETWORK,
        "station_latitude": round(station_latitude, 4),
        "station_longitude": round(station_longitude, 4),
        "start_time": stream[0].stats.starttime.isoformat(),
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "p_arrival_sample": int(p_time * SAMPLE_RATE),
        "s_arrival_sample": int(s_time * SAMPLE_RATE),
        "p_arrival_time": p_time,
        "s_arrival_time": s_time,
        "snr_db": 10 * np.log10(1.0 / NOISE_LEVEL),
        "channels": ["HHZ", "HHN", "HHE"],
        "files": {
            "mseed": [f"{base_name}_{ch}.mseed" for ch in ["HHZ", "HHN", "HHE"]],
            "npy": npy_filename
        }
    }
    
    json_filename = f"{base_name}_metadata.json"
    with open(json_filename, "w") as f:
        json.dump(metadata, f, indent=4)
        print(f"[✓] Saved {json_filename}")

    # Plot
    plot_filename = f"{base_name}_plot.png"
    plot_3c_seismogram(stream, time, p_time, s_time, save_path=plot_filename)
    print("=" * 60)

if __name__ == "__main__":
    main()
