#!/usr/bin/env python3
"""
Create visualization comparing different noise levels using the SeisBench dataset.

This script loads the processed SeisBench dataset from the ../data directory,
sorts events by SNR, and creates a comparison plot showing low, medium, and
high noise examples.

Usage:
    python V002_plot_noise_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd

def main():
    # Load dataset
    data_path = '../data'
    print(f"Loading dataset from {data_path}...")
    try:
        dataset = sbd.WaveformDataset(data_path, sampling_rate=100)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    metadata = dataset.metadata
    print(f"Found {len(metadata)} traces.")

    if len(metadata) < 3:
        print("Error: Need at least 3 traces.")
        return

    # Sort by SNR (Higher SNR = Lower Noise)
    # Ensure numeric
    metadata['trace_snr_db'] = pd.to_numeric(metadata['trace_snr_db'])
    sorted_meta = metadata.sort_values('trace_snr_db', ascending=False)
    
    # Select examples: High SNR (Low Noise), Medium, Low SNR (High Noise)
    indices = [0, len(sorted_meta)//2, len(sorted_meta)-1]
    labels = ['Low Noise (High SNR)', 'Medium Noise', 'High Noise (Low SNR)']
    
    # V001 Styling
    z_color = '#344e41'  # Dark green from V001
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    
    for i, (idx, label) in enumerate(zip(indices, labels)):
        # Get trace index (iloc gives row, need original index for get_waveforms)
        trace_idx = sorted_meta.index[idx]
        row = sorted_meta.iloc[idx]
        
        # Get waveform (Z component assumed index 0)
        waveforms = dataset.get_waveforms(trace_idx)
        z_trace = waveforms[0]
        
        # Time axis
        sr = row['trace_sampling_rate_hz']
        time = np.arange(len(z_trace)) / sr
        
        ax = axes[i]
        ax.plot(time, z_trace, color=z_color, linewidth=1.5)
        
        # Plot phase arrivals
        if 'trace_p_arrival_sample' in row and not pd.isna(row['trace_p_arrival_sample']):
            p_time = row['trace_p_arrival_sample'] / sr
            ax.axvline(p_time, color="#d33b14", linestyle='--', linewidth=2, label="P-arrival" if i == 0 else "")
            ax.text(p_time, ax.get_ylim()[1], "P", color="#ab2838", ha="center", va="bottom", fontweight="bold")

        if 'trace_s_arrival_sample' in row and not pd.isna(row['trace_s_arrival_sample']):
            s_time = row['trace_s_arrival_sample'] / sr
            ax.axvline(s_time, color='#ff7d00', linestyle='--', linewidth=2, label="S-arrival" if i == 0 else "")
            ax.text(s_time, ax.get_ylim()[1], "S", color="#f07f14", ha="center", va="bottom", fontweight="bold")
        
        # Labels and title
        snr = row['trace_snr_db']
        eid = row['source_id']
        ax.set_ylabel('Vertical (Z)\nAmplitude', fontsize=10)
        ax.set_title(f"{label}: SNR={snr:.1f} dB | ID: {eid}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_xlim(0, time[-1])
        
        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
            
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    fig.suptitle('Seismic Noise Level Comparison', fontsize=14, fontweight='bold')
    
    plt.savefig('noise_comparison.png', dpi=150)
    print("Saved noise_comparison.png")

if __name__ == "__main__":
    main()
