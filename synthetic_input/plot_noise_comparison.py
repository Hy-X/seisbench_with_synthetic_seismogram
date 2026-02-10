#!/usr/bin/env python3
"""
Create visualization comparing different noise levels.

This script is self-contained and automatically discovers all synthetic
seismogram files in the current directory. It loads individual metadata
files, sorts events by noise level, and creates a comparison plot showing
low, medium, and high noise examples.

Usage:
    python plot_noise_comparison.py

Requirements:
    - SYNTHETIC_*_3C.npy files (seismogram data)
    - SYNTHETIC_*_metadata.json files (event metadata)
    - numpy, matplotlib

No dependency on batch_summary.json.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

# Discover all synthetic seismogram files
npy_files = sorted(glob.glob('SYNTHETIC_*_3C.npy'))

if len(npy_files) == 0:
    print("Error: No synthetic seismogram files found (SYNTHETIC_*_3C.npy)")
    print("Please run batch_generate_synthetic_3c.py first.")
    exit(1)

print(f"Found {len(npy_files)} synthetic seismograms")

# Load metadata for each event
events = []
for npy_file in npy_files:
    # Extract event ID from filename
    # Format: SYNTHETIC_XXX_synthetic_SYN_XX_3C.npy
    event_id = npy_file.replace('_3C.npy', '').replace('_synthetic_SYN_XX', '')
    
    # Load corresponding metadata JSON
    json_file = npy_file.replace('_3C.npy', '_metadata.json')
    
    if not os.path.exists(json_file):
        print(f"Warning: Metadata file not found for {npy_file}, skipping...")
        continue
    
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    events.append({
        'event_id': event_id,
        'npy_file': npy_file,
        'noise_level': metadata.get('snr_db', 0),  # Use SNR to infer noise if available
        'snr_db': metadata.get('snr_db', 0),
        'metadata': metadata
    })

# Calculate noise level from SNR if not directly stored
for event in events:
    snr_db = event['snr_db']
    # SNR (dB) = 10 * log10(1 / noise_level)
    # noise_level = 1 / 10^(SNR/10)
    if snr_db > 0:
        event['noise_level'] = 1.0 / (10 ** (snr_db / 10))
    else:
        event['noise_level'] = 0.25  # Default fallback

print(f"Loaded metadata for {len(events)} events")

# Sort by noise level
events_sorted = sorted(events, key=lambda x: x['noise_level'])

# Pick low, medium, high noise examples
low_noise = events_sorted[0]
mid_noise = events_sorted[len(events_sorted)//2]
high_noise = events_sorted[-1]

examples = [low_noise, mid_noise, high_noise]
labels = ['Low Noise', 'Medium Noise', 'High Noise']

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (event, label) in enumerate(zip(examples, labels)):
    event_id = event['event_id']
    npy_file = event['npy_file']
    noise = event['noise_level']
    snr = event['snr_db']
    
    # Load data
    data = np.load(npy_file)
    time = np.arange(data.shape[1]) / 100.0
    
    # Plot Z component
    ax = axes[i]
    ax.plot(time, data[0], 'k-', linewidth=0.5)
    ax.set_ylabel('Vertical (Z)', fontweight='bold')
    ax.set_title(f'{label}: Noise={noise:.3f}, SNR={snr:.1f} dB ({event_id})', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1])

axes[-1].set_xlabel('Time (s)', fontweight='bold')
plt.suptitle('Effect of Noise Level Randomization', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved noise_comparison.png')
print(f'  Low noise:  {examples[0]["event_id"]:15s} Noise={examples[0]["noise_level"]:.4f} (SNR: {examples[0]["snr_db"]:.1f} dB)')
print(f'  Mid noise:  {examples[1]["event_id"]:15s} Noise={examples[1]["noise_level"]:.4f} (SNR: {examples[1]["snr_db"]:.1f} dB)')
print(f'  High noise: {examples[2]["event_id"]:15s} Noise={examples[2]["noise_level"]:.4f} (SNR: {examples[2]["snr_db"]:.1f} dB)')
