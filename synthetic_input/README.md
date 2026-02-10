# Synthetic Seismogram Generation

This directory contains scripts for generating realistic 3-component synthetic seismograms for training and testing seismic phase picking models.

## Scripts

### 1. `generate_synthetic_3c_seismogram.py`
Generates a **single** 3-component seismogram with visualization.

**Usage:**
```bash
python generate_synthetic_3c_seismogram.py
```

**Output:**
- 3 MiniSEED files (one per channel: HHZ, HHN, HHE)
- 1 NumPy array file (.npy) with stacked 3C data
- 1 metadata JSON file with arrival times and parameters
- 1 plot image showing waveforms and phase windows

### 2. `batch_generate_synthetic_3c.py` ⭐ **New**
Generates **multiple** 3-component seismograms for batch processing.

**Usage:**
```bash
python batch_generate_synthetic_3c.py
```

**Configuration:**
Edit `Syn_Config.json` to set the number of seismograms:
```json
{
    "num_seismograms": 10,
    "sample_rate": 100,
    "duration": 16.0,
    ...
}
```

**Output per seismogram:**
- 3 MiniSEED files: `SYNTHETIC_XXX_synthetic_SYN_XX_{HHZ,HHN,HHE}.mseed`
- 1 NumPy array: `SYNTHETIC_XXX_synthetic_SYN_XX_3C.npy` (shape: 3 × n_samples)
- 1 metadata JSON: `SYNTHETIC_XXX_synthetic_SYN_XX_metadata.json`

**Batch output:**
- `batch_summary.json`: Summary of all generated events with arrival times

## Configuration File

`Syn_Config.json` controls all generation parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_seismograms` | Number of seismograms to generate (batch mode) | 10 |
| `sample_rate` | Sampling rate in Hz | 100 |
| `duration` | Total duration in seconds | 16.0 |
| `p_frequency` | P-wave dominant frequency (Hz) | 20.0 |
| `s_frequency` | S-wave dominant frequency (Hz) | 10.0 |
| `noise_level` | Background noise amplitude (if not randomized) | 0.25 |
| `randomize_noise_level` | Enable random noise level per event | false |
| `noise_level_range` | Range for noise randomization [min, max] | [0.1, 0.4] |
| `coda_amplitude` | Coda wave amplitude | 0.3 |
| `window_length` | Phase window length (seconds) | 6.0 |
| `pre_arrival_time` | Pre-arrival buffer (seconds) | 2.0 |

### Noise Randomization

When `randomize_noise_level` is enabled, each seismogram gets a **unique random noise level** drawn uniformly from `noise_level_range`. This creates datasets with varying Signal-to-Noise Ratios (SNR) for more robust model training.

**Benefits:**
- **Realistic variability**: Real earthquakes have varying recording quality
- **Better generalization**: Models trained on varied SNR perform better on unseen data
- **Controlled difficulty**: Adjust range to include easy (low noise) to challenging (high noise) examples

**Example configuration:**
```json
{
    "noise_level": 0.25,
    "randomize_noise_level": true,
    "noise_level_range": [0.1, 0.4]
}
```

With this config:
- Noise levels vary from 0.1 (clean, ~10 dB SNR) to 0.4 (noisy, ~4 dB SNR)
- Each event's actual noise level is saved in its metadata JSON
- Batch summary includes SNR for each event

## Data Format

### NumPy Arrays (.npy)
- **Shape**: `(3, n_samples)` where:
  - Row 0: Vertical (Z) component
  - Row 1: North (N) component  
  - Row 2: East (E) component
- **Normalization**: Raw amplitude (not normalized)
- **Sample Rate**: Specified in config (default: 100 Hz)

### MiniSEED Files
- Standard seismic data format compatible with ObsPy
- One file per channel (HHZ, HHN, HHE)
- Contains full metadata (station, network, timestamps)

### Metadata JSON
Each event includes:
- Event ID and station/network codes
- P and S arrival times (seconds and sample indices)
- Sampling rate and duration
- SNR in dB and noise level used
- File paths for associated data

## Physical Model

The synthetic seismograms simulate realistic earthquake recordings:

1. **Noise Model**: Coherent background noise across components (simulating ground coupling)
2. **P-Wave**: Primarily vertical motion (70-100% on Z, 10-40% on H)
3. **S-Wave**: Primarily horizontal motion (60-100% on N/E, 10-30% on Z)
4. **Coda Waves**: Scattered energy following S-arrival
5. **Realistic Envelopes**: Exponential decay with smooth onsets

## Example Workflow

### Generate 100 Training Examples
```bash
# 1. Edit config
cat > Syn_Config.json <<EOF
{
    "num_seismograms": 100,
    "sample_rate": 100,
    "duration": 16.0,
    "noise_level": 0.25,
    ...
}
EOF

# 2. Generate batch
python batch_generate_synthetic_3c.py

# 3. Check output
ls SYNTHETIC_*.npy | wc -l  # Should show 100
cat batch_summary.json       # View summary
```

### Load Data in Python
```python
import numpy as np
import json
from obspy import read

# Load NumPy array (fastest)
data_3c = np.load('SYNTHETIC_001_synthetic_SYN_XX_3C.npy')
print(f"Shape: {data_3c.shape}")  # (3, 1600)

# Load MiniSEED (full metadata preserved)
stream = read('SYNTHETIC_001_synthetic_SYN_XX_HHZ.mseed')
trace_z = stream[0]

# Load metadata
with open('SYNTHETIC_001_synthetic_SYN_XX_metadata.json') as f:
    metadata = json.load(f)
    
p_time = metadata['p_arrival_time']
s_time = metadata['s_arrival_time']
```

## Dependencies

- NumPy
- ObsPy
- SciPy (for signal processing)
- tqdm (optional, for progress bars)

Install with:
```bash
conda install numpy obspy scipy tqdm
```

## Notes

- **Random Arrivals**: P and S times are randomized within valid ranges each run
- **Polarization**: Randomized but physically realistic for each event
- **Repeatability**: Set `np.random.seed()` for reproducible datasets
- **File Naming**: Sequential event IDs (SYNTHETIC_001, SYNTHETIC_002, ...)
