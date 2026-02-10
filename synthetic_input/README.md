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

### 3. `pack_to_seisbench.py` ⭐ **New**
Converts synthetic seismograms into **SeisBench HDF5 and CSV format** for integration with SeisBench models.

**Usage:**
```bash
python pack_to_seisbench.py
```

**Output:**
- `synthetic_dataset.hdf5`: Waveform data in HDF5 format
  - Each trace stored as 3C array (Z, N, E)
  - Includes metadata attributes (sampling rate, picks, SNR)
  - Compressed for efficient storage
- `synthetic_metadata.csv`: Phase picks and metadata in CSV format
  - Compatible with SeisBench dataset conventions
  - Includes P/S arrival samples, times, SNR, pick weights

**Loading with SeisBench:**
```python
import seisbench.data
data = seisbench.data.WaveformDataset('synthetic_dataset.hdf5', 'synthetic_metadata.csv')
waveforms, metadata = data.get_idx(0)
```

### 4. `test_seisbench_dataset.py`
Tests and visualizes the packed SeisBench dataset.

**Usage:**
```bash
python test_seisbench_dataset.py
```

**Output:**
- Loads HDF5 and CSV files
- Prints dataset statistics
- Generates example plots showing low/medium/high SNR traces with phase picks

### 5. `plot_noise_comparison.py`
Creates visualization comparing different noise levels (self-contained).

**Usage:**
```bash
python plot_noise_comparison.py
```

**Output:**
- `noise_comparison.png`: Side-by-side comparison of low/medium/high noise examples

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

### SeisBench HDF5 Format
- **HDF5 File**: Hierarchical data structure
  - Each trace stored as dataset named by `trace_name` (event_id)
  - Dataset shape: `(3, n_samples)` for Z, N, E components
  - Dataset attributes: sampling_rate, station, network, picks, SNR
  - Compressed with gzip (level 4) for efficient storage
  - Component order: ZNE (vertical, north, east)

- **CSV Metadata**: Tabular format with columns:
  - `trace_name`: Unique trace identifier
  - `station_code`, `network_code`: Station/network codes
  - `trace_sampling_rate_hz`: Sampling rate
  - `trace_npts`: Number of samples
  - `trace_p_arrival_sample`, `trace_s_arrival_sample`: Pick samples
  - `trace_p_arrival_time`, `trace_s_arrival_time`: Pick times in seconds
  - `trace_p_status`, `trace_s_status`: Pick quality ('manual' for synthetic)
  - `trace_p_weight`, `trace_s_weight`: Pick confidence (1.0 = high)
  - `snr_db`: Signal-to-noise ratio
  - `source_type`, `source_id`: Source information

**Compatibility**: Format follows SeisBench conventions for seamless integration with:
- PhaseNet
- EQTransformer  
- Other SeisBench-compatible models

## Physical Model

The synthetic seismograms simulate realistic earthquake recordings:

1. **Noise Model**: Coherent background noise across components (simulating ground coupling)
2. **P-Wave**: Primarily vertical motion (70-100% on Z, 10-40% on H)
3. **S-Wave**: Primarily horizontal motion (60-100% on N/E, 10-30% on Z)
4. **Coda Waves**: Scattered energy following S-arrival
5. **Realistic Envelopes**: Exponential decay with smooth onsets

## Example Workflows

### Complete Pipeline: Generate → Pack → Test
```bash
# 1. Configure generation
# Edit Syn_Config.json to set num_seismograms=100

# 2. Generate synthetic seismograms
python batch_generate_synthetic_3c.py
# Output: 100 events × 5 files each = 500 files

# 3. Pack into SeisBench format
python pack_to_seisbench.py
# Output: synthetic_dataset.hdf5, synthetic_metadata.csv

# 4. Test and visualize
python test_seisbench_dataset.py
# Output: example plots with phase picks

# 5. Verify
ls SYNTHETIC_*.npy | wc -l          # Should show 100
ls -lh synthetic_dataset.hdf5       # Check size
head synthetic_metadata.csv         # View picks
```

### Generate Training Dataset with Varied SNR
```bash
# Edit Syn_Config.json
cat > Syn_Config.json <<EOF
{
    "num_seismograms": 100,
    "sample_rate": 100,
    "duration": 30.0,
    "randomize_noise_level": true,
    "noise_level_range": [0.1, 0.4],
    ...
}
EOF

# Generate with randomized noise
python batch_generate_synthetic_3c.py

# Pack for SeisBench
python pack_to_seisbench.py

# Result: 100 traces with SNR ranging from 4-10 dB
```

### Load Data in Python
```python
import numpy as np
import json
from obspy import read

# Option 1: Load NumPy array (fastest)
data_3c = np.load('SYNTHETIC_001_synthetic_SYN_XX_3C.npy')
print(f"Shape: {data_3c.shape}")  # (3, 3000)

# Option 2: Load MiniSEED (full metadata preserved)
stream = read('SYNTHETIC_001_synthetic_SYN_XX_HHZ.mseed')
trace_z = stream[0]

# Option 3: Load from SeisBench format
import h5py
import pandas as pd

with h5py.File('synthetic_dataset.hdf5', 'r') as hdf:
    data = hdf['SYNTHETIC_001'][:]  # (3, 3000)

df = pd.read_csv('synthetic_metadata.csv')
picks = df[df['trace_name'] == 'SYNTHETIC_001'].iloc[0]
p_sample = picks['trace_p_arrival_sample']
s_sample = picks['trace_s_arrival_sample']
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
