# Model Training Scripts

This directory contains scripts for loading synthetic seismogram data and training phase picking models.

## Scripts

### 1. `load_seisbench_dataset.py` ⭐
Comprehensive script for loading and using the synthetic dataset in SeisBench format.

**Features:**
- Load HDF5 and CSV files without requiring full SeisBench installation
- Generate probabilistic pick labels (Gaussian, σ=50 samples)
- PyTorch Dataset and DataLoader integration
- Visualization of waveforms with pick labels
- Automatic file discovery from multiple locations

**Usage:**
```bash
python load_seisbench_dataset.py
```

**Output:**
- Dataset statistics summary
- 3 example visualization plots (low/medium/high SNR)
- PyTorch DataLoader demonstration

**Classes:**

#### `SyntheticSeismicDataset`
Wrapper for HDF5/CSV data access:
```python
from load_seisbench_dataset import SyntheticSeismicDataset

dataset = SyntheticSeismicDataset('synthetic_dataset.hdf5', 'synthetic_metadata.csv')
waveform, metadata = dataset[0]  # Get first trace
print(f"Shape: {waveform.shape}")  # (3, n_samples)
```

#### `SeismicPhaseDataset`
PyTorch Dataset with automatic label generation:
```python
from load_seisbench_dataset import SeismicPhaseDataset
from torch.utils.data import DataLoader

dataset = SeismicPhaseDataset('synthetic_dataset.hdf5', 'synthetic_metadata.csv', sigma=50.0)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for waveforms, labels in loader:
    # waveforms: (batch, 3, n_samples) - Z, N, E components
    # labels: (batch, 3, n_samples) - P, S, Noise probabilities
    model_output = model(waveforms)
    loss = criterion(model_output, labels)
```

**Functions:**

#### `generate_gaussian_pick_labels(n_samples, pick_sample, sigma=50.0)`
Generate Gaussian probabilistic label for a single phase pick:
```python
from load_seisbench_dataset import generate_gaussian_pick_labels

p_label = generate_gaussian_pick_labels(3000, pick_sample=1500, sigma=50)
# Returns normalized Gaussian: shape (3000,), max value 1.0 at pick_sample
```

#### `create_pick_labels(waveform, metadata, sigma=50.0)`
Generate 3-channel labels (P, S, Noise) from metadata:
```python
from load_seisbench_dataset import create_pick_labels

labels_p, labels_s, labels_n = create_pick_labels(waveform, metadata, sigma=50.0)
# Returns three arrays, each shape (n_samples,)
```

#### `plot_waveform_with_labels(waveform, labels, metadata, ...)`
Visualize waveform with probabilistic labels:
```python
from load_seisbench_dataset import plot_waveform_with_labels
import numpy as np

labels = np.stack([labels_p, labels_s, labels_n], axis=0)
plot_waveform_with_labels(waveform, labels, metadata, save_path='example.png')
```

### 2. `verify_data.py`
Quick verification script to check HDF5 and CSV data consistency.

**Usage:**
```bash
python verify_data.py
```

## Data Format

### Input Files
- **HDF5 File** (`synthetic_dataset.hdf5`): 
  - Contains 3C waveforms as datasets
  - Each dataset named by `trace_name` (e.g., "SYNTHETIC_001")
  - Shape: `(3, n_samples)` for Z, N, E components
  - Dtype: `float32`
  - Attributes: `sampling_rate`, `station`, `network`, `p_arrival_sample`, `s_arrival_sample`, `snr_db`

- **CSV File** (`synthetic_metadata.csv`):
  - Tabular metadata for all traces
  - Key columns:
    - `trace_name`: Unique identifier
    - `trace_p_arrival_sample`, `trace_s_arrival_sample`: Pick sample indices
    - `trace_p_arrival_time`, `trace_s_arrival_time`: Pick times in seconds
    - `trace_sampling_rate_hz`: Sampling rate (100 Hz)
    - `trace_npts`: Number of samples
    - `snr_db`: Signal-to-noise ratio

### Label Format

Following **SeisBench PhaseNet convention**:

```python
labels[0, :] = P-wave probability    # Gaussian centered at P-arrival
labels[1, :] = S-wave probability    # Gaussian centered at S-arrival  
labels[2, :] = Noise probability     # 1 - max(P, S)
```

**Gaussian Parameters:**
- **Sigma**: 50 samples (0.5 seconds at 100 Hz)
- **Normalization**: Peak value = 1.0 at pick sample
- **Shape**: Smooth Gaussian to account for pick uncertainty

This format is **compatible with**:
- PhaseNet
- EQTransformer
- Other SeisBench-based models

## Example Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from load_seisbench_dataset import SeismicPhaseDataset

# Load dataset
dataset = SeismicPhaseDataset(
    'synthetic_dataset.hdf5', 
    'synthetic_metadata.csv',
    sigma=50.0,
    normalize=True  # Standardize waveforms
)

# Create data loader
train_loader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True,
    num_workers=4
)

# Initialize model (example: simple U-Net or PhaseNet)
model = YourPhasePickingModel(in_channels=3, out_channels=3)
criterion = nn.BCELoss()  # Or custom loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for waveforms, labels in train_loader:
        waveforms = waveforms.to(device)  # (batch, 3, n_samples)
        labels = labels.to(device)        # (batch, 3, n_samples)
        
        # Forward pass
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

## Dataset Statistics

Based on the synthetic dataset in `../data/`:

```
Total traces: 12
Sampling rate: 100 Hz
Duration: 120 seconds (12,000 samples)
Components: 3 (Z, N, E)

Phase Picks:
  P-wave picks: 12/12 (100%)
  S-wave picks: 12/12 (100%)
  P-arrival range: 8.6 - 109.6 s
  S-arrival range: 23.1 - 111.5 s
  P-S separation: 1.8 - 49.0 s

SNR:
  Range: 4.3 - 9.8 dB
  Mean: 6.7 ± 1.9 dB
```

## Integration with SeisBench Models

The dataset can be used with actual SeisBench library:

```python
import seisbench.models as sbm
from load_seisbench_dataset import SeismicPhaseDataset

# Option 1: Use custom dataset with SeisBench models
dataset = SeismicPhaseDataset('synthetic_dataset.hdf5', 'synthetic_metadata.csv')

# Load pre-trained model
model = sbm.PhaseNet.from_pretrained("original")
model.eval()

# Make predictions
waveforms, labels = dataset[0]
waveforms_tensor = torch.from_numpy(waveforms).unsqueeze(0).float()
predictions = model(waveforms_tensor)

# Option 2: Use native SeisBench WaveformDataset
import seisbench.data as sbd
data = sbd.WaveformDataset('synthetic_dataset.hdf5', 'synthetic_metadata.csv')
```

## Requirements

```bash
pip install torch numpy pandas h5py matplotlib
# Optional: seisbench (for using pre-trained models)
```

## File Locations

The script automatically searches for data files in:
1. `../data/` (preferred location)
2. `../synthetic_input/`
3. Current directory

## Notes

- **Normalization**: Waveforms are standardized (zero mean, unit variance per component) when `normalize=True`
- **Sigma Selection**: σ=50 samples (0.5s at 100 Hz) is SeisBench standard for pick uncertainty
- **Memory Efficiency**: HDF5 format allows lazy loading - only requested traces are loaded into memory
- **Batch Processing**: PyTorch DataLoader handles efficient batching and shuffling
- **GPU Support**: All tensors are compatible with CUDA via `.to(device)`

## Next Steps

1. **Generate More Data**: Increase `num_seismograms` in `../synthetic_input/Syn_Config.json`
2. **Train Model**: Implement U-Net or PhaseNet architecture
3. **Evaluate**: Test on held-out synthetic data or real seismograms
4. **Transfer Learning**: Fine-tune pre-trained SeisBench models on synthetic data
