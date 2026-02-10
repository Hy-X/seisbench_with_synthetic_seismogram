# load and Visualize SeisBench Synthetic Dataset

This notebook demonstrates loading and using the synthetic dataset created with the SeisBench WaveformDataWriter API.

## Load the Dataset

```python
import seisbench.data as sbd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset using SeisBench
dataset = sbd.WaveformDataset('../data', sampling_rate=100)

print(f"Dataset size: {len(dataset)} traces")
print(f"\nMetadata columns: {list(dataset.metadata.columns)}")
print(f"\nData format: {dataset.data_format}")
```

## Inspect Metadata

```python
# Display first few rows
dataset.metadata.head()
```

## Visualize a Waveform with Picks

```python
# Get first trace
idx = 0
waveform = dataset.get_waveforms(idx)
metadata = dataset.metadata.iloc[idx]

# Extract pick information
p_sample = metadata['trace_p_arrival_sample']
s_sample = metadata['trace_s_arrival_sample']
sampling_rate = metadata['trace_sampling_rate_hz']
snr_db = metadata['trace_snr_db']

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
component_names = ['Z (Vertical)', 'N (North)', 'E (East)']
time_axis = np.arange(waveform.shape[1]) / sampling_rate

for i, (ax, name) in enumerate(zip(axes, component_names)):
    ax.plot(time_axis, waveform[i], 'k-', linewidth=0.5)
    ax.axvline(p_sample / sampling_rate, color='b', linestyle='--', linewidth=2, label='P-arrival')
    ax.axvline(s_sample / sampling_rate, color='r', linestyle='--', linewidth=2, label='S-arrival')
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()
        ax.set_title(f"Event: {metadata['source_id']} | SNR: {snr_db:.2f} dB")
    if i == 2:
        ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
```

## Access Training Split

```python
# Get training data
train_data = dataset.train()
print(f"Training set size: {len(train_data)} traces")

# SeisBench datasets support split() method
dev_data = dataset.dev()
test_data = dataset.test()
print(f"Development set size: {len(dev_data)} traces")
print(f"Test set size: {len(test_data)} traces")
```

## Statistics

```python
# Get P and S arrival statistics
p_arrivals = dataset.metadata['trace_p_arrival_sample']
s_arrivals = dataset.metadata['trace_s_arrival_sample']
snr = dataset.metadata['trace_snr_db']

print("Phase Arrival Statistics:")
print(f"  P-wave samples: {p_arrivals.min():.0f} - {p_arrivals.max():.0f}")
print(f"  S-wave samples: {s_arrivals.min():.0f} - {s_arrivals.max():.0f}")
print(f"\nSNR Statistics:")
print(f"  Mean: {snr.mean():.2f} dB")
print(f"  Std:  {snr.std():.2f} dB")
print(f"  Range: {snr.min():.2f} - {snr.max():.2f} dB")
```

## Use with SeisBench Models

```python
# Example: Load PhaseNet and make predictions
import seisbench.models as sbm

# Load pretrained model
model = sbm.PhaseNet.from_pretrained("instance")

# Generate predictions (requires preprocessing)
# See SeisBench documentation for full pipeline
```
