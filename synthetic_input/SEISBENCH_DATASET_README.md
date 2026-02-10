# SeisBench Dataset Creation - Summary

## Overview

The [pack_to_seisbench.py](pack_to_seisbench.py) script has been updated to use the **official SeisBench `WaveformDataWriter` API**, following the reference example from [SeisBench documentation](https://seisbench.readthedocs.io/).

## Key Changes

### Before (Manual Approach)
- **Manual HDF5 creation** using h5py directly
- **Manual CSV generation** with 35 columns
- **3 rows per event** (one for each component: HHZ, HHN, HHE)  
- Custom column names not fully aligned with SeisBench conventions
- Manual data_format group creation

### After (SeisBench API)
- **Uses `sbd.WaveformDataWriter`** - the official SeisBench method
- **Automatic metadata management** with 22 standard columns
- **1 row per trace** (3-component data stored as single trace)
- **Proper column naming** with prefixes: `station_`, `trace_`, `source_`
- **Automatic HDF5 optimization** (bucketing, compression)
- **Direct SeisBench compatibility** - verified by loading with `sbd.WaveformDataset`

## Generated Files

```
data/
├── metadata.csv      # Trace metadata with phase picks (2.5 KB, 12 rows)
└── waveforms.hdf5    # 3C waveform data (3.31 MB, optimized)
```

## Metadata Structure

The new format follows SeisBench conventions:

```python
# Station information
station_network_code, station_code, station_location_code

# Trace properties
trace_channel, trace_sampling_rate_hz, trace_npts, trace_start_time
trace_p_arrival_sample, trace_p_status, trace_p_weight
trace_s_arrival_sample, trace_s_status, trace_s_weight  
trace_snr_db, trace_name

# Source information
source_id, source_origin_time, source_type
source_magnitude, source_magnitude_type, source_magnitude_author

# Dataset split
split
```

## Usage Example

```python
import seisbench.data as sbd
import matplotlib.pyplot as plt

# Load dataset
dataset = sbd.WaveformDataset('../data', sampling_rate=100)

# Get a waveform
waveform = dataset.get_waveforms(0)  # Shape: (3, 12000) - 3 components
metadata = dataset.metadata.iloc[0]

# Extract picks
p_sample = metadata['trace_p_arrival_sample']
s_sample = metadata['trace_s_arrival_sample']
snr_db = metadata['trace_snr_db']

# Plot
plt.plot(waveform[0])  # Z component
plt.axvline(p_sample, color='b', label='P')
plt.axvline(s_sample, color='r', label='S')
plt.legend()
plt.show()
```

## Verification

The dataset is verified by:
1. ✅ Loading successfully with `sbd.WaveformDataset()`
2. ✅ Correct waveform shape (3, n_samples)
3. ✅ All required metadata columns present
4. ✅ Phase picks accessible and valid

## Benefits

1. **Standards Compliant**: Follows official SeisBench data format
2. **Simplified**: Less code, automatic handling of HDF5/CSV details
3. **Optimized**: Built-in compression and bucketing for large datasets
4. **Compatible**: Direct integration with SeisBench models (PhaseNet, EQTransformer, etc.)
5. **Maintainable**: Uses stable API that will be supported long-term

## Dataset Statistics

- **Total traces**: 12
- **Sampling rate**: 100 Hz
- **Trace length**: 12,000 samples (120 seconds)
- **P-arrivals**: 12/12 (range: 860 - 10,959 samples)
- **S-arrivals**: 12/12 (range: 2,306 - 11,146 samples)
- **SNR**: 6.67 ± 1.86 dB (range: 4.29 - 9.78 dB)
- **Split**: All in training set

## References

- [SeisBench Documentation](https://seisbench.readthedocs.io/)
- [Creating a Dataset Tutorial](https://github.com/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb)
- [Data Format Specification](https://seisbench.readthedocs.io/en/stable/pages/data_format.html)
