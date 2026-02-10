# Synthetic Seismogram Generation Pipeline

Complete workflow for generating realistic 3-component synthetic seismograms for training seismic phase picking models.

## Overview

This pipeline creates synthetic seismograms with realistic P and S wave arrivals, complete polarization characteristics, and configurable noise levels. The generated data is automatically converted to SeisBench format for seamless integration with existing seismic deep learning workflows.

## Workflow

```
1. Configure → 2. Generate → 3. Pack → 4. Validate
   (Config)     (Batch)       (SB)      (Test)
```

### Step 1: Configure Parameters

Edit **`Syn_Config.json`** to set generation parameters:

```json
{
    "sample_rate": 100,              // Sampling rate (Hz)
    "duration": 120.0,               // Seismogram duration (seconds)
    "p_frequency": 20.0,             // P-wave dominant frequency (Hz)
    "s_frequency": 10.0,             // S-wave dominant frequency (Hz)
    "noise_level": 0.25,             // Base noise amplitude
    "randomize_noise_level": true,   // Vary noise between traces
    "noise_level_range": [0.1, 0.4], // Noise variation range
    "coda_amplitude": 0.3,           // Coda decay amplitude
    "window_length": 10.0,           // Signal window (seconds)
    "pre_arrival_time": 2.0,         // Pre-arrival buffer (seconds)
    "num_seismograms": 12            // Number of traces to generate
}
```

### Step 2: Generate Synthetic Data

#### Single Seismogram (for testing)

```bash
python generate_synthetic_3c_seismogram.py
```

**Output**: Single 3-component seismogram with visualization

#### Batch Generation (for training datasets)

```bash
python batch_generate_synthetic_3c.py
```

**Output**:
```
synthetic_output/
├── seismogram_001.mseed         # MiniSEED format
├── seismogram_001_metadata.json # Phase picks and parameters
├── seismogram_001.npy           # NumPy array (3, n_samples)
├── seismogram_002.mseed
├── ...
└── batch_summary.json           # Summary of all generated traces
```

**Features**:
- Randomized P and S arrival times
- Variable noise levels (if configured)
- Progress tracking with tqdm
- Comprehensive metadata for each trace

### Step 3: Pack to SeisBench Format

```bash
python pack_to_seisbench.py
```

**Output**:
```
../data/
├── metadata.csv      # Trace metadata with phase picks
└── waveforms.hdf5    # 3-component waveform data
```

**Features**:
- Automatic discovery of all synthetic traces
- Conversion to SeisBench standard format
- Metadata consolidation
- HDF5 compression for efficient storage

### Step 4: Validate Dataset

```bash
python test_seisbench_dataset.py
```

**Output**:
- Statistical summary of dataset
- Sample waveform visualizations
- Verification of phase picks
- Data integrity checks

## Scripts Reference

### `generate_synthetic_3c_seismogram.py`

Generates a single 3-component synthetic seismogram with realistic wave propagation.

**Key Functions**:
- `generate_p_wave_3c()`: Synthesize P-wave with realistic polarization
- `generate_s_wave_3c()`: Synthesize S-wave with horizontal motion
- `add_noise_3c()`: Add frequency-dependent background noise
- `get_random_times()`: Randomize phase arrival times

**Wave Characteristics**:
- **P-wave**: Predominantly vertical (Z) component, with minor horizontal energy
- **S-wave**: Predominantly horizontal (N, E) components, minimal vertical
- **Noise**: Realistic frequency spectrum, added to all components
- **Coda**: Exponentially decaying tail following main arrivals

### `batch_generate_synthetic_3c.py`

Batch generation wrapper for creating large training datasets.

**Key Functions**:
- `generate_batch()`: Main batch generation loop
- `save_metadata()`: Write comprehensive JSON metadata
- `create_summary()`: Generate batch summary statistics

**Saved Metadata** (per trace):
```json
{
    "trace_id": "seismogram_001",
    "p_arrival_time": 15.32,
    "s_arrival_time": 35.68,
    "p_arrival_sample": 1532,
    "s_arrival_sample": 3568,
    "sample_rate": 100,
    "duration": 120.0,
    "noise_level": 0.23,
    "generated_at": "2026-02-10T10:30:45"
}
```

### `pack_to_seisbench.py`

Converts synthetic data to SeisBench HDF5/CSV format.

**Key Functions**:
- `discover_synthetic_data()`: Find all generated traces
- `load_synthetic_trace()`: Read MiniSEED and metadata
- `build_metadata_table()`: Create SeisBench-compatible CSV
- `write_to_seisbench()`: Use SeisBench WaveformDataWriter API

**SeisBench Metadata Columns**:
- `trace_name`: Unique identifier
- `trace_p_arrival_sample`: P-wave sample index
- `trace_s_arrival_sample`: S-wave sample index
- `trace_sampling_rate_hz`: Sampling rate
- `trace_npts`: Number of samples
- `station_code`: Station identifier (synthetic)
- `path_trace_Z/N/E`: HDF5 internal paths

### `test_seisbench_dataset.py`

Validates the packed SeisBench dataset.

**Key Functions**:
- `load_trace_from_hdf5()`: Load waveforms from HDF5
- `load_metadata()`: Read and parse CSV metadata
- `plot_trace_with_picks()`: Visualize waveforms with phase arrivals
- `display_dataset_stats()`: Print dataset statistics

**Validation Checks**:
- ✅ HDF5 file integrity
- ✅ Metadata completeness
- ✅ Phase pick consistency
- ✅ Waveform shape correctness
- ✅ Sampling rate uniformity

## Advanced Usage

### Custom Wave Parameters

Modify wave generation in `generate_synthetic_3c_seismogram.py`:

```python
# Adjust P-wave polarization
p_z, p_n, p_e = generate_p_wave_3c(
    p_time, p_freq, 
    p_amp=1.0,           # Amplitude
    polarization_angle=15  # Angle from vertical (degrees)
)

# Adjust S-wave characteristics
s_z, s_n, s_e = generate_s_wave_3c(
    s_time, s_freq,
    s_amp=1.2,           # Amplitude
    azimuth=45           # Propagation azimuth (degrees)
)
```

### Noise Customization

Control noise characteristics:

```python
# In generate_synthetic_3c_seismogram.py
def add_noise_3c(n_samples, noise_level, sample_rate):
    # Add custom frequency-dependent noise
    freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)
    
    # Modify noise spectrum
    noise_spectrum = 1.0 / (1.0 + (freqs/5.0)**2)  # Low-pass character
    
    # Apply to each component...
```

### Batch with Variable Magnitudes

Simulate different event magnitudes:

```python
# In batch_generate_synthetic_3c.py
def generate_with_variable_magnitude():
    for i in range(num_seismograms):
        magnitude_scale = np.random.uniform(0.5, 2.0)
        # Scale wave amplitudes...
```

## Output File Formats

### MiniSEED (.mseed)

Standard seismological format with ObsPy Stream containing 3 traces:
- Channel codes: BHZ, BHN, BHE (vertical, north, east)
- Network: SY (synthetic)
- Station: STN01
- Compatible with ObsPy, SAC, and other seismological tools

### NumPy Array (.npy)

Direct array format for ML pipelines:
- Shape: `(3, n_samples)`
- Dtype: `float32`
- Fast loading with `np.load()`

### Metadata JSON

Comprehensive event information:
- Phase arrival times and samples
- Generation parameters
- Quality metrics
- Timestamps

### SeisBench Format

**metadata.csv**: Pandas-compatible CSV
```csv
trace_name,trace_p_arrival_sample,trace_s_arrival_sample,trace_sampling_rate_hz,...
synthetic_001,1532,3568,100.0,...
synthetic_002,2341,4982,100.0,...
```

**waveforms.hdf5**: Hierarchical storage
```
waveforms.hdf5
├── synthetic_001 (dataset: shape=(3, 12000))
│   └── attrs: {sampling_rate: 100.0, ...}
├── synthetic_002 (dataset: shape=(3, 12000))
└── ...
```

## Performance

### Generation Speed

- Single seismogram: ~0.1 seconds
- Batch of 1000: ~2 minutes (without tqdm overhead)
- Parallel generation possible for larger datasets

### Storage Requirements

Per 120-second trace at 100 Hz:
- MiniSEED: ~40 KB
- NumPy: ~144 KB
- HDF5 (compressed): ~30 KB per trace in dataset

For 10,000 traces: ~300 MB total

## Troubleshooting

### Issue: No synthetic output files found

```
Solution: Run batch_generate_synthetic_3c.py first before packing
```

### Issue: HDF5 file already exists

```python
# In pack_to_seisbench.py, set overwrite flag:
writer = sbd.WaveformDataWriter(metadata, overwrite=True)
```

### Issue: Arrival time validation errors

```
Check: Ensure P arrives before S in configuration
P-S separation should be > 1 second
```

### Issue: Memory errors with large batches

```
Solution: Generate in smaller batches, then merge datasets
Or increase num_seismograms gradually
```

## Integration with Training

```python
import seisbench.data as sbd

# Load synthetic dataset
dataset = sbd.WaveformDataset("../data")

# Get training splits
train_data = dataset.train()
dev_data = dataset.dev()

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(train_data, batch_size=32, shuffle=True)

for batch in loader:
    waveforms = batch["X"]  # (batch, 3, n_samples)
    p_picks = batch["trace_p_arrival_sample"]
    s_picks = batch["trace_s_arrival_sample"]
    # Train model...
```

## Best Practices

1. **Start Small**: Test with 10-20 seismograms before large batches
2. **Validate Often**: Run test_seisbench_dataset.py after each packing
3. **Version Control**: Keep Syn_Config.json in version control
4. **Document Changes**: Update batch_summary.json with generation notes
5. **Backup Data**: Store precious synthetic datasets before regeneration

## Future Enhancements

- [ ] Add more realistic attenuation models
- [ ] Support for different source mechanisms
- [ ] Multi-station geometry simulation
- [ ] Integration with earthquake catalogs
- [ ] Automated quality scoring
- [ ] GPU-accelerated batch generation

## References

- **SeisBench**: https://seisbench.readthedocs.io/
- **ObsPy**: https://docs.obspy.org/
- **PhaseNet Paper**: Zhu & Beroza (2019), Seismological Research Letters
- **Synthetic Seismograms**: Aki & Richards (2002), Quantitative Seismology

## Support

For issues or questions about the synthetic data pipeline:
- Check this README first
- Review example notebooks in the directory
- Consult SeisBench documentation for format questions
- Open an issue on the project repository
