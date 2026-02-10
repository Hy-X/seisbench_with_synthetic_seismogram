# Xiao Net v2

Deep learning framework for seismic phase picking using PyTorch and SeisBench.

## Overview

Xiao Net v2 is a neural network-based system for detecting and picking seismic phase arrivals (P and S waves) from three-component seismograms. The project includes tools for:

- **Synthetic data generation**: Create realistic 3-component synthetic seismograms for training
- **Data preprocessing**: Convert data to SeisBench format for standardized workflows
- **Model training**: Train deep learning models for phase picking
- **Evaluation**: Benchmark model performance on test datasets

## Project Structure

```
xiao_net_ver_2/
├── synthetic_input/     # Synthetic seismogram generation pipeline
│   ├── README.md        # Detailed documentation for synthetic data
│   ├── Syn_Config.json  # Configuration file for generation parameters
│   ├── P001_generate_synthetic_3c_seismogram.py  # Single seismogram generator
│   ├── P002_batch_generate_synthetic_3c.py       # Batch generation script
│   ├── P003_pack_to_seisbench.py                 # Convert to SeisBench format
│   └── T001_test_seisbench_dataset.py            # Verify dataset integrity
├── data/                # Generated datasets
│   ├── metadata.csv     # Trace metadata with phase picks
│   └── waveforms.hdf5   # 3-component waveform data
├── model/               # Model architectures and experiments
├── training/            # Training scripts and configurations
└── output/              # Model outputs and results
```

## Quick Start

### 1. Generate Synthetic Data

```bash
cd synthetic_input

# Generate a batch of synthetic seismograms
python P002_batch_generate_synthetic_3c.py

# Pack into SeisBench format
python P003_pack_to_seisbench.py

# Verify the dataset
python T001_test_seisbench_dataset.py
```

### 2. Configure Generation Parameters

Edit `synthetic_input/Syn_Config.json` to customize:

```json
{
    "sample_rate": 100,           // Hz
    "duration": 120.0,            // seconds
    "p_frequency": 20.0,          // Hz
    "s_frequency": 10.0,          // Hz
    "noise_level": 0.25,          // amplitude
    "num_seismograms": 12         // number to generate
}
```

### 3. Train Model

```bash
cd training
# Training scripts will be added here
```

## Requirements

### Core Dependencies

```bash
pip install torch numpy scipy matplotlib
pip install obspy seisbench pandas h5py
```

### Optional Dependencies

```bash
pip install tqdm  # Progress bars for batch generation
pip install jupyter  # For notebooks
```

## Dataset Format

The project uses **SeisBench format** for data storage:

- **metadata.csv**: Contains trace information and phase picks
  - `trace_name`: Unique identifier for each trace
  - `trace_P_arrival_sample`: P-wave arrival sample index
  - `trace_S_arrival_sample`: S-wave arrival sample index
  - Additional metadata (station, sampling rate, etc.)

- **waveforms.hdf5**: HDF5 file with 3-component waveforms
  - Shape: `(3, n_samples)` for each trace
  - Channels: [Z, N, E] (Vertical, North, East)
  - Compatible with SeisBench models

## Features

### Synthetic Data Generation

- **Realistic waveforms**: Simulates P and S wave arrivals with proper polarization
- **3-component synthesis**: Vertical, North, and East components
- **Noise modeling**: Configurable realistic seismic noise
- **Batch processing**: Generate large training datasets efficiently
- **SeisBench integration**: Direct compatibility with SeisBench models

### Data Processing

- **Automated packing**: Convert synthetic data to standard format
- **Metadata extraction**: Automatic phase pick labeling
- **Quality control**: Validation scripts to verify dataset integrity

## Development Status

- ✅ Synthetic data generation pipeline
- ✅ SeisBench format conversion
- ✅ Dataset testing and validation
- 🚧 Model training pipeline (in progress)
- 🚧 Evaluation and benchmarking (in progress)

## Documentation

- [Synthetic Data Generation](synthetic_input/README.md) - Comprehensive guide for creating synthetic seismograms
- [SeisBench Documentation](https://seisbench.readthedocs.io/) - Official SeisBench docs

## Contributing

When adding new features:

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include comprehensive docstrings
4. Write unit tests for new functionality
5. Update relevant README files

## Citation

If you use this code in your research, please cite:

```bibtex
@software{xiao_net_v2,
  author = {Hongyu Xiao},
  title = {Xiao Net v2: Deep Learning for Seismic Phase Picking},
  year = {2026},
  url = {https://github.com/Hy-X/xiao_net_ver_2}
}
```

## License

[Add license information here]

## Contact

- Author: Hongyu Xiao
- GitHub: [Hy-X](https://github.com/Hy-X)

## Acknowledgments

- Built with [SeisBench](https://github.com/seisbench/seisbench) framework
- Inspired by PhaseNet and EQTransformer architectures
- Uses [ObsPy](https://github.com/obspy/obspy) for seismological data processing
