# SeisBench with Synthetic Seismogram

Deep learning framework for seismic phase picking using PyTorch, SeisBench, and synthetic training data.

## Overview

This project provides a complete pipeline for generating synthetic seismic training data and building neural network-based systems for detecting and picking seismic phase arrivals (P and S waves) from three-component seismograms. The project includes tools for:

- **Synthetic data generation**: Create realistic 3-component synthetic seismograms for training
- **Data preprocessing**: Convert data to SeisBench format for standardized workflows
- **Model training**: Train deep learning models for phase picking with PhaseNet architecture
- **Model evaluation**: Generate predictions and analyze model performance
- **Visualization**: Comprehensive plotting tools for data and results analysis

## Project Structure

```
seisbench_with_synthetic_seismogram/
├── synthetic_input/     # Synthetic seismogram generation pipeline
│   ├── README.md        # Detailed documentation for synthetic data
│   ├── Syn_Config.json  # Configuration for data generation
│   ├── Training_Config.json                      # Configuration for model training
│   ├── P001_generate_synthetic_3c_seismogram.py  # Single seismogram generator
│   ├── P002_batch_generate_synthetic_3c.py       # Batch generation script
│   ├── P003_pack_to_seisbench.py                 # Convert to SeisBench format
│   ├── T001_test_seisbench_dataset.py            # Verify dataset integrity
│   ├── V001_demo_synthetic_dataset.ipynb         # Demo notebook for visualization
│   ├── V002_plot_noise_comparison.py             # Noise analysis visualization
│   ├── V003_demo_training.ipynb                  # Training demonstration notebook
│   ├── batch_summary.json                        # Generation log and metrics
│   └── REF_creating_a_dataset.ipynb              # Reference notebook
├── data/                # Generated datasets
│   ├── metadata.csv     # Trace metadata with phase picks
│   └── waveforms.hdf5   # 3-component waveform data
├── model/               # Model architectures and experiments
│   └── REF_dataset_basics.ipynb                  # SeisBench dataset basics
├── checkpoints/         # Saved model weights
│   ├── best_model.pth   # Best performing model checkpoint
│   ├── final_model.pth  # Final epoch model checkpoint
│   └── loss_history.json # Training loss history
├── output/              # Model predictions and visualizations
│   ├── Syn_Model_Pred_*_Results.txt              # Prediction results
│   ├── Syn_Model_Pred_*_Plot.png                 # Prediction visualizations
│   ├── loss_history_plot.png                     # Training loss curves
│   └── test_trace_*.png                          # Individual trace predictions
└── .github/             # GitHub configurations
    ├── workflows/       # CI/CD workflows
    ├── ISSUE_TEMPLATE/  # Issue templates
    └── agents/          # GitHub Copilot agent configurations
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

# Visualize the generated data
jupyter notebook V001_demo_synthetic_dataset.ipynb
```

### 2. Configure Parameters

**Data Generation** - Edit `synthetic_input/Syn_Config.json`:

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

**Model Training** - Edit `synthetic_input/Training_Config.json`:

```json
{
    "training": {
        "batch_size": 4,
        "learning_rate": 0.01,
        "epochs": 50,
        "patience": 5
    }
}
```

### 3. Train Model

```bash
# Open training demonstration notebook
jupyter notebook synthetic_input/V003_demo_training.ipynb

# Or use the configuration file for training
# Edit Training_Config.json to customize training parameters
```

### 4. Explore Dataset and Results

```bash
# Open dataset basics notebook
jupyter notebook model/REF_dataset_basics.ipynb

# Compare noise characteristics
python synthetic_input/V002_plot_noise_comparison.py

# View model predictions in output/ folder
ls output/
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

## Complete Workflow

This project provides an end-to-end pipeline:

1. **Generate** → Create synthetic seismograms with `P002_batch_generate_synthetic_3c.py`
2. **Pack** → Convert to SeisBench format with `P003_pack_to_seisbench.py`
3. **Validate** → Test dataset integrity with `T001_test_seisbench_dataset.py`
4. **Train** → Train PhaseNet model using `V003_demo_training.ipynb`
5. **Evaluate** → Generate predictions and analyze results
6. **Visualize** → Explore results in `output/` folder

## Features

### Synthetic Data Generation

- **Realistic waveforms**: Simulates P and S wave arrivals with proper polarization
- **3-component synthesis**: Vertical, North, and East components
- **Noise modeling**: Configurable realistic seismic noise with SNR tracking
- **Batch processing**: Generate large training datasets efficiently (72 traces generated as of Feb 17, 2026)
- **SeisBench integration**: Direct compatibility with SeisBench models
- **Generation tracking**: Automatic logging via `batch_summary.json`

### Data Processing

- **Automated packing**: Convert synthetic data to SeisBench format with configurable train/dev/test splits
- **Metadata extraction**: Automatic phase pick labeling with sample indices
- **Quality control**: Validation scripts to verify dataset integrity
- **Visualization tools**: Interactive notebooks and plotting scripts for data inspection
- **Noise analysis**: Tools to compare and analyze noise characteristics across traces

### Model Training

- **PhaseNet architecture**: State-of-the-art U-Net based phase picker
- **Configurable training**: Customize batch size, learning rate, epochs via Training_Config.json
- **Model checkpointing**: Automatic saving of best and final models
- **Loss tracking**: JSON-formatted training history with visualization
- **Mixed precision training**: Optional FP16 training for faster computation
- **Early stopping**: Prevent overfitting with configurable patience

### Model Evaluation

- **Batch prediction**: Generate predictions on test datasets
- **Performance metrics**: Residual analysis and pick accuracy statistics
- **Visualization**: Automated plotting of predictions vs ground truth
- **Result export**: Text-based results and parameter logs for each run

## Development Status

**As of February 17, 2026:**

- ✅ Synthetic data generation pipeline (72 seismograms generated)
- ✅ SeisBench format conversion with data splits
- ✅ Dataset testing and validation
- ✅ Visualization and analysis tools
- ✅ Generation tracking and metadata logging
- ✅ **Model training pipeline (PhaseNet implementation)**
- ✅ **Model checkpointing and loss tracking**
- ✅ **Prediction and evaluation framework**
- ✅ **Comprehensive visualization of results**
- 🚧 Hyperparameter optimization (planned)
- 🚧 Multi-model comparison (planned)

## Documentation

- [Synthetic Data Generation](synthetic_input/README.md) - Comprehensive guide for creating synthetic seismograms
- [V001_demo_synthetic_dataset.ipynb](synthetic_input/V001_demo_synthetic_dataset.ipynb) - Interactive demo of synthetic dataset
- [V003_demo_training.ipynb](synthetic_input/V003_demo_training.ipynb) - Model training demonstration
- [REF_dataset_basics.ipynb](model/REF_dataset_basics.ipynb) - SeisBench dataset fundamentals
- [Training_Config.json](synthetic_input/Training_Config.json) - Training configuration parameters
- [SeisBench Documentation](https://seisbench.readthedocs.io/) - Official SeisBench docs

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](.github/CONTRIBUTING.md) for details.

When adding new features:

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include comprehensive docstrings
4. Write unit tests for new functionality
5. Update relevant README files
6. Submit a pull request using our [PR template](.github/PULL_REQUEST_TEMPLATE.md)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{seisbench_synthetic,
  author = {Hongyu Xiao},
  title = {SeisBench with Synthetic Seismogram: Synthetic Data Generation for Seismic Phase Picking},
  year = {2026},
  url = {https://github.com/Hy-X/seisbench_with_synthetic_seismogram}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Getting Help

- **Issues**: Use our [issue templates](.github/ISSUE_TEMPLATE/) for bugs, features, or questions
- **Discussions**: For general questions and community support
- **Documentation**: Check the [docs](#documentation) section for guides and references

## Contact

- Author: Hongyu Xiao
- GitHub: [@Hy-X](https://github.com/Hy-X)
- Repository: [seisbench_with_synthetic_seismogram](https://github.com/Hy-X/seisbench_with_synthetic_seismogram)

## Acknowledgments

- Built with [SeisBench](https://github.com/seisbench/seisbench) framework
- Inspired by PhaseNet and EQTransformer architectures
- Uses [ObsPy](https://github.com/obspy/obspy) for seismological data processing
