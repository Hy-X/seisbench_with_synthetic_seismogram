# xiao_net_ver_2
Seismic phase picking using U-Net with Gaussian probabilistic labels

## Overview

This repository implements a complete workflow for seismic phase picking using a 1D U-Net architecture with SeisBench-compatible Gaussian probabilistic labeling.

## Features

- **miniSEED Data Loader**: Read and preprocess seismic data (20s sequences at 100 Hz)
- **U-Net Model**: Flexible 1D U-Net architecture with multiple variants
- **Gaussian Labels**: SeisBench-compatible probabilistic labels (sigma=50 samples)
- **Evaluation**: True positive detection with 1-second tolerance
- **Visualization**: Plot 3-channel data, labels, and predictions
- **Statistics**: Comprehensive performance metrics (precision, recall, F1)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see the complete workflow:

```bash
python example.py
```

This will:
1. Generate synthetic 3-channel seismic data
2. Create and initialize a U-Net model
3. Generate Gaussian probabilistic labels (sigma=50)
4. Make predictions
5. Evaluate with 1-second tolerance
6. Generate visualizations and statistics

Output will be saved in the `output/` directory.

## Components

### 1. Data Loader (`data_loader.py`)

```python
from data_loader import MiniSEEDLoader

loader = MiniSEEDLoader(target_sample_rate=100, window_length=20.0)
data, channel_names = loader.read_mseed('path/to/file.mseed')
```

### 2. U-Net Model (`unet_model.py`)

```python
from unet_model import create_unet

# Create model variants
model = create_unet(variant='standard')  # or 'small', 'large', 'deep'
```

### 3. Label Generator (`label_generator.py`)

```python
from label_generator import GaussianLabelGenerator

label_gen = GaussianLabelGenerator(sigma=50.0, sample_rate=100.0)
label = label_gen(n_samples=2000, picks=[5.2, 8.7])  # picks in seconds
```

### 4. Evaluation (`evaluation.py`)

```python
from evaluation import evaluate_predictions

metrics = evaluate_predictions(
    predictions, 
    labels,
    threshold=0.5,
    tolerance_samples=100  # 1 second at 100 Hz
)
```

### 5. Visualization (`visualization.py`)

```python
from visualization import plot_waveform_with_predictions, plot_statistics

plot_waveform_with_predictions(data, label, prediction, sample_rate=100.0)
plot_statistics(metrics)
```

## Configuration

- **Sample Rate**: 100 Hz (default)
- **Window Length**: 20 seconds (2000 samples)
- **Label Sigma**: 50 samples
- **True Positive Tolerance**: 1 second (100 samples)
- **Detection Threshold**: 0.5

## Model Variants

| Variant | Base Filters | Depth | Kernel Size | Parameters |
|---------|--------------|-------|-------------|------------|
| small   | 8            | 3     | 3           | ~25K       |
| standard| 16           | 4     | 3           | ~200K      |
| large   | 32           | 4     | 5           | ~800K      |
| deep    | 16           | 5     | 3           | ~400K      |

## Output Examples

The example script generates:
- Individual waveform plots with predictions
- Statistics visualization
- Performance metrics (precision, recall, F1 score)

## License

See LICENSE file for details.
