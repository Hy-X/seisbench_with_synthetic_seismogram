# Quick Start Guide

## Installation

```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Import the Model

```python
from src.models import SeismicUNet
import torch

model = SeismicUNet(n_channels=3, n_classes=3)
```

### 2. Prepare Your Data

```python
from src.data import SeismicDataset, get_dataloader
import numpy as np

# Load your data (shape: n_samples x n_channels x n_timesteps)
data = np.load('your_data.npy')
labels = np.load('your_labels.npy')  # Optional for inference

dataset = SeismicDataset(data, labels)
dataloader = get_dataloader(dataset, batch_size=16)
```

### 3. Train the Model

```bash
# Using command line
python train.py --config configs/train_config.yaml

# Or programmatically
from train import train
config = {...}  # Your config dict
train(config)
```

### 4. Run Inference

```bash
# Using command line
python inference.py --model outputs/models/best_model.pth \
                    --input your_data.npy \
                    --output outputs/predictions

# Or programmatically
from inference import PhasePicker

picker = PhasePicker('path/to/model.pth')
predictions = picker.predict(waveform)
arrivals = picker.extract_phase_arrivals(predictions)
```

## Data Format

### Input Data
- **Shape**: `(n_samples, n_channels, n_timesteps)`
- **Channels**: 3 (East, North, Vertical components)
- **Type**: float32 numpy array

### Labels (Training)
- **Shape**: `(n_samples, n_classes, n_timesteps)`
- **Classes**:
  - 0: Noise/Background
  - 1: P-wave
  - 2: S-wave
- **Type**: float32 numpy array (one-hot encoded)

## Examples

### Example 1: Simple Training
```python
from src.models import SeismicUNet
from src.data import create_synthetic_data, SeismicDataset, get_dataloader
from src.utils import CombinedLoss
import torch.optim as optim

# Create data
data, labels = create_synthetic_data(n_samples=1000)

# Setup
model = SeismicUNet()
dataset = SeismicDataset(data, labels)
loader = get_dataloader(dataset, batch_size=16)
criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train one epoch
for batch_data, batch_labels in loader:
    optimizer.zero_grad()
    output = model(batch_data)
    loss = criterion(output, batch_labels)
    loss.backward()
    optimizer.step()
```

### Example 2: Phase Detection
```python
from inference import PhasePicker
import numpy as np

# Load model
picker = PhasePicker('best_model.pth')

# Your waveform (3 x n_timesteps)
waveform = np.load('waveform.npy')

# Detect phases
probabilities = picker.predict(waveform)
arrivals = picker.extract_phase_arrivals(probabilities, threshold=0.5)

print(f"P-wave arrivals at: {arrivals['P']} seconds")
print(f"S-wave arrivals at: {arrivals['S']} seconds")
```

### Example 3: Catalog Creation
```python
catalog = []
for i, waveform in enumerate(waveforms):
    entry = picker.create_catalog_entry(
        waveform=waveform,
        station_name=f'STATION_{i:03d}',
        start_time='2024-01-01T00:00:00.000Z'
    )
    catalog.append(entry)

# Save catalog
import json
with open('catalog.json', 'w') as f:
    json.dump(catalog, f, indent=2)
```

## Configuration

Edit `configs/train_config.yaml` to customize:

- `n_channels`: Number of input channels (default: 3)
- `n_classes`: Number of output classes (default: 3)
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Learning rate (default: 0.001)
- `num_epochs`: Number of training epochs (default: 50)
- Loss weights, dropout, etc.

## Tips

1. **Data Normalization**: Always normalize your waveforms before inference
2. **Threshold Tuning**: Adjust the detection threshold based on your data quality
3. **GPU Usage**: Set device to 'cuda' for faster training/inference
4. **Batch Size**: Reduce if you encounter memory issues
5. **Learning Rate**: Use learning rate scheduling for better convergence

## Troubleshooting

**Q: Model not detecting phases?**
- Check if your data is normalized
- Try lowering the detection threshold
- Ensure input shape is correct (n_channels x n_timesteps)

**Q: Training loss not decreasing?**
- Reduce learning rate
- Check data quality and labels
- Try adjusting loss weights

**Q: Out of memory error?**
- Reduce batch size
- Use shorter waveforms
- Enable gradient accumulation

## More Examples

Run the complete example script:
```bash
python examples/example_usage.py
```

This demonstrates:
- Model initialization
- Data generation
- Training step
- Inference
- Catalog curation
- Visualization
