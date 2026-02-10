# Usage Guide

This guide provides detailed instructions for using the seismic phase picking system.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Example

```bash
python example.py
```

This will generate synthetic data and demonstrate the complete workflow, saving visualizations to the `output/` directory.

## Working with Real Data

### Reading miniSEED Files

```python
from data_loader import MiniSEEDLoader

# Initialize loader
loader = MiniSEEDLoader(target_sample_rate=100, window_length=20.0)

# Read a miniSEED file
data, channel_names = loader.read_mseed('path/to/your/file.mseed', normalize=True)
# data shape: (n_channels, n_samples)
# For 20s at 100 Hz: (3, 2000)
```

### Creating Labels

```python
from label_generator import GaussianLabelGenerator

# Initialize label generator with sigma=50 samples
label_gen = GaussianLabelGenerator(sigma=50.0, sample_rate=100.0)

# Generate labels from pick times (in seconds)
pick_times = [5.2, 8.7, 12.3]  # Example: picks at these times
label = label_gen(n_samples=2000, picks=pick_times)
```

### Model Selection

Choose a model variant based on your needs:

| Variant  | Parameters | Use Case |
|----------|-----------|----------|
| small    | ~43K      | Fast inference, limited data |
| standard | ~680K     | Balanced performance |
| large    | ~4.3M     | Best accuracy, more data needed |
| deep     | ~2.7M     | Complex patterns |

```python
from unet_model import create_unet

# Create a model
model = create_unet(variant='standard', in_channels=3, out_channels=1)
```

## Training Your Model

### Prepare Training Data

```python
import numpy as np
from torch.utils.data import DataLoader
from train import SeismicDataset, train_model

# Load your data
data_list = []  # List of waveform arrays (n_channels, n_samples)
label_list = []  # List of label arrays (n_samples,)

# ... load your miniSEED files and create labels ...

# Split into train/validation
train_size = int(0.8 * len(data_list))
train_data = data_list[:train_size]
train_labels = label_list[:train_size]
val_data = data_list[train_size:]
val_labels = label_list[train_size:]

# Create datasets
train_dataset = SeismicDataset(train_data, train_labels)
val_dataset = SeismicDataset(val_data, val_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Train the Model

```python
from unet_model import create_unet
from train import train_model

# Create model
model = create_unet(variant='standard')

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=0.001,
    device='cuda',  # or 'cpu'
    save_dir='models'
)
```

### Load a Trained Model

```python
from train import load_model

# Load best model
model, epoch = load_model(model, 'models/best_model.pth', device='cuda')
model.eval()
```

## Making Predictions

```python
import torch

# Prepare data
data_tensor = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
data_tensor = data_tensor.to(device)

# Make prediction
model.eval()
with torch.no_grad():
    prediction = model(data_tensor)
    prediction = prediction.squeeze().cpu().numpy()
```

## Evaluation

### Evaluate with 1-Second Tolerance

```python
from evaluation import evaluate_predictions

# Evaluate predictions
metrics = evaluate_predictions(
    predictions=prediction_array,
    labels=label_array,
    threshold=0.5,
    tolerance_samples=100,  # 1 second at 100 Hz
    sample_rate=100.0
)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Extract Pick Times

```python
from evaluation import extract_picks_from_prediction

# Extract predicted pick locations
pred_picks = extract_picks_from_prediction(
    prediction=prediction,
    threshold=0.5,
    min_distance=50  # Minimum 50 samples between picks
)

# Convert to time (seconds)
pick_times = pred_picks / 100.0  # Divide by sample rate
print(f"Predicted picks at: {pick_times} seconds")
```

## Visualization

### Plot Single Example

```python
from visualization import plot_waveform_with_predictions

plot_waveform_with_predictions(
    data=data,
    label=label,
    prediction=prediction,
    channel_names=['Z', 'N', 'E'],
    sample_rate=100.0,
    threshold=0.5,
    save_path='example_plot.png'
)
```

### Plot Multiple Examples

```python
from visualization import plot_multiple_examples

plot_multiple_examples(
    data_list=data_list[:3],
    label_list=label_list[:3],
    prediction_list=prediction_list[:3],
    n_examples=3,
    sample_rate=100.0,
    save_prefix='output/example'
)
```

### Plot Statistics

```python
from visualization import plot_statistics, print_statistics

# Print to console
print_statistics(metrics)

# Save plot
plot_statistics(metrics, save_path='statistics.png')
```

## Configuration Parameters

### Key Parameters

- **Sample Rate**: 100 Hz (configurable in MiniSEEDLoader)
- **Window Length**: 20 seconds (configurable in MiniSEEDLoader)
- **Label Sigma**: 50 samples (standard deviation of Gaussian label)
- **Tolerance**: 100 samples (1 second for true positive detection)
- **Threshold**: 0.5 (detection threshold for predictions)

### Model Architectures

All variants use the same U-Net architecture but with different sizes:

```python
# Customize model parameters
model = create_unet(
    variant='standard',
    in_channels=3,      # Number of input channels
    out_channels=1,     # Number of output channels
    base_filters=16,    # Number of filters in first layer
    depth=4,            # Number of downsampling/upsampling layers
    kernel_size=3       # Convolution kernel size
)
```

## Tips and Best Practices

1. **Data Normalization**: Always normalize your waveforms to zero mean and unit standard deviation
2. **Label Sigma**: sigma=50 samples (0.5s at 100 Hz) provides good localization
3. **Tolerance**: 1 second tolerance is standard for automatic phase picking
4. **Batch Size**: Adjust based on GPU memory (typical: 16-64)
5. **Learning Rate**: Start with 0.001, reduce if loss plateaus
6. **Model Selection**: Start with 'standard' variant, adjust based on performance

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller model variant
2. **Poor Performance**: Check data normalization, increase model size, train longer
3. **Too Many False Positives**: Increase detection threshold
4. **Missing Picks**: Lower detection threshold, adjust min_distance parameter

### Data Requirements

- Minimum recommended: 1000+ training examples
- 3-component seismograms (Z, N, E)
- Consistent sampling rate (100 Hz recommended)
- Quality manual picks for training labels

## Examples

See `example.py` for a complete working example with synthetic data.

## Support

For issues or questions, please refer to the repository README or open an issue on GitHub.
