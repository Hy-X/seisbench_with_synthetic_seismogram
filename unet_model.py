"""
U-Net model for seismic phase picking.
Supports various architectural variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if needed
        diff = x2.size()[2] - x1.size()[2]
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x2 = F.pad(x2, [-diff // 2, -diff + (-diff // 2)])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net for seismic phase detection.
    
    Args:
        in_channels: Number of input channels (typically 3 for 3-component seismograms)
        out_channels: Number of output channels (typically 1 for phase detection)
        base_filters: Number of filters in the first layer
        depth: Depth of the U-Net (number of downsampling/upsampling layers)
        kernel_size: Kernel size for convolutions
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_filters=16, depth=4, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_filters, kernel_size)
        
        # Encoder (downsampling path)
        self.down_layers = nn.ModuleList()
        for i in range(depth):
            in_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch, kernel_size))
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        for i in range(depth):
            in_ch = base_filters * (2 ** (depth - i))
            out_ch = base_filters * (2 ** (depth - i - 1))
            self.up_layers.append(Up(in_ch, out_ch, kernel_size))
        
        # Output convolution
        self.outc = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        
        down_outputs = [x1]
        x = x1
        for down in self.down_layers:
            x = down(x)
            down_outputs.append(x)
        
        # Decoder
        x = down_outputs[-1]
        for i, up in enumerate(self.up_layers):
            x = up(x, down_outputs[-(i + 2)])
        
        # Output
        logits = self.outc(x)
        
        return torch.sigmoid(logits)


def create_unet(variant='standard', **kwargs):
    """
    Factory function to create U-Net variants.
    
    Args:
        variant: Model variant ('standard', 'small', 'large', 'deep')
        **kwargs: Additional arguments passed to UNet1D
        
    Returns:
        UNet1D model
    """
    variants = {
        'standard': {'base_filters': 16, 'depth': 4, 'kernel_size': 3},
        'small': {'base_filters': 8, 'depth': 3, 'kernel_size': 3},
        'large': {'base_filters': 32, 'depth': 4, 'kernel_size': 5},
        'deep': {'base_filters': 16, 'depth': 5, 'kernel_size': 3},
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    params = variants[variant]
    params.update(kwargs)
    
    return UNet1D(**params)
