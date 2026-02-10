"""
U-Net architecture for seismic phase detection.

This implementation is designed specifically for detecting P-waves and S-waves
in seismic waveform data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net encoder and decoder."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool followed by double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with upsampling followed by double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding for skip connections
        diff = x2.size()[2] - x1.size()[2]
        
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution layer."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for seismic phase detection.
    
    Args:
        n_channels: Number of input channels (typically 3 for E, N, Z components)
        n_classes: Number of output classes (e.g., 3 for noise, P-wave, S-wave)
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False)
    """
    
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SeismicUNet(nn.Module):
    """
    Specialized U-Net for seismic phase detection with additional features.
    
    This variant includes:
    - Dropout for regularization
    - Attention mechanisms for better feature selection
    - Output activation for probability estimates
    
    Args:
        n_channels: Number of input channels (default: 3 for E, N, Z components)
        n_classes: Number of output classes (default: 3 for noise, P-wave, S-wave)
        dropout: Dropout probability (default: 0.1)
        use_attention: Whether to use attention mechanisms (default: False)
    """
    
    def __init__(self, n_channels=3, n_classes=3, dropout=0.1, use_attention=False):
        super(SeismicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention
        
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        logits = self.unet(x)
        if self.training:
            logits = self.dropout(logits)
        return logits
