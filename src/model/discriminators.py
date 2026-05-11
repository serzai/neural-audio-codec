import torch
import torch.nn as nn
from torch.nn import functional as F


class STFTResidualUnit(nn.Module):
    """
    ResidualUnit for STFT Disciminator
    """

    def __init__(self, in_channels, out_channels, strides):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            strides (tuple): strides (s_t, s_f).
        """
        super().__init__()

        kernel_size = (strides[0] + 2, strides[1] + 2)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            ),
        )

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, C, F, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, C_out, F/s_f, T/s_t).
        """
        return self.skip(x) + self.block(x)


class STFTDiscriminator(nn.Module):
    """
    STFT-based discriminator (Section 3.3, Figure 4)
    """

    def __init__(self, base_channels=32, n_fft=1024, hop_length=256):
        """
        Args:
            base_channels (int): base number of channels.
            n_fft (int): FFT window size.
            hop_lenght (int): hop lenght for STFT.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=7, padding=3), nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList(
            [
                STFTResidualUnit(base_channels, base_channels, (1, 2)),
                STFTResidualUnit(base_channels, 2 * base_channels, (2, 2)),
                STFTResidualUnit(2 * base_channels, 4 * base_channels, (1, 2)),
                STFTResidualUnit(4 * base_channels, 4 * base_channels, (2, 2)),
                STFTResidualUnit(4 * base_channels, 8 * base_channels, (1, 2)),
                STFTResidualUnit(8 * base_channels, 8 * base_channels, (2, 2)),
            ]
        )

        self.conv2 = nn.Conv2d(8 * base_channels, 1, kernel_size=(1, 8))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): audio tensor of shape (B, 1, T).
        Returns:
            logits (list): final output scores.
            feature_maps (list): activations for feature matching loss.
        """
        stft = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=False,
        )

        stft = stft.permute(0, 3, 1, 2)

        feature_maps = []

        out = self.conv1(stft)
        feature_maps.append(out)

        for block in self.blocks:
            out = block(out)
            feature_maps.append(out)

        logits = self.conv2(out).squeeze(1).squeeze(1)

        return [logits], feature_maps


class ScaleDiscriminator(nn.Module):
    """
    Single-scale discriminator (from MelGAN paper)
    """

    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=15, padding=7),
                nn.Conv1d(16, 64, kernel_size=41, padding=20, stride=4, groups=4),
                nn.Conv1d(64, 256, kernel_size=41, padding=20, stride=4, groups=16),
                nn.Conv1d(256, 1024, kernel_size=41, padding=20, stride=4, groups=64),
                nn.Conv1d(1024, 1024, kernel_size=41, padding=20, stride=4, groups=256),
                nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            ]
        )

        self.final_conv = nn.Conv1d(1024, 1, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, 1, T).
        Returns:
            logits (torch.Tensor): final scores.
            feature_maps (list): activations for feature matching loss.
        """
        feature_maps = []

        for conv in self.convs:
            x = self.relu(conv(x))
            feature_maps.append(x)

        logits = self.final_conv(x)

        return logits, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """
    Waveform-based discriminator
    """

    def __init__(self, scales=3):
        """
        Args:
            scales (int): number of scales to process the audio.
        """
        super().__init__()

        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator() for i in range(scales)]
        )

        self.pools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2) for i in range(scales - 1)]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): audio tensor of shape (B, 1, T).
        Returns:
            logits_list (list): final output scores from all scales.
            features_list (list): aggregated activations.
        """
        logits_list = []
        features_list = []

        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x = self.pools[i - 1](x)

            logits, features = disc(x)
            logits_list.append(logits)
            features_list.extend(features)

        return logits_list, features_list
