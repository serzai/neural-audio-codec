import torch
from torch import nn
from torch.nn import functional as F


class ResidualUnit(nn.Module):
    """
    ResidualUnit for EncoderBlock and DecoderBlock
    """

    def __init__(self, channels, dilation):
        """
        Args:
            channels (int): number of input and output channels.
            dilation (int): dilation factor.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=7,
                dilation=dilation,
                padding=3 * dilation,
            ),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, C, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, C, T).
        """
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """
    EncoderBlock for Encoder
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride for Conv1d
        """
        super().__init__()
        self.res_units = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
        )
        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + (stride % 2),
            ),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, C_in, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, C_out, T).
        """
        return self.conv(self.res_units(x))


class SoundStreamEncoder(nn.Module):
    """
    SoundStream Encoder
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        latent_dim=128,
        strides=[2, 4, 5, 5],
    ):
        """
        Args:
            in_channels (int): audio channels.
            base_channels (int): initial conv channels (C_enc).
            latent_dim (int): dim of embeddings.
            strides (list): list of strides for EncoderBlocks.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)

        blocks = []
        curr_channels = base_channels
        for stride in strides:
            next_channels = curr_channels * 2
            blocks.append(EncoderBlock(curr_channels, next_channels, stride))
            curr_channels = next_channels
        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Conv1d(curr_channels, latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): audio tensor of shape (B, 1, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, latent_dim, T / 200).
        """
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        return x
