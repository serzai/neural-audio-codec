import torch
from torch import nn

from src.model.encoder import ResidualUnit


class DecoderBlock(nn.Module):
    """
    DecoderBlock for Decoder
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): stride for ConvTranspose1d
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + (stride % 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, C_in, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, C_out, T * stride).
        """
        return self.block(x)


class SoundStreamDecoder(nn.Module):
    """
    SoundStream Decoder
    """

    def __init__(self, base_channels=32, latent_dim=128, strides=[5, 5, 4, 2]):
        """
        Args:
            base_channels (int): number of channels in the last block.
            latent_dim (int): dim of embeddings.
            strides (list): list of strides for DecoderBlocks.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            latent_dim, base_channels * (2 ** len(strides)), kernel_size=7, padding=3
        )

        blocks = []
        curr_channels = base_channels * (2 ** len(strides))
        for stride in strides:
            next_channels = curr_channels // 2
            blocks.append(DecoderBlock(curr_channels, next_channels, stride))
            curr_channels = next_channels
        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Sequential(
            nn.ELU(), nn.Conv1d(curr_channels, 1, kernel_size=7, padding=3), nn.Tanh()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tensor of shape (B, latent_dim, T_latent).
        Returns:
            output (torch.Tensor): audio tensor of shape (B, 1, T_latent * 200).
        """
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        return x
