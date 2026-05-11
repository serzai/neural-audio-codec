import torch
from torch import nn

from src.model.decoder import SoundStreamDecoder
from src.model.encoder import SoundStreamEncoder
from src.model.rvq import ResidualVectorQuantizer


class SoundStreamModel(nn.Module):
    """
    SoundStream neural audio codec model
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        latent_dim=128,
        strides=[2, 4, 5, 5],
        num_quantizers=8,
        codebook_size=1024,
    ):
        """
        Args:
            in_channels (int): input audio channels.
            base_channels (int): base number of channels for convolutions.
            latent_dim (int): dim of embeddings.
            strides (list): list of strides for Encoder and Decoder.
            num_quantizers (int): number of quantization stages.
            codebook_size (int): number of vectors in the codebook.
        """
        super().__init__()

        self.encoder = SoundStreamEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
        )

        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
        )

        self.decoder = SoundStreamDecoder(
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides[::-1],
        )

    def forward(self, audio, **batch):
        """
        Args:
            audio (torch.Tensor): audio waveform of shape (B, 1, T).
        Returns:
            output (dict): reconstructed audio and commitment loss.
        """
        encoded_features = self.encoder(audio)
        quantized_features, loss = self.rvq(encoded_features)
        reconstructed_audio = self.decoder(quantized_features)

        return {
            "audio": reconstructed_audio,
            "loss": loss,
        }

    def encode(self, audio):
        """
        Method to extract unquantized latent features.

        Args:
            audio (torch.Tensor): audio waveform of shape (B, 1, T).
        Returns:
            output (torch.Tensor): tensor of shape (B, latent_dim, T // 200).
        """

        return self.encoder(audio)

    def decode(self, quantized_features):
        """
        Method to reconstruct audio from quantized features.

        Args:
            audio (torch.Tensor): tensor of shape (B, latent_dim, T_latent).
        Returns:
            output (torch.Tensor): audio tensor of shape (B, 1, T_latent * 200).
        """

        return self.decoder(quantized_features)
