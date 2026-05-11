import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """
    Single-layer vector quantization
    """

    def __init__(self, codebook_size, embedding_dim):
        """
        Args:
            codebook_size (int): number of vectors in the codebook.
            embedding_dim (int): dim of vectors
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): tensor of shape (B, D, T).
        Returns:
            z_q (torch.Tensor): quantized tensor of shape (B, D, T).
            loss (torch.Tensor): commitment loss.
        """
        z_T = z.transpose(1, 2).contiguous()
        flat_z = z_T.view(-1, self.embedding_dim)

        dist = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.t())
        )

        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook(indices).view(z_T.shape)
        z_q = z_q.transpose(1, 2).contiguous()

        loss = torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()

        return z_q, loss


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (Section 3.3, Algorithm 1)
    """

    def __init__(self, num_quantizers=8, codebook_size=1024, embedding_dim=128):
        """
        Args:
            num_quantizers (int): number of quantization stages.
            codebook_size (int): number of vectors in the codebook.
            embedding_dim (int): dim of vectors.
        """
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim)
                for i in range(num_quantizers)
            ]
        )

    def forward(self, y):
        """
        Args:
            y (torch.Tensor): tensor of shape (B, D, T).
        Returns:
            y_q (torch.Tensor): quantized tensor of shape (B, D, T).
            total_loss (torch.Tensor): total commitment loss.
        """
        y_q = 0.0
        residual = y
        total_loss = 0.0
        for quantizer in self.quantizers:
            z_q_i, loss_i = quantizer(residual)
            y_q += z_q_i
            residual -= z_q_i.detach()
            total_loss += loss_i

        return y_q, total_loss
