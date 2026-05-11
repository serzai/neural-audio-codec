import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class DiscriminatorLoss(nn.Module):
    """
    Hinge loss for discriminators (Eq. 2).
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_real_scores, d_fake_scores, **batch):
        """
        Args:
            d_real_scores (list): list of tensors containing discriminator scores for real audio.
            d_fake_scores (list): list of tensors containing discriminator scores for fake audio.
        Returns:
            output (dict): dict with loss value.
        """
        loss = 0.0

        for real_score, fake_score in zip(d_real_scores, d_fake_scores):
            real_loss = torch.mean(F.relu(1 - real_score))
            fake_loss = torch.mean(F.relu(1 + fake_score))
            loss += real_loss + fake_loss

        final_loss = loss / len(d_real_scores)

        return {"loss": final_loss}


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-Scale spectral reconstruction loss (Eq. 4 & 5).
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=64,
        window_sizes=[64, 128, 256, 512, 1024, 2048],
    ):
        super().__init__()
        self.transforms = nn.ModuleList()
        self.alphas = []

        for size in window_sizes:
            hop_length = size // 4
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=size,
                win_length=size,
                hop_length=hop_length,
                n_mels=n_mels,
                power=1.0,
                normalized=True,
                center=False,
            )
            self.transforms.append(transform)
            self.alphas.append(math.sqrt(size / 2))

    def forward(self, x_real, x_fake):
        """
        Args:
            x_real (torch.Tensor): real audio tensor of shape (B, 1, T).
            x_fake (torch.Tensor): reconstruced audio tensor of shape (B, 1, T).
        """
        eps = 1e-5
        loss = 0.0

        for transform, alpha in zip(self.transforms, self.alphas):
            mel_real = transform(x_real)
            mel_fake = transform(x_fake)

            l1_loss = torch.mean(torch.abs(mel_real - mel_fake))
            l2_loss = alpha * torch.mean(
                (
                    torch.log(torch.clamp(mel_real, min=eps))
                    - torch.log(torch.clamp(mel_fake, min=eps))
                )
                ** 2
            )

            loss += l1_loss + l2_loss

        return loss


class GeneratorLoss(nn.Module):
    """
    Overall generator loss (Eq. 6).
    """

    def __init__(self, lambda_adv=1.0, lambda_feat=100.0, lambda_rec=1.0):
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.spectral_loss = MultiScaleSpectralLoss()

    def forward(
        self, x_real, x_fake, d_real_features, d_fake_features, d_fake_scores, **batch
    ):
        """
        Args:
            x_real (torch.Tensor): real audio tensor of shape (B, 1, T).
            x_fake (torch.Tensor): fake audio tensor of shape (B, 1, T).
            d_real_features (torch.Tensor): discriminator activations for real audio.
            d_fake_features (torch.Tensor): discriminator activations for fake audio.
            d_fake_scores: final discriminator scores for fake audio.
        Returns:
            output (dict): dict with loss values (loss, adv_loss, feat_loss, rec_loss).
        """
        adv_loss = 0.0
        for score in d_fake_scores:
            adv_loss += torch.mean(F.relu(1 - score))
        adv_loss = adv_loss / len(d_fake_scores)

        feat_loss = 0.0
        for feat_real, feat_fake in zip(d_real_features, d_fake_features):
            feat_loss += torch.mean(torch.abs(feat_real - feat_fake))
        feat_loss = feat_loss / len(d_real_features)

        rec_loss = self.spectral_loss(x_real, x_fake)

        total_loss = (
            self.lambda_adv * adv_loss
            + self.lambda_feat * feat_loss
            + self.lambda_rec * rec_loss
        )

        return {
            "loss": total_loss,
            "adv_loss": adv_loss,
            "feat_loss": feat_loss,
            "rec_loss": rec_loss,
        }
