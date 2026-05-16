import torch
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    ShortTimeObjectiveIntelligibility,
)

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    """
    STOI metric.
    """

    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """

        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = metric.to(device)

    def __call__(self, audio, fake_audio, **kwargs):
        """
        Args:
            audio (torch.Tensor): audio tensor of real audio (B, 1, T).
            fake_audio (torch.Tensor): audio tensor of reconstructed audio (B, 1, T).
        Returns:
            stoi (float): STOI metric.
        """
        audio = audio.squeeze(1)
        fake_audio = fake_audio.squeeze(1)

        real_len = audio.shape[-1]
        fake_len = fake_audio.shape[-1]

        if fake_len > real_len:
            fake_audio = fake_audio[..., :real_len]
        elif fake_len < real_len:
            fake_audio = torch.nn.functional.pad(fake_audio, (0, real_len - fake_len))

        return self.metric(fake_audio, audio).mean()


class NISQAMetric(BaseMetric):
    """
    NISQA metric.
    """

    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """

        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = metric.to(device)

    def __call__(self, fake_audio, **kwargs):
        """
        Args:
            fake_audio (torch.Tensor): audio tensor of reconstructed audio (B, 1, T).
        """
        fake_audio = fake_audio.squeeze(1)

        return self.metric(fake_audio).mean()
