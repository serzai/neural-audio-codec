import random
from pathlib import glob

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    """
    Dataset for LibriSpeech.
    """

    def __init__(self, data_dir, segment_len=8000, sample_rate=16000):
        """
        Args:
            data_dir (str): path to partition.
            segment_len (int): target length in samples.
            sample_rate (int): target sample rate.
        """
        super().__init__()
        self.segment_len = segment_len
        self.sample_rate = sample_rate

        self.audio_files = sorted(
            list(torch.utils.data.dataset.glob.glob(f"{data_dir}/**/*.flac"))
        )

        if len(self.audio_files) == 0:
            raise FileNotFoundError(f"No .flac files found in {data_dir}")

    def __len__(self):
        """
        Returns:
            output (int): number of files in dataset
        """
        return len(self.audio_files)

    def __getitem__(self, key):
        """
        Args:
            key (int): index of the file.
        Returns:
            output (dict): dict with audio tensor of shape (1, segment_length).
        """
        path = self.audio_files[key]
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform - waveform.mean()
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform /= max_val

        num_samples = waveform.size(1)
        if num_samples > self.segment_len:
            max_start = num_samples - self.segment_len
            start = random.randint(0, max_start)
            waveform = waveform[:, start : start + self.segment_len]
        else:
            pad = self.segment_len - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad), mode="replicate")

        return {"audio": waveform}
