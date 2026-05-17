import os
import random
from glob import glob

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from IPython.display import display

from src.model import SoundStreamModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
CHECKPOINT_PATH = "saved/model_best.pth"

# Sample links
IN_DOMAIN_ID = "1T9ogudzzKiALKBxVNbQqwh-k0_RmUATQ"  # LibriSpeech
EXTERNAL_URL = "https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav"  # LJSpeech
RUSSIAN_CLEAN_ID = "1BBO_AomGFkeYl-srjP4EehpGJkpHiA4v"  # Russian clean
RUSSIAN_NOISY_ID = "1DQwoHC372BGBx1we6zDJehJLL6yK587h"  # Russian noisy


def load_model():
    model = SoundStreamModel(
        in_channels=1,
        base_channels=32,
        latent_dim=128,
        strides=[2, 4, 5, 5],
        num_quantizers=8,
        codebook_size=1024,
    ).to(device)

    path = CHECKPOINT_PATH

    if not os.path.exists(path):
        print("Downloading checkpoint from HuggingFace")
        os.makedirs("saved", exist_ok=True)
        hf_path = hf_hub_download(
            repo_id="serzai/neural-audio-codec",
            filename="checkpoint-epoch100.pth",
            local_dir="saved",
        )
        os.rename(hf_path, CHECKPOINT_PATH)
        path = CHECKPOINT_PATH

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def process_audio(path, model):
    waveform, sr = torchaudio.load(path)
    target_sr = 16000

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform - waveform.mean()
    if waveform.abs().max() > 0:
        waveform /= waveform.abs().max()

    with torch.no_grad():
        input_tensor = waveform.unsqueeze(0).to(device)
        output = model(input_tensor)
        reconstructed = output["fake_audio"].squeeze(0).cpu()

    reconstructed = reconstructed[:, : waveform.shape[1]]

    return waveform, reconstructed


def plot_comparison(waveform, reconstructed, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    n_fft = 400
    hop = 160
    window = torch.hann_window(n_fft)
    target_sr = 16000

    # Waveforms
    axes[0, 0].plot(waveform[0].numpy(), color="blue", alpha=0.7)
    axes[0, 0].set_title(f"{title}: Original waveform")
    axes[0, 1].plot(reconstructed[0].numpy(), color="green", alpha=0.7)
    axes[0, 1].set_title(f"{title}: Reconstructed waveform")

    # Spectrograms
    for i, (wav, t) in enumerate(
        [(waveform[0], "Original"), (reconstructed[0], "Reconstructed")]
    ):
        spec = torch.stft(
            wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
        )
        spec_db = 20 * torch.log10(torch.abs(spec) + 1e-6)

        axes[1, i].imshow(
            spec_db.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, waveform.shape[1] / target_sr, 0, target_sr / 2],
            cmap="viridis",
        )
        axes[1, i].set_title(f"{t} spectrogram")
        axes[1, i].set_ylabel("Frequency, Hz")

    plt.tight_layout()
    plt.show()


def get_in_domain_analysis():
    model = load_model()
    url = f"https://docs.google.com/uc?export=download&id={IN_DOMAIN_ID}"
    os.system(f'wget --no-check-certificate -q "{url}" -O in_domain.wav')

    waveform, reconstructed = process_audio("in_domain.wav", model)
    plot_comparison(waveform, reconstructed, "LibriSpeech")

    print("Original audio:")
    display(ipd.Audio(waveform[0].numpy(), rate=16000))

    print("Reconstructed audio:")
    display(ipd.Audio(reconstructed[0].numpy(), rate=16000))


def get_external_analysis():
    model = load_model()
    os.system(f'wget -q "{EXTERNAL_URL}" -O external.wav')

    waveform, reconstructed = process_audio("external.wav", model)
    plot_comparison(waveform, reconstructed, "LJSpeech")

    print("Original audio:")
    display(ipd.Audio(waveform[0].numpy(), rate=16000))

    print("Reconstructed audio:")
    display(ipd.Audio(reconstructed[0].numpy(), rate=16000))


def get_russian_analysis():
    model = load_model()

    print("Clean Russian Speech")
    url_clean = f"https://docs.google.com/uc?export=download&id={RUSSIAN_CLEAN_ID}"
    os.system(f'wget -q "{url_clean}" -O russian_clean.wav')

    waveform, reconstructed = process_audio("russian_clean.wav", model)
    plot_comparison(waveform, reconstructed, "Clean Russian Speech")

    print("Original audio:")
    display(ipd.Audio(waveform[0].numpy(), rate=16000))

    print("Reconstructed audio:")
    display(ipd.Audio(reconstructed[0].numpy(), rate=16000))

    print("\nNoisy Russian Speech")
    url_noisy = f"https://docs.google.com/uc?export=download&id={RUSSIAN_NOISY_ID}"
    os.system(f'wget -q "{url_noisy}" -O russian_noisy.wav')

    waveform, reconstructed = process_audio("russian_noisy.wav", model)
    plot_comparison(waveform, reconstructed, "Russian speech (noisy)")

    print("Original audio:")
    display(ipd.Audio(waveform[0].numpy(), rate=16000))

    print("Reconstructed audio:")
    display(ipd.Audio(reconstructed[0].numpy(), rate=16000))
