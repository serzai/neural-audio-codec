# SoundStream Neural Audio Codec

This is an implementation of the [SoundStream](https://arxiv.org/abs/2107.03312) neural audio codec.

## Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/serzai/neural-audio-codec.git
cd neural-audio-codec
pip install -r requirements.txt
```

## Downloading Weights

Final model trained for 45000 steps on the `train-clean-100` split of LibriSpeech. You can download the checkpoint from Hugging Face directly into the `saved/` using the code below.

```bash
mkdir -p saved
wget "https://huggingface.co/serzai/neural-audio-codec/resolve/main/checkpoint-epoch100.pth" -O saved/model_best.pth
```

## Running the Demo

The repository includes a `demo.ipynb` notebook. You can open it in Google Colab/locally. It will download the weights, let you provide a URL to a custom `.wav` file, and play back the original and the reconstructed audio.

## Inference

To calculate the final STOI and NISQA metrics on the LibriSpeech `test-clean` dataset, use the `inference.py` script.

```bash
python inference.py \
    datasets=librispeech_test \
    datasets.test.data_dir="path to dataset" \
    inferencer.from_pretrained="saved/model_best.pth"
```

My final run achieved a STOI of 0.931 and a NISQA of 3.32 on the full `test-clean` set.

## Training from Scratch

If you want to reproduce the training process, you need the `train-clean-100` and `test-clean` partitions.

```bash
python train.py \
    writer.run_name="training" \
    datasets.train.data_dir="path to train-clean-100" \
    datasets.val.data_dir="path to test-clean"
```

## Training Logs

All training metrics were logged to CometML. You can view the full training curves in
[CometML Dashboard](https://www.comet.com/cep3au/neural-audio-codec/dyco8oaromx9sb5zur6iiu5dp768jwv3).

## Report

Report with curves is available in [CometML Report](https://www.comet.com/cep3au/neural-audio-codec/reports/DuOnraTnTzpIFirJJ7VQaxNwk).
Analysis of the results is located in the `report.ipynb`.

## Template

Project based on template:
> Grinberg, P. (2024). *PyTorch Project Template* [Computer software]. https://github.com/Blinorot/pytorch_project_template
