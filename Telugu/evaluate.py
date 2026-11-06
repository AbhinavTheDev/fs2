"""Validation loop (modified from upstream FastSpeech2 evaluate.py).

Changes: single config.yaml, `evaluate` returns the numeric mean losses in addition to the log
message (needed for validation_loss.png), vocoder / TensorBoard output is optional.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, load_configs
from model import FastSpeech2Loss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        log(logger, step, losses=loss_means)
        if vocoder is not None:
            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                batch, output, vocoder, model_config, preprocess_config
            )
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            log(logger, fig=fig, tag="Validation/step_{}_{}".format(step, tag))
            log(logger, audio=wav_reconstruction, sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_reconstructed".format(step, tag))
            log(logger, audio=wav_prediction, sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_synthesized".format(step, tag))

    return message, loss_means


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint (.pth.tar)")
    args = parser.parse_args()
    args.restore_step = 0

    p, m, t, _ = load_configs(args.config)
    configs = (p, m, t)
    model = get_model(args, configs, device, train=False).to(device)
    message, _ = evaluate(model, 0, configs)
    print(message)
