"""Model / checkpoint / vocoder helpers.

Modified from https://github.com/ming024/FastSpeech2 (utils/model.py):
  * checkpoints are loaded with map_location, so a GPU-trained checkpoint also loads on a
    CPU-only machine (the upstream version crashes there);
  * `get_model` can load (a) a full training checkpoint (resume), (b) a model-only,
    possibly fp16, exported checkpoint (inference) and (c) a *pretrained* checkpoint used
    to warm-start fine-tuning (only tensors whose name and shape match are copied);
  * the HiFi-GAN vocoder weights are downloaded automatically on first use instead of
    being committed to the repository (they are ~50 MB each).
"""
import json
import os
import urllib.request
import zipfile

import numpy as np
import torch

import hifigan
from model import FastSpeech2, ScheduledOptim

_VOCODER_URL = (
    "https://github.com/ming024/FastSpeech2/raw/master/hifigan/"
    "generator_{speaker}.pth.tar.zip"
)

# Tensors that depend on the *dataset statistics* or on the number of speakers must never be
# copied from a pretrained model, even when their shapes happen to match.
_NEVER_TRANSFER = ("pitch_bins", "energy_bins", "speaker_emb")


def torch_load(path, device="cpu"):
    """torch.load that works across torch versions (weights_only default changed in 2.6)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # very old torch without the weights_only argument
        return torch.load(path, map_location=device)


def _to_float(state_dict):
    """Exported checkpoints store weights as fp16 to stay below GitHub's 100 MB limit."""
    return {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in state_dict.items()
    }


def load_pretrained(model, path, device="cpu"):
    """Warm-start `model` from another FastSpeech2 checkpoint (e.g. the LJSpeech one).

    Only tensors with identical name AND shape are copied. For Telugu the symbol
    embedding has a different shape and is therefore re-initialised, while the encoder /
    decoder / variance-adaptor weights are re-used (cross-lingual transfer).
    """
    ckpt = torch_load(path, device)
    src = _to_float(ckpt["model"] if "model" in ckpt else ckpt)
    dst = model.state_dict()
    copied, skipped = [], []
    for name, tensor in src.items():
        if (
            name in dst
            and dst[name].shape == tensor.shape
            and not any(bad in name for bad in _NEVER_TRANSFER)
        ):
            dst[name] = tensor
            copied.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(dst)
    print(
        "Warm start from {}: copied {} tensors, skipped {} (shape mismatch / dataset-specific)".format(
            path, len(copied), len(skipped)
        )
    )
    return model


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)

    ckpt = None
    checkpoint = getattr(args, "checkpoint", None)
    restore_step = getattr(args, "restore_step", 0)
    if checkpoint:
        ckpt = torch_load(checkpoint, device)
    elif restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"], "{}.pth.tar".format(restore_step)
        )
        ckpt = torch_load(ckpt_path, device)

    if ckpt is not None:
        model.load_state_dict(_to_float(ckpt["model"]))
    elif train and getattr(args, "pretrained", None):
        load_pretrained(model, args.pretrained, device)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step if ckpt is not None else 0
        )
        if ckpt is not None and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _ensure_vocoder_weights(speaker):
    path = "hifigan/generator_{}.pth.tar".format(speaker)
    if os.path.exists(path):
        return path
    zip_path = path + ".zip"
    if not os.path.exists(zip_path):
        url = _VOCODER_URL.format(speaker=speaker)
        print("Downloading HiFi-GAN vocoder weights from", url)
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("hifigan")
    return path


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    if name != "HiFi-GAN":
        raise ValueError("Only the HiFi-GAN vocoder is supported in this repository")

    with open("hifigan/config.json", "r") as f:
        cfg = json.load(f)
    cfg = hifigan.AttrDict(cfg)
    vocoder = hifigan.Generator(cfg)
    ckpt = torch_load(_ensure_vocoder_weights(speaker), device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def export_for_release(full_ckpt_path, out_path, extra=None):
    """Write a small, model-only checkpoint (fp16 weights) suitable for GitHub (<100 MB)."""
    ckpt = torch_load(full_ckpt_path, "cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    small = {}
    for k, v in state.items():
        keep_fp32 = ("bins" in k) or ("position_enc" in k)
        small[k] = v.half() if (v.is_floating_point() and not keep_fp32) else v
    payload = {"model": small}
    if extra:
        payload.update(extra)
    torch.save(payload, out_path)
    return os.path.getsize(out_path) / 1e6
