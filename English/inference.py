"""Inference: text -> speech (+ variance-adaptor plots).

For every test sentence (config.yaml -> inference.test_sentences, or --text) this script writes
    <out_dir>/test_generated_audios/sentence_<k>.wav
    <out_dir>/variance_adaptor_plots/duration/sentence_<k>.png   predicted duration per symbol
    <out_dir>/variance_adaptor_plots/pitch/sentence_<k>.png      predicted F0 (Hz) over time
    <out_dir>/variance_adaptor_plots/energy/sentence_<k>.png     predicted energy over time
    <out_dir>/variance_adaptor_plots/variance_values.csv         the numbers behind the plots

Usage (from inside the language folder; runs fine on CPU):
    python inference.py -c config.yaml
    python inference.py -c config.yaml --text "Hello world" --out_dir /tmp/demo
"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np
import torch
from scipy.io import wavfile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from text import clean_to_symbols
from text.symbols import symbols
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import load_configs, expand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}


def find_font_for(text):
    """Return a matplotlib FontProperties able to draw `text`, or None if it needs a font we lack.
    Only Telugu needs special handling (install with: apt-get install -y fonts-noto-core)."""
    if not any(0x0C00 <= ord(c) < 0x0C80 for c in text):
        return None
    keys = ("telugu", "gautami", "pothana", "lohit-te", "mallanna", "vani", "nirmala")
    for path in fm.findSystemFonts(fontext="ttf"):
        if any(k in os.path.basename(path).lower() for k in keys):
            return fm.FontProperties(fname=path)
    return None


def _is_native(ch):
    return ord(ch) > 0x7F


def label_for(ch, font):
    """x-tick label of one symbol. Spaces are left empty (they are drawn as grey bars)."""
    if ch == " ":
        return ""
    if font is None and _is_native(ch):
        from unidecode import unidecode

        return unidecode(ch) or "."
    return ch


def style_ticks(ax, symbol_text, font):
    """Native (Telugu) glyphs use the Telugu font, ASCII punctuation the default font, because
    script-specific fonts often lack Latin glyphs."""
    if font is None:
        return
    for lbl, ch in zip(ax.get_xticklabels(), symbol_text):
        if _is_native(ch):
            lbl.set_fontproperties(font)


def set_titles(ax, k, sentence, font):
    ax.set_title("Sentence {}".format(k), loc="left", fontsize=10, fontweight="bold")
    short = sentence if len(sentence) <= 40 else sentence[:37] + "..."
    if font is None and any(_is_native(c) for c in short):
        from unidecode import unidecode

        short = unidecode(short) + " (romanised)"
        ax.set_title(short, loc="right", fontsize=9)
    elif font is not None and any(_is_native(c) for c in short):
        ax.set_title(short, loc="right", fontsize=10, fontproperties=font)
    else:
        ax.set_title(short, loc="right", fontsize=9)


def prepare_text(sentence, cleaners):
    symbol_text, _ = clean_to_symbols(sentence, cleaners)
    ids = np.array([_SYMBOL_TO_ID[c] for c in symbol_text], dtype=np.int64)
    return symbol_text, ids


def plot_duration(path, k, sentence, symbol_text, dur_ms, font):
    n = len(symbol_text)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * n), 4))
    ax.bar(np.arange(n), dur_ms, color=["#c7c7c7" if c == " " else "tab:blue" for c in symbol_text])
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([label_for(c, font) for c in symbol_text], fontsize=9)
    style_ticks(ax, symbol_text, font)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_xlabel("input symbol (grey bar = space / pause)")
    set_titles(ax, k, sentence, font)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contour(path, k, sentence, symbol_text, values_frames, bounds, hop_sec, ylabel, color, font):
    t = np.arange(len(values_frames)) * hop_sec
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, values_frames, color=color, lw=1.8, drawstyle="steps-post")
    for b, ch in zip(bounds, symbol_text):
        if ch == " ":
            ax.axvline(b * hop_sec, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("time (s)   [dotted lines = word boundaries]")
    ax.set_ylabel(ylabel)
    set_titles(ax, k, sentence, font)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    preprocess_config, model_config, train_config, inf_cfg = load_configs(args.config)
    configs = (preprocess_config, model_config, train_config)
    pre = preprocess_config["preprocessing"]
    sr, hop = pre["audio"]["sampling_rate"], pre["stft"]["hop_length"]
    hop_sec = hop / sr
    cleaners = pre["text"]["text_cleaners"]

    # ---- model / vocoder -------------------------------------------------------------
    ckpt = args.checkpoint or os.path.join(train_config["path"]["release_path"], train_config["export_name"])
    if not os.path.exists(ckpt):
        raise SystemExit("Checkpoint not found: {}\nTrain first, or pass --checkpoint.".format(ckpt))
    args.checkpoint = ckpt
    args.restore_step = 0
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]

    # ---- speaker ---------------------------------------------------------------------
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_map = json.load(f)
    want = args.speaker if args.speaker is not None else inf_cfg.get("speaker", 0)
    if str(want) in speaker_map:
        speaker_id = speaker_map[str(want)]
    else:
        speaker_id = int(want)
    print("Speaker id:", speaker_id, "| available:", list(speaker_map)[:10], "..." if len(speaker_map) > 10 else "")

    sentences = [args.text] if args.text else inf_cfg["test_sentences"]
    out_dir = args.out_dir or inf_cfg["results_path"]
    wav_dir = os.path.join(out_dir, "test_generated_audios")
    plot_root = os.path.join(out_dir, "variance_adaptor_plots")
    for sub in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(plot_root, sub), exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    rows = []
    with open(os.path.join(wav_dir, "test_sentences.txt"), "w", encoding="utf-8") as ts:
        for k, sentence in enumerate(sentences, 1):
            symbol_text, ids = prepare_text(sentence, cleaners)
            if len(ids) == 0:
                print("Skipping sentence {} (no supported characters): {!r}".format(k, sentence))
                continue
            font = find_font_for(symbol_text)
            if font is None and any(ord(c) > 0x7F for c in symbol_text):
                print("(no Telugu font found - plot labels are romanised. `apt-get install -y fonts-noto-core` for native labels)")

            texts = torch.from_numpy(ids).long().unsqueeze(0).to(device)
            src_lens = torch.tensor([len(ids)]).to(device)
            speakers = torch.tensor([speaker_id]).long().to(device)
            with torch.no_grad():
                out = model(
                    speakers, texts, src_lens, len(ids),
                    p_control=args.pitch_control, e_control=args.energy_control, d_control=args.duration_control,
                )
                mel = out[1]
                mel_len = int(out[9][0].item())
                wav = vocoder_infer(
                    mel[:, :mel_len].transpose(1, 2), vocoder, model_config, preprocess_config,
                    lengths=[mel_len * hop],
                )[0]
            wavfile.write(os.path.join(wav_dir, "sentence_{}.wav".format(k)), sr, wav)
            ts.write("sentence_{}.wav\t{}\n".format(k, sentence))

            n = len(ids)
            dur = out[5][0, :n].cpu().numpy()                      # predicted duration (mel frames)
            pitch = out[2][0, :n].cpu().numpy() * p_std + p_mean   # predicted F0 in Hz (per symbol)
            energy = out[3][0, :n].cpu().numpy() * e_std + e_mean  # predicted energy (per symbol)
            if pre["pitch"]["feature"] == "phoneme_level":
                pitch_f = expand(pitch, dur)
            else:
                pitch_f = out[2][0, :mel_len].cpu().numpy() * p_std + p_mean
            if pre["energy"]["feature"] == "phoneme_level":
                energy_f = expand(energy, dur)
            else:
                energy_f = out[3][0, :mel_len].cpu().numpy() * e_std + e_mean
            bounds = np.concatenate([[0], np.cumsum(dur)[:-1]])

            plot_duration(os.path.join(plot_root, "duration", "sentence_{}.png".format(k)), k, sentence,
                          symbol_text, dur * hop_sec * 1000, font)
            plot_contour(os.path.join(plot_root, "pitch", "sentence_{}.png".format(k)), k, sentence, symbol_text,
                         pitch_f, bounds, hop_sec, "predicted F0 (Hz)", "tab:red", font)
            plot_contour(os.path.join(plot_root, "energy", "sentence_{}.png".format(k)), k, sentence, symbol_text,
                         energy_f, bounds, hop_sec, "predicted energy", "tab:purple", font)
            for pos, ch in enumerate(symbol_text):
                rows.append([k, pos, ch, int(dur[pos]), round(float(dur[pos]) * hop_sec * 1000, 1),
                             round(float(pitch[pos]), 2), round(float(energy[pos]), 4)])
            print("sentence {}: {:.2f} s of audio  <-  {!r}".format(k, len(wav) / sr, sentence[:60]))

    with open(os.path.join(plot_root, "variance_values.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "position", "symbol", "duration_frames", "duration_ms", "f0_hz", "energy"])
        w.writerows(rows)
    print("Done. Results in", out_dir)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="default: checkpoints/<export_name> from config.yaml")
    parser.add_argument("--text", default=None, help="synthesize this single text instead of the test sentences")
    parser.add_argument("--speaker", default=None, help="speaker label (see speakers.json) or integer id")
    parser.add_argument("--out_dir", default=None, help="default: inference.results_path from config.yaml")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0, help="larger = slower speech")
    main(parser.parse_args())
