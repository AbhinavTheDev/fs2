"""Pre-processing for FastSpeech2 fine-tuning (English / Svarah and Telugu / IndicTTS).

Upstream FastSpeech2 needs phoneme durations from the Montreal Forced Aligner (MFA). MFA can only
be installed through conda and has no ready-made Telugu model, so this script replaces it with a
*CTC forced aligner* that needs nothing but `pip`:

  1. a pretrained wav2vec2-CTC model (config: preprocess.aligner.model) produces frame-level
     character probabilities for each utterance;
  2. a small Viterbi routine (`ctc_forced_align`, pure numpy) forces the known transcript onto
     those probabilities -> start / end time of every character;
  3. `build_durations` turns the character times into per-symbol durations in mel frames
     (silence between words is given to the space symbol) and tells us where speech starts / ends;
  4. the rest is the original FastSpeech2 recipe: mel-spectrogram, energy (L2 norm of the STFT
     magnitude) and pitch (WORLD DIO + StoneMask), both averaged per symbol, then z-normalised.

The text input of the model is therefore *character level* (letters for English, Unicode code
points for Telugu), see text/symbols.py.

Usage (run from inside the language folder, on a GPU runtime):
    python preprocess.py -c config.yaml                 # full run (resumable)
    python preprocess.py -c config.yaml --max_utts 50   # quick smoke test
    python preprocess.py -c config.yaml --restart       # wipe output and start again
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text import clean_to_symbols
from utils.tools import load_configs

CTC_SR = 16000  # sampling rate expected by wav2vec2

_BREAK_CHARS = set(" ,.?!;:-")


# --------------------------------------------------------------------------------------
# 1. CTC forced alignment (replacement for the Montreal Forced Aligner)
# --------------------------------------------------------------------------------------
def ctc_forced_align(log_probs, targets, blank=0):
    """Viterbi forced alignment of `targets` (list of token ids) to CTC `log_probs` (T, V).

    Returns (spans, score) where spans[k] = (start_frame, end_frame_exclusive) of target k and
    score is the mean log-probability along the best path (closer to 0 = better match).
    Returns None if the transcript cannot be aligned (e.g. audio too short for the text).
    """
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0 or T < L:
        return None
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    emit = log_probs[:, ext].astype(np.float64)  # (T, S)

    # a transition s-2 -> s is only allowed into a label state whose label differs from s-2's
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = emit[0, 0]
    alpha[1] = emit[0, 1]
    back = np.zeros((T, S), dtype=np.int8)
    idx = np.arange(S)
    for t in range(1, T):
        stay = alpha
        prev = np.concatenate(([NEG], alpha[:-1]))
        skip = np.concatenate(([NEG, NEG], alpha[:-2]))
        skip = np.where(can_skip, skip, NEG)
        cand = np.stack([stay, prev, skip])  # (3, S)
        best = cand.argmax(axis=0)
        alpha = cand[best, idx] + emit[t]
        back[t] = best

    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    final = alpha[s]
    if final < -1e20:  # no valid path
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])

    starts, ends = {}, {}
    for t, st in enumerate(path):
        if st % 2 == 1:
            k = st // 2
            if k not in starts:
                starts[k] = t
            ends[k] = t + 1
    spans = [(starts[k], ends[k]) for k in range(L)]
    return spans, float(final / T)


class CTCAligner:
    """wav2vec2-CTC wrapper: waveform + transcript -> character spans."""

    def __init__(self, model_name, device, hf_token=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        kw = {"token": hf_token} if hf_token else {}
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, **kw)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, **kw).to(device).eval()
        tok = self.processor.tokenizer
        self.vocab = tok.get_vocab()
        self.blank = tok.pad_token_id
        self.delim = self.vocab.get(tok.word_delimiter_token)
        self.frame_sec = float(np.prod(self.model.config.conv_stride)) / CTC_SR

    def char_id(self, ch):
        for cand in (ch, ch.upper(), ch.lower()):
            if cand in self.vocab and cand not in ("|",):
                return self.vocab[cand]
        return None

    @torch.no_grad()
    def emissions(self, wav16k):
        inputs = self.processor(wav16k, sampling_rate=CTC_SR, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.device)).logits
        return torch.log_softmax(logits.float(), dim=-1)[0].cpu().numpy()

    def align(self, wav16k, symbol_text):
        return align_symbols(self.emissions(wav16k), symbol_text, self.char_id, self.delim, self.blank, self.frame_sec)


def align_symbols(log_probs, symbol_text, char_id_fn, delim_id, blank, frame_sec):
    """Map every alignable character of `symbol_text` to a time span (in seconds).

    Characters without a token in the CTC vocabulary (punctuation, rare letters) are simply not
    aligned; they receive duration 0 (or the pause between words, for spaces).
    Returns dict(spans={symbol_index: (t_start, t_end)}, score, n_missing) or None.
    """
    targets, owner = [], []
    pending_break = False
    n_missing = 0
    for i, ch in enumerate(symbol_text):
        if ch in _BREAK_CHARS:
            pending_break = True
            continue
        cid = char_id_fn(ch)
        if cid is None:
            n_missing += 1
            continue
        if pending_break and owner and delim_id is not None:
            targets.append(delim_id)
            owner.append(None)
        pending_break = False
        targets.append(cid)
        owner.append(i)
    n_letters = sum(o is not None for o in owner)
    if n_letters == 0:
        return None
    res = ctc_forced_align(log_probs, targets, blank)
    if res is None:
        return None
    spans, score = res
    out = {}
    for (s, e), o in zip(spans, owner):
        if o is not None:
            out[o] = (s * frame_sec, e * frame_sec)
    return {"spans": out, "score": score, "n_missing": n_missing}


def build_durations(symbol_text, spans, sampling_rate, hop_length, audio_seconds, max_edge=0.06):
    """Convert character time spans to integer durations (mel frames) for every symbol.

    * letters inside a word share the gap between them (boundary = midpoint);
    * between words each neighbouring letter may extend by at most `max_edge` seconds and the
      remaining silence is assigned to the space (or punctuation) symbol between them;
    * symbols without a span (punctuation next to letters) get duration 0.
    The speech region [t_start, t_end] excludes leading / trailing silence.

    Returns (durations, t_start, t_end) with sum(durations) == number of mel frames of the region.
    """
    idxs = sorted(spans)
    if not idxs:
        return None
    n = len(symbol_text)
    st = {i: spans[i][0] for i in idxs}
    en = {i: spans[i][1] for i in idxs}
    left, right = {}, {}
    first, last = idxs[0], idxs[-1]
    left[first] = max(0.0, st[first] - max_edge)
    right[last] = min(audio_seconds, en[last] + max_edge)
    holder = {}  # symbol index that owns the pause after letter `a`  -> (a, b)
    for a, b in zip(idxs[:-1], idxs[1:]):
        gap = max(0.0, st[b] - en[a])
        if b - a == 1:  # neighbours in the same word
            mid = en[a] + gap / 2
            right[a] = mid
            left[b] = mid
        else:
            e = min(gap / 2, max_edge)
            right[a] = en[a] + e
            left[b] = st[b] - e
            between = range(a + 1, b)
            h = next((j for j in between if symbol_text[j] == " "), a + 1)
            holder[h] = (a, b)
    t0 = left[first]
    if right[last] <= t0:
        return None

    def f(t):
        return int(round((t - t0) * sampling_rate / hop_length))

    dur = [0] * n
    for i in idxs:
        dur[i] = max(0, f(right[i]) - f(left[i]))
    for h, (a, b) in holder.items():
        dur[h] = max(0, f(left[b]) - f(right[a]))
    return dur, t0, right[last]


# --------------------------------------------------------------------------------------
# 2. audio / dataset helpers
# --------------------------------------------------------------------------------------
def decode_audio(field):
    """Decode a Hugging Face audio field loaded with Audio(decode=False) -> (mono float32, sr).

    Decoding the raw bytes ourselves avoids the torchcodec / ffmpeg dependency of recent
    `datasets` versions."""
    if isinstance(field, dict) and field.get("array") is not None:
        return np.asarray(field["array"], dtype=np.float32), int(field["sampling_rate"])
    src = None
    if isinstance(field, dict):
        if field.get("bytes") is not None:
            import io

            src = io.BytesIO(field["bytes"])
        elif field.get("path"):
            src = field["path"]
    if src is None:
        raise ValueError("audio field has neither bytes nor path")
    try:
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
    except Exception:
        if hasattr(src, "seek"):
            src.seek(0)
        wav, sr = librosa.load(src, sr=None, mono=True)
    return wav.astype(np.float32), int(sr)


def speaker_key(example, columns, features):
    parts = []
    for c in columns:
        if c not in example:
            continue
        v = example[c]
        feat = features.get(c)
        if hasattr(feat, "int2str") and isinstance(v, (int, np.integer)):
            v = feat.int2str(int(v))
        parts.append(str(v).strip().lower().replace(" ", "-"))
    return "_".join(parts) if parts else "speaker0"


def speaker_label(key):
    """File-name friendly speaker id: short alphanumeric keys are kept, others are hashed."""
    if re.fullmatch(r"[A-Za-z0-9]{1,20}", key):
        return key
    return "S" + hashlib.md5(key.encode("utf-8")).hexdigest()[:6]


# --------------------------------------------------------------------------------------
# 3. the preprocessor
# --------------------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, config, aligner):
        self.config = config
        pre = config["preprocessing"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sr = pre["audio"]["sampling_rate"]
        self.hop = pre["stft"]["hop_length"]
        self.cleaners = pre["text"]["text_cleaners"]
        self.pitch_avg = pre["pitch"]["feature"] == "phoneme_level"
        self.energy_avg = pre["energy"]["feature"] == "phoneme_level"
        self.pitch_norm = pre["pitch"]["normalization"]
        self.energy_norm = pre["energy"]["normalization"]
        flt = config.get("filters", {})
        self.min_sec = flt.get("min_seconds", 0.8)
        self.max_raw_sec = flt.get("max_raw_seconds", 20.0)
        self.max_frames = flt.get("max_mel_frames", 940)
        self.min_chars = flt.get("min_chars", 3)
        self.min_score = flt.get("min_align_score", -1.0)
        self.aligner = aligner
        self.STFT = Audio.stft.TacotronSTFT(
            pre["stft"]["filter_length"],
            pre["stft"]["hop_length"],
            pre["stft"]["win_length"],
            pre["mel"]["n_mel_channels"],
            pre["audio"]["sampling_rate"],
            pre["mel"]["mel_fmin"],
            pre["mel"]["mel_fmax"],
        )
        for sub in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

    # ---- one utterance -------------------------------------------------------------
    def process_utterance(self, wav, sr, raw_text, spk_label, basename):
        """Returns (status, info). status == "ok" or a short reason string for dropped items."""
        seconds = len(wav) / sr
        if seconds < self.min_sec:
            return "too_short", None
        if seconds > self.max_raw_sec:
            return "too_long", None

        symbol_text, n_dropped = clean_to_symbols(raw_text, self.cleaners)
        if n_dropped > 0:
            return "unsupported_chars", None
        if len(symbol_text) < self.min_chars:
            return "text_too_short", None

        wav = np.nan_to_num(wav.astype(np.float32))
        peak = np.max(np.abs(wav))
        if peak < 1e-4:
            return "silent", None
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=CTC_SR) if sr != CTC_SR else wav
        wav22 = librosa.resample(wav, orig_sr=sr, target_sr=self.sr) if sr != self.sr else wav
        wav22 = wav22 / max(np.max(np.abs(wav22)), 1e-9) * 0.999  # peak normalisation (as upstream)

        ali = self.aligner.align(wav16, symbol_text)
        if ali is None:
            return "align_failed", None
        if ali["score"] < self.min_score:
            return "bad_alignment", None
        if ali["n_missing"] > 0.1 * len(symbol_text):
            return "too_many_unaligned_chars", None

        res = build_durations(symbol_text, ali["spans"], self.sr, self.hop, len(wav22) / self.sr)
        if res is None:
            return "bad_durations", None
        duration, t0, t1 = res
        n_frames = int(sum(duration))
        if n_frames < 10:
            return "bad_durations", None
        if n_frames > self.max_frames:
            return "too_long_after_trim", None

        seg = wav22[int(self.sr * t0): int(self.sr * t1)].astype(np.float32)
        feats = self.extract_features(seg, duration)
        if feats is None:
            return "no_pitch", None
        mel, pitch, energy = feats

        np.save(os.path.join(self.out_dir, "duration", "{}-duration-{}.npy".format(spk_label, basename)), np.array(duration))
        np.save(os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(spk_label, basename)), pitch)
        np.save(os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(spk_label, basename)), energy)
        np.save(os.path.join(self.out_dir, "mel", "{}-mel-{}.npy".format(spk_label, basename)), mel.T)
        info = {
            "id": basename,
            "speaker": spk_label,
            "text": symbol_text,
            "raw": raw_text.replace("|", " ").replace("\n", " ").strip(),
            "frames": n_frames,
            "score": round(ali["score"], 4),
        }
        return "ok", info

    def extract_features(self, wav, duration):
        total = int(sum(duration))
        w64 = wav.astype(np.float64)
        pitch, t = pw.dio(w64, self.sr, frame_period=self.hop / self.sr * 1000)
        pitch = pw.stonemask(w64, pitch, t, self.sr)
        mel, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        def fit(x, axis=-1):  # trim / edge-pad to exactly `total` frames
            n = x.shape[axis]
            if n >= total:
                return x[..., :total] if axis == -1 else x[:total]
            if total - n > 3:
                return None
            pad = [(0, 0)] * x.ndim
            pad[axis] = (0, total - n)
            return np.pad(x, pad, mode="edge")

        pitch, energy, mel = fit(pitch, 0), fit(energy, 0), fit(mel, -1)
        if pitch is None or energy is None or mel is None:
            return None
        if np.sum(pitch != 0) <= 1:
            return None

        dur = np.array(duration)
        if self.pitch_avg:
            nz = np.where(pitch != 0)[0]
            interp = interp1d(nz, pitch[nz], fill_value=(pitch[nz[0]], pitch[nz[-1]]), bounds_error=False)
            pitch = self._avg(interp(np.arange(len(pitch))), dur)
        if self.energy_avg:
            energy = self._avg(energy, dur)
        return mel, pitch.astype(np.float32), energy.astype(np.float32)

    @staticmethod
    def _avg(values, dur):
        """Average `values` (frame level) over every symbol. Symbols of duration 0 receive the
        utterance mean (upstream wrote 0 here and could also corrupt later averages)."""
        out = np.zeros(len(dur), dtype=np.float64)
        pos = 0
        for i, d in enumerate(dur):
            if d > 0:
                out[i] = np.mean(values[pos: pos + d])
            pos += d
        mean = out[dur > 0].mean() if np.any(dur > 0) else 0.0
        out[dur == 0] = mean
        return out

    # ---- global statistics / normalisation ---------------------------------------------
    @staticmethod
    def remove_outlier(values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lower, upper = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
        return values[np.logical_and(values > lower, values < upper)]

    def finalize(self, entries, val_size, seed=1234):
        """Compute pitch / energy statistics, z-normalise the saved arrays in place and write
        stats.json, speakers.json, train.txt, val.txt and preprocess_summary.json."""
        flag = os.path.join(self.out_dir, "normalized.flag")
        if os.path.exists(flag):
            raise RuntimeError("Features are already normalised. Use --restart to rebuild from scratch.")
        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        n_frames = 0
        for e in tqdm(entries, desc="Statistics"):
            p = np.load(self._path("pitch", e))
            en = np.load(self._path("energy", e))
            dur = np.load(self._path("duration", e))
            keep = dur > 0
            if keep.sum() > 3:
                pitch_scaler.partial_fit(self.remove_outlier(p[keep]).reshape(-1, 1))
                energy_scaler.partial_fit(self.remove_outlier(en[keep]).reshape(-1, 1))
            n_frames += e["frames"]

        p_mean, p_std = (pitch_scaler.mean_[0], pitch_scaler.scale_[0]) if self.pitch_norm else (0.0, 1.0)
        e_mean, e_std = (energy_scaler.mean_[0], energy_scaler.scale_[0]) if self.energy_norm else (0.0, 1.0)
        p_min, p_max = self._normalize("pitch", entries, p_mean, p_std)
        e_min, e_max = self._normalize("energy", entries, e_mean, e_std)

        labels = sorted({e["speaker"] for e in entries})
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump({l: i for i, l in enumerate(labels)}, f, indent=1)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump(
                {
                    "pitch": [float(p_min), float(p_max), float(p_mean), float(p_std)],
                    "energy": [float(e_min), float(e_max), float(e_mean), float(e_std)],
                },
                f,
            )
        random.Random(seed).shuffle(entries)
        lines = ["|".join([e["id"], e["speaker"], e["text"], e["raw"]]) for e in entries]
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines[val_size:]) + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:val_size]) + "\n")
        open(flag, "w").write("done")
        hours = n_frames * self.hop / self.sr / 3600
        return {
            "n_utterances": len(entries),
            "n_train": len(entries) - val_size,
            "n_val": val_size,
            "n_speakers": len(labels),
            "hours": round(hours, 3),
            "pitch_mean_hz": float(p_mean),
            "pitch_std_hz": float(p_std),
            "mean_align_score": float(np.mean([e["score"] for e in entries])),
        }

    def _path(self, kind, e):
        return os.path.join(self.out_dir, kind, "{}-{}-{}.npy".format(e["speaker"], kind, e["id"]))

    def _normalize(self, kind, entries, mean, std):
        lo, hi = np.inf, -np.inf
        for e in tqdm(entries, desc="Normalising " + kind):
            path = self._path(kind, e)
            v = (np.load(path) - mean) / std
            np.save(path, v.astype(np.float32))
            lo, hi = min(lo, v.min()), max(hi, v.max())
        return lo, hi


# --------------------------------------------------------------------------------------
# 4. driver
# --------------------------------------------------------------------------------------
def read_index(path):
    """index.jsonl stores one line per processed utterance (also dropped ones), which makes the
    script resumable after a Colab disconnect."""
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except Exception:
                    pass
    return done


def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config, _, _, _ = load_configs(args.config)
    out_dir = config["path"]["preprocessed_path"]
    if args.restart and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from datasets import Audio as HFAudio
    from datasets import load_dataset

    token = args.hf_token or os.environ.get("HF_TOKEN")
    kw = {"token": token} if token else {}
    ds_cfg = config["hf_dataset"]
    print("Loading dataset", ds_cfg["name"], "split", ds_cfg["split"])
    ds = load_dataset(ds_cfg["name"], split=ds_cfg["split"], **kw)
    audio_col = ds_cfg["audio_column"]
    if audio_col not in ds.features:  # auto-detect
        audio_col = next(k for k, v in ds.features.items() if isinstance(v, HFAudio))
        print("Audio column auto-detected:", audio_col)
    text_col = ds_cfg["text_column"]
    ds = ds.cast_column(audio_col, HFAudio(decode=False))
    spk_cols = config["speaker_columns"]
    print("Rows:", len(ds), "| columns:", list(ds.features))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found - the wav2vec2 aligner will be slow. Use a GPU runtime.")
    aligner = CTCAligner(config["aligner"]["model"], device, token)
    pre = Preprocessor(config, aligner)

    index_path = os.path.join(out_dir, "index.jsonl")
    done = read_index(index_path)
    print("Already processed (resume):", len(done))
    prefix = ds_cfg.get("id_prefix", "utt")
    n_rows = len(ds) if not args.max_utts else min(args.max_utts, len(ds))
    reasons = Counter(r["status"] for r in done.values())
    t_start = time.time()
    with open(index_path, "a", encoding="utf-8") as idx_file:
        for i in tqdm(range(n_rows), desc="Preprocessing"):
            basename = "{}{:05d}".format(prefix, i)
            if basename in done:
                continue
            ex = ds[i]
            try:
                wav, sr = decode_audio(ex[audio_col])
                key = speaker_key(ex, spk_cols, ds.features)
                label = speaker_label(key)
                status, info = pre.process_utterance(wav, sr, ex[text_col], label, basename)
            except Exception as err:  # never let one bad file stop an hours-long run
                status, info = "error:" + type(err).__name__, None
            rec = info if status == "ok" else {"id": basename}
            rec = dict(rec, status=status)
            if status == "ok":
                rec["speaker_key"] = key
            idx_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx_file.flush()
            done[basename] = rec
            reasons[status] += 1

    ok = [r for r in done.values() if r["status"] == "ok"]
    print("\nKept {} / {} utterances. Outcome counts: {}".format(len(ok), len(done), dict(reasons)))
    if len(ok) < config["preprocessing"]["val_size"] * 3:
        raise SystemExit("Too few usable utterances ({}). Check the dropped-reason counts above.".format(len(ok)))
    if len(ok) < 0.5 * len(done):
        print("WARNING: more than half of the utterances were dropped - see counts above.")

    summary = pre.finalize(ok, config["preprocessing"]["val_size"])
    # speaker description (useful for choosing a voice at inference time)
    info = {}
    for r in ok:
        d = info.setdefault(r["speaker"], {"key": r["speaker_key"], "n": 0})
        d["n"] += 1
    with open(os.path.join(out_dir, "speaker_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    summary["dropped_reasons"] = {k: v for k, v in reasons.items() if k != "ok"}
    summary["dataset"] = ds_cfg["name"]
    summary["wall_time_min"] = round((time.time() - t_start) / 60, 1)
    with open(os.path.join(out_dir, "preprocess_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--hf_token", default=None, help="HF access token (needed for the gated Svarah dataset)")
    parser.add_argument("--max_utts", type=int, default=0, help="only process the first N rows (smoke test)")
    parser.add_argument("--restart", action="store_true", help="delete previous output and start over")
    main(parser.parse_args())
