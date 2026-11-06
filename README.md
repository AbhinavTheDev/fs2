# FastSpeech2 fine-tuning: Indian-accented English (Svarah) and Telugu (IndicTTS)

Two **separate** FastSpeech2 text-to-speech models, built on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2):

| | English | Telugu |
|---|---|---|
| Dataset | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (9.6 h, 117 speakers, gated) | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (8.58k clips) |
| Input symbols | characters (a-z, punctuation) | Telugu Unicode characters |
| Aligner (durations) | wav2vec2-base-960h CTC | wav2vec2-large-xlsr-53-telugu CTC |
| Config | `English/config.yaml` | `Telugu/config.yaml` |

```
project/
├── English/   train.py  preprocess.py  inference.py  config.yaml  checkpoints/   (+ model/ transformer/ audio/ text/ utils/ hifigan/ dataset.py evaluate.py from upstream, modified)
├── Telugu/    same layout
├── Results/   English|Telugu / training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/
├── Report/    make_report.py  ->  Technical_Report.pdf
├── Colab_Runbook.ipynb     step-by-step GPU notebook (run once per language)
├── requirements.txt
└── README.md
```

## Status of this repository (read this first)

The code was written and **smoke-tested end to end on CPU** (synthetic data, stubbed aligner): preprocessing, training, validation, resume, warm-start, checkpoint export, loss plots, inference (audio + all three variance plots) and report generation all run.
**It has not been trained on the real datasets**, because that needs a GPU and Hugging Face access. So `Results/`, `*/checkpoints/*.pth.tar` and `Report/Technical_Report.pdf` are produced by *you* following the steps below. Nothing in them is pre-made, so nothing is faked.

## What you need to do (about 4-6 hours total, mostly waiting)

1. **Accounts / access**
   - Make a Hugging Face account, open the [Svarah page](https://huggingface.co/datasets/ai4bharat/Svarah), accept its terms, and create a read token at huggingface.co/settings/tokens. (Telugu needs no token.)
   - Put this repo on GitHub (your account), keep the folder structure as is.
2. **Pretrained weights (recommended, 5 min).** Download `900000.pth.tar` (LJSpeech) from the upstream [pretrained models folder](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F) and upload it to your Google Drive. It gives a much better start than random weights. Without it, training still works but needs more steps.
3. **Open `Colab_Runbook.ipynb` in Google Colab** (T4 GPU), set `LANG="English"`, your repo URL and token, and run all cells. Then repeat with `LANG="Telugu"`. Kaggle notebooks (free P100/T4, 30 h/week) work the same way if Colab runs out.
4. **Check the preprocessing summary** before training (cell 6). Healthy: most utterances kept, `mean_align_score` above about -0.8. If most clips are dropped, see Troubleshooting.
5. **Download `<LANG>_outputs.zip`** (last cell) and unzip it into your repo root: it drops the checkpoint into `<LANG>/checkpoints/` and the plots/audio into `Results/<LANG>/`.
6. **Listen to the 5 audio files per language**, then `python Report/make_report.py` to build `Report/Technical_Report.pdf` from the real logs; add your own observations to it.
7. `git add . && git commit && git push`. The release checkpoints are ~72 MB (fp16), under GitHub's 100 MB limit, so plain git works.

Test sentences are in `config.yaml -> inference.test_sentences`. **If your assignment gives specific five sentences, replace mine** and re-run `python inference.py -c config.yaml` (CPU is fine, ~1 min).

## Commands (what the notebook runs)

```bash
pip install -r requirements.txt
cd English                                    # or Telugu
python preprocess.py -c config.yaml --hf_token $HF_TOKEN       # alignment + features (GPU helps)
python train.py -c config.yaml --pretrained LJSpeech_900000.pth.tar
python train.py -c config.yaml --restore_step 7500             # resume after a disconnect
python inference.py -c config.yaml                             # 5 sentences -> ../Results/<LANG>/
python inference.py -c config.yaml --text "Any sentence"       # ad-hoc synthesis
```

## What was modified from upstream (constraint 2)

- **Alignment.** Upstream needs Montreal Forced Aligner TextGrids (conda only, no Telugu model). `preprocess.py` replaces it with a pip-only **CTC forced aligner** (wav2vec2 + Viterbi, unit-tested) that produces per-symbol durations. Pauses go to the space symbol.
- **Text front end.** Upstream's English uses ARPAbet phonemes from a lexicon; both languages here use **character-level** input (`text/symbols.py`, `text/cleaners.py`, `text/__init__.py`). Telugu is nearly phonemic, so this is reasonable; for English it is a simplification (see Limitations).
- **One `config.yaml` per language** instead of three YAMLs (sections `preprocess`, `model`, `train`, `inference`).
- **`train.py`**: warm start (`--pretrained`), resume, CSV loss logs, loss plots, fp16 release checkpoint. **`evaluate.py`** returns numeric losses.
- **`inference.py`**: new; synthesizes text and draws duration / pitch / energy plots per sentence.
- **`utils/model.py`**: CPU-safe `torch.load`, vocoder weights auto-downloaded, partial-weight transfer (embeddings that don't fit are re-initialised).
- **`model/modules.py`**: fixed an upstream bug (pitch control was applied to the energy embedding). **`audio/*`**: removed hard-coded `.cuda()`, ported to current librosa.
- Zero-duration symbols get the utterance-mean pitch/energy (upstream used 0).

## Design decisions worth knowing (you may be asked about these)

- **Speaker ids.** Neither dataset has a speaker-id column. Svarah: a speaker key is built from the demographic columns (gender, age group, native place, qualification, job) which identifies a speaker practically uniquely; Telugu: the `gender` label (so at most 2 "voices"). Check `preprocessed_data/*/speaker_info.json` after preprocessing. If the speaker count looks wrong, edit `speaker_columns` in `config.yaml`.
- **Audio rate / vocoder.** All audio is resampled to 22.05 kHz to match the pretrained HiFi-GAN "universal" vocoder, so the vocoder is *not* fine-tuned; only FastSpeech2 is.
- **Steps.** English 20k, Telugu 30k steps (batch 16) as a deadline-friendly budget; upstream uses 900k from scratch. Raise `train.step.total_step` if time allows, use `--total_step N` to override.
- **Val split.** 200 (English) / 300 (Telugu) utterances. Note random utterance-level splitting means the same speaker appears in train and val.

## Learning resources (for someone new to this)

- FastSpeech 2 paper: [arxiv.org/abs/2006.04558](https://arxiv.org/abs/2006.04558). Read sections 2-3 (variance adaptor: duration, pitch, energy predictors).
- FastSpeech 1 paper (duration/length regulator idea): [arxiv.org/abs/1905.09263](https://arxiv.org/abs/1905.09263)
- HiFi-GAN vocoder (turns mel-spectrograms into audio): [arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646)
- Upstream code: [github.com/ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- Hugging Face Audio course, TTS unit: [huggingface.co/learn/audio-course](https://huggingface.co/learn/audio-course)
- Forced alignment with CTC (the idea behind our aligner): [PyTorch tutorial](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- Mel-spectrograms explained: search "mel spectrogram explained" on YouTube (e.g. Valerio Velardo's Audio Signal Processing series).

Mental model: **text → encoder → (duration, pitch, energy predictors expand and decorate the sequence) → decoder → mel-spectrogram → HiFi-GAN → waveform.** "Fine-tuning" here means starting from weights that already know how speech works and continuing training on your data.

## Troubleshooting

- **Svarah 401/403 error**: you haven't accepted the terms on the dataset page, or the token is missing/wrong.
- **Most Telugu utterances dropped** (`too_long`, `too_long_after_trim`, `unsupported_chars`): IndicTTS clips can be long and contain digits/Latin letters. Check the counts in the summary; raise `filters.max_raw_seconds` only together with `max_mel_frames` and `model.max_seq_len` (memory grows). Utterances with Latin letters/digits are dropped on purpose, since we cannot pronounce them faithfully.
- **Many `bad_alignment`**: lower `min_align_score` slightly (for example -1.5), or inspect a few clips; the Svarah audio is real-world quality.
- **Colab disconnects**: checkpoints are on Drive every 2500 steps; use the resume cell. The loss CSVs are kept in `checkpoints/logs/`.
- **CUDA out of memory**: lower `train.optimizer.batch_size` to 8.
- **Telugu plot labels look romanised / boxes**: `apt-get install -y fonts-noto-core` and re-run inference. Individual vowel signs may show with a dotted circle, since each symbol is labelled separately.
- **Robotic or mumbling audio** after a short run is expected; check that validation loss is still going down, and train longer.

## Limitations (also stated in the report)

Character-level English cannot learn irregular spellings as well as phonemes would; Svarah is a ~9.6 h test-style corpus recorded in varied conditions (noisier than a studio TTS corpus); the CTC aligner has ~20 ms resolution and is less precise than MFA; the short training budget gives intelligible-but-imperfect voices. The universal vocoder is not adapted to either dataset.

Upstream code is MIT-licensed (© Chung-Ming Chien); HiFi-GAN is MIT-licensed. Svarah is CC BY 4.0; check the IndicTTS dataset card for its license.
