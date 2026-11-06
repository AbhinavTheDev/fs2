"""Builds Report/Technical_Report.pdf from the real artefacts (loss CSVs, plots, summaries).

Run AFTER training + inference for both languages, from the repo root:
    python Report/make_report.py
Numbers in the report are read from files produced by the pipeline, nothing is typed by hand.
Edit the TEXT blocks below if you want to add your own observations (listen to the audio first!).
"""
import csv, json, os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ss = getSampleStyleSheet()
H1, H2, B = ss["Heading1"], ss["Heading2"], ss["BodyText"]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def last_row(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None


def best_val(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return min(rows, key=lambda r: float(r["total"])) if rows else None


def img(path, width=16 * cm):
    if not os.path.exists(path):
        return Paragraph("<i>missing: %s</i>" % os.path.relpath(path, ROOT), B)
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=width, height=width * h / w)


story = [Paragraph("FastSpeech2 fine-tuning for Indian English and Telugu", ss["Title"]),
         Paragraph("Technical report", ss["Heading3"]), Spacer(1, 0.4 * cm)]

story += [Paragraph("1. Overview", H1), Paragraph(
    "Two separate FastSpeech2 models (ming024/FastSpeech2 implementation) were trained: an English model on "
    "ai4bharat/Svarah (Indian-accented English, 9.6 h, 117 speakers) and a Telugu model on "
    "SPRINGLab/IndicTTS_Telugu. Both are warm-started from the public LJSpeech FastSpeech2 checkpoint where tensor "
    "shapes match, and use the universal HiFi-GAN vocoder (22.05 kHz, 80 mel bins, hop 256).", B)]

story += [Paragraph("2. Method and modifications to the upstream code", H1), Paragraph(
    "<b>Alignment.</b> Upstream needs Montreal Forced Aligner (MFA) TextGrids; no Telugu model is available and MFA needs conda. "
    "We replaced it with a CTC forced aligner: a pretrained wav2vec2-CTC model (English: facebook/wav2vec2-base-960h, "
    "Telugu: anuragshas/wav2vec2-large-xlsr-53-telugu) gives character posteriors, a Viterbi pass forces the transcript, "
    "and character spans are converted to per-symbol durations in mel frames (pauses are assigned to the space symbol). "
    "<b>Input symbols.</b> Character level (letters for English, Unicode Telugu block U+0C00-0C7F for Telugu) instead of ARPAbet phonemes. "
    "<b>Other changes.</b> Mean pitch/energy for zero-duration symbols; fixed upstream bug where the pitch control scaled energy; "
    "CPU-safe STFT and torch.load; modern librosa API; losses logged to CSV and plotted; fp16 release checkpoint; "
    "multi-speaker embedding where a speaker is defined by the dataset's metadata columns (see README).", B)]

for lang in ("English", "Telugu"):
    d = os.path.join(ROOT, lang)
    pdir = os.path.join(d, "preprocessed_data")
    sub = os.listdir(pdir)[0] if os.path.isdir(pdir) and os.listdir(pdir) else None
    summ = jload(os.path.join(pdir, sub, "preprocess_summary.json")) if sub else {}
    logd = os.path.join(d, "checkpoints", "logs")
    ts = jload(os.path.join(logd, "training_summary.json"))
    tr, va, bv = last_row(os.path.join(logd, "train_loss.csv")), last_row(os.path.join(logd, "val_loss.csv")), best_val(os.path.join(logd, "val_loss.csv"))
    res = os.path.join(ROOT, "Results", lang)
    story += [PageBreak(), Paragraph("3. %s model" % lang, H1), Paragraph("3.1 Data", H2)]
    if summ:
        story.append(Paragraph("Usable utterances: %s (train %s / val %s), %s h, %s speaker ids. Dropped: %s." % (
            summ.get("n_utterances"), summ.get("n_train"), summ.get("n_val"), summ.get("hours"), summ.get("n_speakers"),
            json.dumps(summ.get("dropped_reasons", {}), ensure_ascii=False)), B))
    else:
        story.append(Paragraph("<i>preprocess_summary.json not found - copy %s/preprocessed_data/*/preprocess_summary.json back before building.</i>" % lang, B))
    story.append(Paragraph("3.2 Training", H2))
    rows = [["metric", "value"]]
    if ts:
        rows += [["final step", ts.get("final_step")], ["release checkpoint size (MB)", ts.get("release_size_mb")]]
    if tr:
        rows.append(["final training total loss (mean of last log window)", "%.4f" % float(tr["total"])])
    if va:
        rows.append(["final validation total loss", "%.4f" % float(va["total"])])
    if bv:
        rows.append(["best validation total loss (step)", "%.4f (%s)" % (float(bv["total"]), bv["step"])])
    t = Table(rows, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, "#999999"), ("BACKGROUND", (0, 0), (-1, 0), "#e8e8e8")]))
    story += [t, Spacer(1, 0.3 * cm), img(os.path.join(res, "training_loss.png")), Spacer(1, 0.2 * cm), img(os.path.join(res, "validation_loss.png"))]
    story += [PageBreak(), Paragraph("3.3 Variance adaptor outputs (predicted duration, F0, energy)", H2)]
    for k in range(1, 6):
        for kind in ("duration", "pitch", "energy"):
            story.append(img(os.path.join(res, "variance_adaptor_plots", kind, "sentence_%d.png" % k), 13 * cm))
        story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Observations:</b> (write your own after listening to Results/%s/test_generated_audios)" % lang, B))

story += [PageBreak(), Paragraph("4. Limitations", H1), Paragraph(
    "Character-level input without a pronunciation lexicon limits English quality on irregular spellings; Svarah is a 9.6 h test-set style corpus recorded "
    "in varied conditions, so the voice is noisier than a studio TTS corpus; alignments come from CTC (peaky, ~20 ms resolution) rather than MFA; "
    "training was short (tens of thousands of steps vs 900k upstream).", B)]

SimpleDocTemplate(os.path.join(ROOT, "Report", "Technical_Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(story)
print("wrote Report/Technical_Report.pdf")
