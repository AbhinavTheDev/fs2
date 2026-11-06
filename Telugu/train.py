"""Training / fine-tuning script (modified from upstream FastSpeech2 train.py).

What is different from upstream
  * one config.yaml per language (sections: preprocess / model / train / inference);
  * `--pretrained <ckpt>` warm-starts from an existing FastSpeech2 checkpoint (fine-tuning);
  * average training loss of every logging window and every validation loss are written to CSV,
    and `training_loss.png` / `validation_loss.png` are produced for the report;
  * a small model-only fp16 checkpoint is exported at the end (fits GitHub's 100 MB limit);
  * `--restore_step N` resumes a run, `--plot_only` only regenerates the plots from the logs.

Run from inside the language folder, e.g.
    python train.py -c config.yaml --pretrained /path/to/LJSpeech/900000.pth.tar
"""
import argparse
import csv
import json
import os
import time

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import Dataset
from evaluate import evaluate, LOSS_NAMES
from model import FastSpeech2Loss
from utils.model import get_model, get_vocoder, get_param_num, export_for_release
from utils.tools import to_device, log, synth_one_sample, load_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def plot_losses(log_dir, results_dir, language):
    """Create training_loss.png and validation_loss.png from the CSV logs."""
    os.makedirs(results_dir, exist_ok=True)
    out = {}
    for split, fname in (("training", "train_loss.csv"), ("validation", "val_loss.csv")):
        path = os.path.join(log_dir, fname)
        if not os.path.exists(path):
            print("No", path, "- skipping", split, "plot")
            continue
        d = read_csv(path)
        if not d:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(d["step"], d["total"], color="tab:blue", lw=1.8)
        ax[0].set_title("{} - total {} loss".format(language, split))
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("loss")
        ax[0].grid(alpha=0.3)
        for name, color in zip(LOSS_NAMES[1:], ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]):
            ax[1].plot(d["step"], d[name], label=name, lw=1.4, color=color)
        ax[1].set_title("{} - loss components ({})".format(language, split))
        ax[1].set_xlabel("step")
        ax[1].set_yscale("log")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        target = os.path.join(results_dir, "{}_loss.png".format(split))
        fig.savefig(target, dpi=150)
        plt.close(fig)
        out[split] = {"final": d["total"][-1], "min": min(d["total"]), "step_of_min": d["step"][d["total"].index(min(d["total"]))]}
        print("Saved", target)
    return out


# ----------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------
def main(args, configs):
    preprocess_config, model_config, train_config = configs
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    log_dir = train_config["path"]["log_path"]

    if args.plot_only:
        plot_losses(log_dir, train_config["path"]["results_path"], train_config["language"])
        return

    print("Prepare training ...  device:", device)
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # larger than 1 enables length-sorted batches inside a group
    assert batch_size * group_size < len(dataset), "dataset too small for this batch size"
    loader = DataLoader(
        dataset, batch_size=batch_size * group_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 parameters:", get_param_num(model))

    try:
        vocoder = get_vocoder(model_config, device)
    except Exception as err:  # training itself does not need the vocoder
        print("WARNING: vocoder unavailable ({}); audio will not be logged during training.".format(err))
        vocoder = None

    try:
        from torch.utils.tensorboard import SummaryWriter

        train_logger = SummaryWriter(os.path.join(log_dir, "tb", "train"))
        val_logger = SummaryWriter(os.path.join(log_dir, "tb", "val"))
    except Exception:
        train_logger = val_logger = None

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    def open_csv(name):
        path = os.path.join(log_dir, name)
        new = not os.path.exists(path) or args.restore_step == 0
        if not new:
            # resuming: drop rows logged after the checkpoint we restart from (avoids duplicate steps)
            with open(path, newline="") as fin:
                kept = [r for r in csv.reader(fin) if r and (r[0] == "step" or float(r[0]) <= args.restore_step)]
            with open(path, "w", newline="") as fout:
                csv.writer(fout).writerows(kept)
        f = open(path, "w" if new else "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + LOSS_NAMES)
        return f, w

    train_f, train_csv = open_csv("train_loss.csv")
    val_f, val_csv = open_csv("val_loss.csv")

    window = [0.0] * 6
    n_window = 0
    t_begin = time.time()
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    def save_full(step_):
        path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step_))
        torch.save({"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict()}, path)
        return path

    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                total_loss = losses[0] / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                vals = [l.item() for l in losses]
                window = [a + b for a, b in zip(window, vals)]
                n_window += 1

                if step % log_step == 0:
                    avg = [w / n_window for w in window]
                    window, n_window = [0.0] * 6, 0
                    train_csv.writerow([step] + ["{:.6f}".format(a) for a in avg])
                    train_f.flush()
                    outer_bar.write(
                        "Step {}/{}, Total {:.4f}, Mel {:.4f}, PostNet {:.4f}, Pitch {:.4f}, Energy {:.4f}, Duration {:.4f}".format(
                            step, total_step, *avg
                        )
                    )
                    if train_logger is not None:
                        log(train_logger, step, losses=avg)

                if step % synth_step == 0 and vocoder is not None and train_logger is not None:
                    fig, wav_rec, wav_pred, tag = synth_one_sample(batch, output, vocoder, model_config, preprocess_config)
                    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                    log(train_logger, audio=wav_pred, sampling_rate=sr, tag="Training/step_{}_{}_synthesized".format(step, tag))

                if step % val_step == 0:
                    model.eval()
                    message, means = evaluate(model, step, configs, val_logger, vocoder if train_logger else None)
                    val_csv.writerow([step] + ["{:.6f}".format(m) for m in means])
                    val_f.flush()
                    outer_bar.write(message)
                    model.train()

                if step % save_step == 0:
                    save_full(step)

                if step >= total_step:
                    final_full = save_full(step)
                    train_f.close()
                    val_f.close()
                    finish(final_full, step, time.time() - t_begin, train_config, preprocess_config)
                    return
                step += 1
                outer_bar.update(1)
        epoch += 1


def finish(final_full, step, seconds, train_config, preprocess_config):
    """Export the release checkpoint, draw the loss plots, write a small training summary."""
    ckpt_dir = os.path.dirname(os.path.abspath(final_full))
    release = os.path.join(train_config["path"]["release_path"], train_config["export_name"])
    os.makedirs(os.path.dirname(release), exist_ok=True)
    size = export_for_release(final_full, release, extra={"step": step, "language": train_config["language"]})
    print("Exported release checkpoint {} ({:.1f} MB)".format(release, size))
    stats = plot_losses(train_config["path"]["log_path"], train_config["path"]["results_path"], train_config["language"])
    summary = {"language": train_config["language"], "final_step": step, "train_minutes_this_run": round(seconds / 60, 1),
               "release_checkpoint": release, "release_size_mb": round(size, 1), "loss_curves": stats}
    with open(os.path.join(train_config["path"]["log_path"], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=0, help="resume from <ckpt_path>/<step>.pth.tar")
    parser.add_argument("--pretrained", type=str, default=None, help="warm-start from this FastSpeech2 checkpoint")
    parser.add_argument("--ckpt_path", type=str, default=None, help="override where full checkpoints are written (e.g. Google Drive)")
    parser.add_argument("--total_step", type=int, default=None, help="override train.step.total_step")
    parser.add_argument("--plot_only", action="store_true", help="only regenerate the loss plots from the CSV logs")
    args = parser.parse_args()
    args.checkpoint = None

    p, m, t, _ = load_configs(args.config)
    if args.ckpt_path:
        t["path"]["ckpt_path"] = args.ckpt_path
    if args.total_step:
        t["step"]["total_step"] = args.total_step
    main(args, (p, m, t))
