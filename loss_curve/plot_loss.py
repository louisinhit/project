"""Plot recorded measurements only; no interpolated or fabricated observations."""

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(ROOT / "_runtime" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "_runtime" / "cache")


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_run(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("All plot outputs must remain inside test/.")
    steps = read_rows(run_dir / "loss" / "steps.csv")
    epochs = read_rows(run_dir / "loss" / "epochs.csv")
    if not steps:
        return
    output = run_dir / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), constrained_layout=True)
    for ax, rows, title in ((axes[0], steps, "Joint Training"),
                             (axes[1], [r for r in steps if int(r["epoch"]) <= 10], "First 10 Epochs")):
        ax.plot([int(r["global_step"]) for r in rows],
                [float(r["loss_mean"]) for r in rows], color="#2474a6", linewidth=1)
        ax.set(xlabel="Optimizer step", ylabel="Mean cross-entropy loss", title=title)
        ax.grid(alpha=0.2)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"training_steps.{extension}", dpi=200)
    plt.close(fig)
    if epochs:
        fig, ax = plt.subplots(figsize=(6.1, 4.1), constrained_layout=True)
        x = [int(r["epoch"]) for r in epochs]
        ax.plot(x, [float(r["train_loss"]) for r in epochs], label="Training (random length)", color="#2474a6")
        ax.plot(x, [float(r["monitor_loss"]) for r in epochs], label="Held-out 2022 (32768)", color="#b74c40")
        ax.set(xlabel="Epoch", ylabel="Mean cross-entropy loss")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"epoch_loss.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_run(parser.parse_args().run_dir)
