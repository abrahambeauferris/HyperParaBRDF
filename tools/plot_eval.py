#!/usr/bin/env python3
"""Publication‑ready plotting utility for HyperBRDF experiments.

Additions in this version (v3)
-----------------------------
* **Training‑loss figure**
  * Plotted on a **log‑scaled y‑axis** to tame extreme first‑epoch spikes.
  * Includes an **inset zoom** (last 25 % of epochs, linear scale) so the
    plateau phase is clearly visible.
  * Minor grid lines + colour‑blind‑safe line colours = improved readability.
* **Bar chart** already uses colour‑blind palette & hatched fills; legend is a
  compact two‑column row above the axes.

Run the script the same way as before; it now writes:
  * `train_loss_all.{svg,pdf}`      – full log‑scale with zoom inset
  * `rmse_mae_bar.{svg,pdf}`       – error comparison chart
"""
from __future__ import annotations

import argparse
import glob
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
})

CB_BLUE = "#377eb8"  # colour‑blind safe palette Set‑1
CB_GREEN = "#4daf4a"
CB_PALETTE = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", comment="=", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.rename(columns={k: "test" for k in ("test_size", "split", "ts") if k in df.columns})

    required = {"test", "rmse", "mae"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(df.columns))}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rmse", "mae"]).sort_values("test").reset_index(drop=True)
    return df


def extract_test_tag(path: str | Path) -> str:
    m = re.search(r"testsize_(\d+(?:\.\d+)?)", str(path))
    return m.group(1) if m else "?"


def load_loss(csv_path: str | Path) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None, engine="python")
    if df.shape[1] == 1:
        col = df.iloc[:, 0]
    else:
        col0, col1 = df.iloc[:, 0], df.iloc[:, 1]
        is_range0 = np.array_equal(col0, np.arange(len(col0)))
        is_range1 = np.array_equal(col1, np.arange(len(col1)))
        col = col1 if is_range0 and not is_range1 else col0 if is_range1 and not is_range0 else col1
    col = pd.to_numeric(col, errors="coerce").dropna()
    return col.to_numpy()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--loss_csv", required=True, nargs="+")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------- 1) Training‑loss curves ---------------------------
    curves_raw: list[tuple[str, np.ndarray]] = []
    for pattern in args.loss_csv:
        for csv_path in sorted(glob.glob(pattern)):
            curves_raw.append((extract_test_tag(csv_path), load_loss(csv_path)))

    if curves_raw:  # only plot if we have data
        # repeat palette as needed so each curve keeps the SAME colour in inset
        palette = (CB_PALETTE * ((len(curves_raw) + len(CB_PALETTE) - 1) // len(CB_PALETTE)))[: len(curves_raw)]
        curves = [(tag, y, col) for (tag, y), col in zip(curves_raw, palette)]

        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        for tag, y, col in curves:
            x = np.arange(len(y))
            ax.plot(x, y, label=f"test={tag}", lw=1, color=col)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (log)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both", linestyle=":", linewidth=0.4)

                # --------------- single‑row legend above plot ---------------
        ncols = len(curves)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.05),
            ncol=ncols, frameon=False, fontsize=8,
            handlelength=1.6, columnspacing=0.8, handletextpad=0.4,
            borderaxespad=0.0,
        )

        # -------- inset zoom (last 25 % of epochs, linear scale) --------
        max_len = max(len(y) for _, y, _ in curves)
        zoom_start = int(max_len * 0.75)
        zoom_ax = inset_axes(ax, width="50%", height="50%", loc="upper right", borderpad=1)
        for tag, y, col in curves:
            x = np.arange(len(y))[zoom_start:]
            zoom_ax.plot(x, y[zoom_start:], lw=1, color=col)
        zoom_ax.set_title("final 25 %", fontsize=7, pad=2)
        zoom_ax.set_xlabel("epoch", fontsize=7)
        zoom_ax.set_ylabel("loss", fontsize=7)
        zoom_ax.tick_params(labelsize=6)
        zoom_ax.grid(alpha=0.3, linestyle=":", linewidth=0.3)

        # reserve space for the legend row
        , pad=0.2)
        for ext in ("svg", "pdf"):
            fig.savefig(outdir / f"train_loss_all.{ext}")
        plt.close(fig)

    # --------------------- 2) RMSE / MAE bar chart ---------------------------) RMSE / MAE bar chart ---------------------------
    df = parse_summary(args.summary)
    width = 0.35
    x_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.bar(x_pos - width / 2, df["rmse"], width, label="RMSE", color=CB_BLUE, hatch="///")
    ax.bar(x_pos + width / 2, df["mae"],  width, label="MAE",  color=CB_GREEN, hatch="\\\\\\")

    ax.legend(ncol=2, frameon=False, fontsize=8,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0.0, handletextpad=0.4, columnspacing=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["test"].astype(str))
    ax.set_xlabel("Test split")
    ax.set_ylabel("Error")
    
    for ext in ("svg", "pdf"):
        fig.savefig(outdir / f"rmse_mae_bar.{ext}")
    plt.close(fig)

    print(f"✓ Plots written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
