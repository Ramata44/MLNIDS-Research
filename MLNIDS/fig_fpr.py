# make_fpr_figure.py -- the false-alarm figure of the JISA manuscript
# (Figure 2 in the compiled PDF; the file is named by content, not by number,
# because float placement decides the numbering).
#
# False-alarm cost of PGD adversarial training. For each dataset the panel plots
# the false-positive rate against the detection rate (TPR) at which it is
# measured, for the standard and the adversarially-trained detectors. The three
# joined points per curve are the fixed operating points TPR = 0.90, 0.95, 0.99,
# so the two detectors are compared at matched detection. The isolated star marks
# where each detector actually sits when the decision threshold is left at 0.5.
#
# Reading the figure: a curve lying LOWER raises fewer false alarms at the same
# detection rate. A star far ABOVE its own curve means the default threshold is
# badly calibrated for that detector; a star far to the LEFT means the detector
# is quiet only because it is missing attacks.
#
# Data: results_fpr_cic.csv and results_fpr_unsw.csv, written by fpr_only.py and
# fpr_only_unsw.py. Points are means over the three seeds and three activations;
# error bars are one population standard deviation over the three seeds of the
# activation-averaged value.
#
# Usage, in the folder that contains the two CSVs:
#     python make_fpr_figure.py
# Output: fpr_figure.pdf (vector, for the LaTeX source) and fpr_figure.png (dpi = 1600).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPI = 1600
TPRS = [0.90, 0.95, 0.99]

# two series, same palette as Figure 1: blue = undefended, red = defended
SERIES = {
    "standard": ("Standard",  (0.231, 0.435, 0.690)),
    "pgd_at":   ("PGD-AT",    (0.753, 0.314, 0.302)),
}

DATASETS = [
    ("results_fpr_cic.csv",  "CSE-CIC-IDS2018"),
    ("results_fpr_unsw.csv", "UNSW-NB15"),
]


def summarise(path):
    """Per defence: activation-averaged mean and seed SD of each quantity."""
    d = pd.read_csv(path)
    cols = [f"fpr_at_tpr{int(t*100)}" for t in TPRS] + ["fpr_at_0.5", "tpr_at_0.5"]
    per_seed = d.groupby(["defence", "seed"])[cols].mean()      # average activations
    mean = per_seed.groupby("defence").mean()
    std = per_seed.groupby("defence").std(ddof=0)
    return mean, std


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for ax, (path, title) in zip(axes, DATASETS):
        mean, std = summarise(path)
        for defence, (label, colour) in SERIES.items():
            y = [mean.loc[defence, f"fpr_at_tpr{int(t*100)}"] for t in TPRS]
            e = [std.loc[defence, f"fpr_at_tpr{int(t*100)}"] for t in TPRS]
            ax.errorbar(TPRS, y, yerr=e, color=colour, marker="o", markersize=5,
                        linewidth=2, capsize=3, label=label, zorder=3)
            # operating point the detector actually uses (threshold 0.5)
            ax.plot(mean.loc[defence, "tpr_at_0.5"], mean.loc[defence, "fpr_at_0.5"],
                    marker="*", markersize=13, color=colour, markeredgecolor="white",
                    markeredgewidth=0.8, linestyle="none", zorder=4)

            print(f"{title:16s} {label:8s} "
                  + "  ".join(f"FPR@{int(t*100)}={mean.loc[defence, f'fpr_at_tpr{int(t*100)}']:.5f}"
                              for t in TPRS)
                  + f"  | at 0.5: TPR={mean.loc[defence, 'tpr_at_0.5']:.4f}"
                    f" FPR={mean.loc[defence, 'fpr_at_0.5']:.5f}")

        ax.set_yscale("log")
        ax.set_xlabel("Detection rate (TPR)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, which="major", zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("False-positive rate (log scale)")

    # legend: the two detectors, plus one neutral entry for the threshold-0.5 marker
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(SERIES["standard"][0]), labels.index(SERIES["pgd_at"][0])]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    star = Line2D([], [], marker="*", markersize=11, linestyle="none",
                  color="0.35", markeredgecolor="white", markeredgewidth=0.8)
    handles.append(star); labels.append("operating point at threshold 0.5")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig("fpr_figure.pdf", dpi=DPI)
    fig.savefig("fpr_figure.png", dpi=DPI)
    print("Saved fpr_figure.pdf and fpr_figure.png")


if __name__ == "__main__":
    main()
