# make_fig2.py -- Figure 2 of the JISA manuscript.
#
# Mean ASR at eps = 0.3 for the four primary attacks (PGD, BIM, HopSkipJump, SPSA)
# against the depth-4, 30-feature model with ReLU, ELU and Tanh activations, on
# CSE-CIC-IDS2018 (left) and UNSW-NB15 (right). Error bars: one population standard
# deviation over the three seeds (1, 7, 42).
#
# Data: results_extra_cic_all.csv and results_extra_unsw_all.csv, written by
# attack1_cic.py / attack1_unsw.py. Rows with level == 0 (unconstrained) and
# eps == 0.3 are used. HopSkipJump is the PGD-initialised variant (attack name
# "hsj_bounded"), so every bar equals its cell in Table 4 of the paper.
#
# Usage, in the folder that contains the two CSVs:
#     python make_fig2.py
# Output: fig2.pdf (vector, for the LaTeX source) and fig2.png (dpi = 1600).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 1600
EPS = 0.3
LEVEL = 0
ACTS = ["relu", "elu", "tanh"]
ACT_LABELS = ["RELU", "ELU", "TANH"]

# attack column name in the CSV -> (legend label, bar colour)
ATTACKS = {
    "pgd":         ("PGD (WB)",  (0.231, 0.435, 0.690)),
    "bim":         ("BIM (WB)",  (0.435, 0.639, 0.847)),
    "hsj_bounded": ("HSJ (BB)",  (0.753, 0.314, 0.302)),
    "spsa":        ("SPSA (BB)", (0.886, 0.569, 0.373)),
}

DATASETS = [
    ("results_extra_cic_all.csv",  "CSE-CIC-IDS2018"),
    ("results_extra_unsw_all.csv", "UNSW-NB15"),
]


def summarise(path):
    d = pd.read_csv(path)
    d = d[(d["level"] == LEVEL) & (d["eps"] == EPS) & (d["attack"].isin(ATTACKS))]
    g = d.groupby(["activation", "attack"])["asr"]
    mean = g.mean()
    std = g.std(ddof=0)          # population SD over the three seeds, as in the tables
    return mean, std


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    width = 0.2
    x = np.arange(len(ACTS))

    for ax, (path, title) in zip(axes, DATASETS):
        mean, std = summarise(path)
        for i, (attack, (label, colour)) in enumerate(ATTACKS.items()):
            m = [mean[(a, attack)] for a in ACTS]
            s = [std[(a, attack)] for a in ACTS]
            ax.bar(x + (i - 1.5) * width, m, width, yerr=s, capsize=3,
                   color=colour, label=label, error_kw=dict(lw=1))
        ax.set_xticks(x)
        ax.set_xticklabels(ACT_LABELS)
        ax.set_title(title)
        ax.set_xlabel("Activation")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # print the plotted numbers so they can be checked against Table 4
        print(f"--- {title} (eps = {EPS}, level {LEVEL}) ---")
        for a in ACTS:
            row = "  ".join(f"{attack}={mean[(a, attack)]:.3f}+/-{std[(a, attack)]:.3f}"
                            for attack in ATTACKS)
            print(f"{a:5s} {row}")

    axes[0].set_ylabel(r"Mean ASR at $\epsilon = 0.3$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("fig2.pdf", dpi=DPI)
    fig.savefig("fig2.png", dpi=DPI)
    print("Saved fig2.pdf and fig2.png")


if __name__ == "__main__":
    main()
