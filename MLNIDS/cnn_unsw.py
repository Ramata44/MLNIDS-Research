# cnn_xiao_unsw.py -- UNSW-NB15 ONLY. Identical to cnn_xiao.py with the dataset
# hard-set, so there is no argument to get wrong.
#
# Does the realizability gap hold for a convolutional detector?
#
# WHY THIS RUN EXISTS
#
# Every network in the paper is a fully-connected MLP. A reviewer will ask whether
# the central result -- that only a small fraction of the attack success reported
# by the unconstrained protocol survives a realizable (L2) adversary -- is a
# property of ML-NIDS or merely of MLPs. This script answers that for one
# convolutional detector, at one configuration, through the experiments that
# carry the paper's central claims. It does NOT rerun the depth sweep, the
# feature sweep, the full HopSkipJump comparison, adversarial training or the FPR
# analysis.
#
# THE MODEL. This is not a reproduction of any published system. It is a CNN
# whose layer structure follows the CNN-IDS design of Xiao et al. ("An Intrusion
# Detection Model Based on Feature Reduction and Convolutional Neural Networks",
# IEEE Access 7, 2019), a widely cited flow-based NIDS CNN. The feature vector is
# reshaped into a small 2-D grid and passed through:
#
#   Conv 2x2 (8 filters), same padding, BatchNorm + ReLU
#   MaxPool 2x2, stride 1
#   Conv 2x2 (16 filters), same padding, ReLU
#   MaxPool 2x2, stride 1, Dropout 0.3
#   Dense 64, ReLU, Dropout 0.3
#   Dense 2 (softmax in the loss)
#
# The 30 selected features are arranged as a 5x6 grid in ANOVA-rank order.
#
# EVERYTHING EXCEPT THE ARCHITECTURE IS THIS PAPER'S OWN, so that the only
# difference from the MLP results is the architecture: the same data loading and
# preprocessing, the fixed split/scaler/pool at seed 42, the same training
# protocol (Adam, lr 1e-3, batch 16,384, early stopping to convergence, PyTorch
# default initialisation, as for the MLPs), the same capability masks, and the
# same four constrained-capable attacks and worst-case ASR -- all imported
# unchanged from your own scripts via capability_unsw.py. The
# transfer attack keeps its MLP substitute, so for the CNN it is a
# cross-architecture transfer.
#
# Usage, in the folder that contains capability_unsw.py:
#     python cnn_xiao_unsw.py
#
# Output: results_cnn_<ds>.csv, one row per (seed, level, eps). RESUMES.

import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------- CONFIG
DS = "unsw"                              # hard-set: this file is UNSW-NB15 only
FEATS  = 30
GRID   = (5, 6)                          # 5 x 6 = 30 features
SEEDS  = [1, 7, 42]
LEVELS = [0, 1, 2]
EPS    = [0.1, 0.3, 0.5]
P_DROP = 0.3
OUT    = f"results_cnn_{DS}.csv"
# -------------------------------------------------------------------------

import capability_unsw as A
G = A.G
DEVICE = G.DEVICE


class FlowCNN(nn.Module):
    """CNN with the layer structure of Xiao et al. (2019). Takes a flat feature
    vector (N, H*W) -- the form every attack in the pipeline passes -- and
    reshapes it to (N, 1, H, W) internally."""

    def __init__(self, grid=GRID, n_classes=2, p_drop=P_DROP):
        super().__init__()
        self.h, self.w = grid
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=1, padding="same")
        self.bn1   = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop2 = nn.Dropout(p_drop)
        self.fc1   = nn.Linear(16 * (self.h - 2) * (self.w - 2), 64)
        self.drop3 = nn.Dropout(p_drop)
        self.fc2   = nn.Linear(64, n_classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, self.h, self.w)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.drop2(self.pool2(x))
        x = self.drop3(self.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.seed, d.level, d.eps))


def main():
    print(f"dataset={DS}  device={DEVICE}  grid={GRID}  levels={LEVELS}  "
          f"eps={EPS}", flush=True)

    Xtr_full, ytr, Xte_full, yte, order = G.load_unsw()
    k = min(FEATS, Xtr_full.shape[1]); ok = order[:k]
    assert k == GRID[0] * GRID[1], f"{k} features do not fill a {GRID} grid"
    Xtr, Xte = Xtr_full[:, ok], Xte_full[:, ok]

    rng = np.random.default_rng(G.SEED)
    pool = rng.choice(np.where(yte != 0)[0],
                      size=min(G.N_EVAL, int((yte != 0).sum())), replace=False)
    Xe, ye = Xte_full[pool][:, ok], yte[pool]

    names = A.load_feature_names(G, DS, order, k)
    masks = {lv: (None if lv == 0 else A.build_mask(names, DS, lv)) for lv in LEVELS}
    for lv in LEVELS:
        n = k if masks[lv] is None else int(masks[lv].sum())
        print(f"  L{lv}: {n} of {k} features perturbable", flush=True)

    have = done_keys(OUT)
    t0 = time.time()
    for seed in SEEDS:
        todo = [(lv, e) for lv in LEVELS for e in EPS if (seed, lv, e) not in have]
        if not todo:
            print(f"skip s{seed}: done", flush=True)
            continue

        torch.manual_seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        Xtrn, Xval, ytrn, yval = train_test_split(
            Xtr, ytr, test_size=0.1,
            random_state=(G.SEED if A.MATCH_PUBLISHED_SPLIT else seed),
            stratify=ytr)

        torch.manual_seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        t1 = time.time()
        model = FlowCNN().to(DEVICE)
        G.train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
        model.eval()
        f1 = G.macro_f1(model, Xte, yte, DEVICE)
        p = G._probs(model, Xte, DEVICE)[:, 1]
        auc = float(roc_auc_score((yte != 0).astype(int), p))
        print(f"  CNN s{seed}: trained in {time.time()-t1:.0f}s  "
              f"clean macro-F1={f1:.4f}  ROC-AUC={auc:.5f}", flush=True)

        clf = G._art_clf(model, Xtrn, DEVICE)
        for lv, e in todo:
            t2 = time.time()
            a = A.worst_case(model, clf, Xtrn, ytrn, Xval, yval, k,
                             Xe, ye, e, seed, masks[lv])
            row = {"model": "cnn", "seed": seed, "level": lv, "eps": e,
                   "features": k, "worst_asr": round(float(a), 4),
                   "clean_f1": round(float(f1), 4), "roc_auc": round(auc, 5)}
            pd.DataFrame([row]).to_csv(OUT, mode="a", index=False,
                                       header=not os.path.exists(OUT))
            have.add((seed, lv, e))
            print(f"      L{lv} eps={e}: worst-case ASR={a:.4f} "
                  f"({time.time()-t2:.0f}s)", flush=True)

    d = pd.read_csv(OUT)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\n=== CNN: worst-case ASR, mean (SD) over seeds ===")
    g = d.groupby(["eps", "level"]).worst_asr.agg(["mean", lambda s: s.std(ddof=0)])
    g.columns = ["mean", "sd"]
    print(g.round(3).to_string())
    print(f"\nclean macro-F1 = {d.drop_duplicates('seed').clean_f1.mean():.4f}   "
          f"ROC-AUC = {d.drop_duplicates('seed').roc_auc.mean():.5f}")

    # side by side with the paper's MLP (Model 4, ReLU), same attacks and pool
    print("\n=== the question: fraction of L0 success surviving at L2 ===")
    mlp_path = f"results_extra_{DS}_all.csv"
    mlp = None
    if os.path.exists(mlp_path):
        m = pd.read_csv(mlp_path)
        m = m[(m.activation == "relu") & (m.attack.isin(["pgd", "bim", "spsa", "transfer"]))]
        mlp = m.groupby(["eps", "level", "seed"]).asr.max().groupby(["eps", "level"]).mean()
    cnn = d.groupby(["eps", "level"]).worst_asr.mean()
    for e in EPS:
        try:
            c0, c2 = cnn[(e, 0)], cnn[(e, 2)]
            line = f"  eps={e}:  CNN  L0={c0:.3f} L2={c2:.3f}  kept={c2/c0 if c0 else float('nan'):.1%}"
            if mlp is not None:
                m0, m2 = mlp[(e, 0)], mlp[(e, 2)]
                line += f"   |  MLP(M4,ReLU)  L0={m0:.3f} L2={m2:.3f}  kept={m2/m0 if m0 else float('nan'):.1%}"
            print(line)
        except KeyError:
            pass


if __name__ == "__main__":
    main()
