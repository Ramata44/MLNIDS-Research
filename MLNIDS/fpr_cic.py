# fpr_only.py -- false-positive cost of PGD adversarial training. No attacks.
#
# Section 4.6 reports the cost of the defense as clean macro-F1. Macro-F1 weights
# the two classes equally, which cancels the class imbalance, so it does not say
# what the defense costs an operator in false alarms. This script answers that
# directly: for the standard and the PGD-AT models it reports the false-positive
# rate at a FIXED true-positive rate, so the alert cost is isolated from any
# change in detection.
#
# Everything that determines the numbers -- data loading and preprocessing, the
# fixed split/scaler at seed 42, model architectures, the training protocol and
# the adversarial-training procedure -- is imported from your own scripts
# (nids_cic.py / nids_unsw.py via capability_cic.py / capability_unsw.py). Nothing
# is re-implemented here; this file only trains the models and scores them.
#
# It runs NO attacks, so it costs training time only: 3 activations x 3 seeds x
# 2 defences per dataset. The standard models are quick; PGD-AT is roughly 8x a
# clean run.
#
# Usage, in the folder that contains capability_cic.py / capability_unsw.py:
#     python fpr_only.py cic
#     python fpr_only.py unsw
# or set DS_DEFAULT below when running inside a notebook.
#
# Output: results_fpr_<ds>.csv, one row per (defence, activation, seed), with
#   fpr@TPR for each target TPR, plus roc_auc, clean macro-F1, and the
#   false-positive rate at the model's own 0.5 threshold.
# The script RESUMES: rows already in the CSV are skipped.

import os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# ----------------------------------------------------------------- CONFIG
DS_DEFAULT = "cic"                       # "cic" | "unsw" (used when no CLI arg)
_args = [a for a in sys.argv[1:] if a.lower() in ("cic", "unsw")]
DS = (_args[0] if _args else DS_DEFAULT).lower()
DEPTH       = 4
FEATS       = 30
ACTIVATIONS = ["relu", "elu", "tanh"]
SEEDS       = [1, 7, 42]
DEFENCES    = ["standard", "pgd_at"]
TPR_TARGETS = [0.90, 0.95, 0.99]         # report FPR at each of these detection rates
OUT = f"results_fpr_{DS}.csv"
# -------------------------------------------------------------------------

if DS == "cic":
    import capability_cic as A
else:
    import capability_unsw as A
G = A.G
DEVICE = G.DEVICE


def fpr_at_tpr(y_true, score, target):
    """Smallest FPR among thresholds whose TPR >= target (linear interpolation
    between the two bracketing ROC points)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    if tpr[-1] < target:
        return float("nan")
    return float(np.interp(target, tpr, fpr))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.defence, d.activation, d.seed))


def main():
    print(f"dataset={DS}  device={DEVICE}  defences={DEFENCES}  "
          f"TPR targets={TPR_TARGETS}  (no attacks)", flush=True)

    Xtr_full, ytr, Xte_full, yte, order = (G.load_cic() if DS == "cic"
                                           else G.load_unsw())
    k = min(FEATS, Xtr_full.shape[1]); ok = order[:k]
    Xtr, Xte = Xtr_full[:, ok], Xte_full[:, ok]
    n_benign, n_mal = int((yte == 0).sum()), int((yte != 0).sum())
    print(f"test set: {len(yte)} flows, {n_benign} benign, {n_mal} malicious "
          f"({100*n_mal/len(yte):.1f}% malicious)", flush=True)

    have = done_keys(OUT)
    t0 = time.time()
    for act in ACTIVATIONS:
        for seed in SEEDS:
            if all((df, act, seed) in have for df in DEFENCES):
                print(f"skip {act} s{seed}: already done", flush=True)
                continue

            torch.manual_seed(seed); np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            Xtrn, Xval, ytrn, yval = train_test_split(
                Xtr, ytr, test_size=0.1,
                random_state=(G.SEED if A.MATCH_PUBLISHED_SPLIT else seed),
                stratify=ytr)

            for defence in DEFENCES:
                if (defence, act, seed) in have:
                    continue
                t1 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                model = G.build_model(DEPTH, k, act).to(DEVICE)
                if defence == "standard":
                    G.train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
                else:
                    A.train_adversarial(model, Xtrn, ytrn, Xval, yval, mask=None)
                model.eval()

                # malicious-class probability on the natural-distribution test set
                p = G._probs(model, Xte, DEVICE)[:, 1]
                y = (yte != 0).astype(int)

                row = {"defence": defence, "activation": act, "seed": seed,
                       "depth": DEPTH, "features": k,
                       "clean_f1": round(G.macro_f1(model, Xte, yte, DEVICE), 4),
                       "roc_auc": round(float(roc_auc_score(y, p)), 4)}
                for t in TPR_TARGETS:
                    row[f"fpr_at_tpr{int(t*100)}"] = round(fpr_at_tpr(y, p, t), 5)
                # operating point the model actually uses
                pred = (p >= 0.5).astype(int)
                row["fpr_at_0.5"] = round(float(((pred == 1) & (y == 0)).sum()
                                                / max((y == 0).sum(), 1)), 5)
                row["tpr_at_0.5"] = round(float(((pred == 1) & (y == 1)).sum()
                                                / max((y == 1).sum(), 1)), 5)

                pd.DataFrame([row]).to_csv(OUT, mode="a", index=False,
                                           header=not os.path.exists(OUT))
                have.add((defence, act, seed))
                print(f"  {defence:8s} {act:4s} s{seed}: "
                      f"F1={row['clean_f1']:.4f}  AUC={row['roc_auc']:.4f}  "
                      f"FPR@95={row['fpr_at_tpr95']:.5f}  "
                      f"({time.time()-t1:.0f}s)", flush=True)

    d = pd.read_csv(OUT)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    cols = ["clean_f1", "roc_auc"] + [f"fpr_at_tpr{int(t*100)}" for t in TPR_TARGETS] \
           + ["fpr_at_0.5", "tpr_at_0.5"]
    print("\n=== mean over seeds, by defence and activation ===")
    print(d.groupby(["defence", "activation"])[cols].mean().round(5).to_string())
    print("\n=== mean over activations and seeds (the numbers for Section 4.6) ===")
    m = d.groupby("defence")[cols].mean().round(5)
    print(m.to_string())
    if {"standard", "pgd_at"} <= set(m.index):
        for t in TPR_TARGETS:
            c = f"fpr_at_tpr{int(t*100)}"
            a, b = m.loc["standard", c], m.loc["pgd_at", c]
            factor = (b / a) if a > 0 else float("nan")
            print(f"TPR={t:.2f}: FPR {a:.5f} -> {b:.5f}  "
                  f"({factor:.1f}x, +{100*(b-a):.2f} percentage points)")


if __name__ == "__main__":
    main()
