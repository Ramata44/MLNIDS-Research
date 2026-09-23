"""
01_clean_f1_cic.py -- FIXES TABLE 1.

The grid records clean_f1 = best VALIDATION macro-F1 on the balanced
(post-undersample) split; the test set is never scored. Section 3.1 claims the
test set is left in its natural distribution "so that evaluation reflects
realistic class proportions" -- this script computes that number.

No attacks are run, so this is far cheaper than the full grid:
15 configs (5 depths x 3 activations, 30 features) x 3 seeds per dataset.

Set SEED_SPLIT below to also quantify the seed issue:
  False -> reproduces the paper (all 3 seeds share ONE train/val split)
  True  -> re-randomises the split per seed (honest run-to-run variance)

Run from the folder containing nids_cic.py / nids_unsw.py:
    python 01_clean_f1_cic.py
Writes clean_f1_test_<ds>.csv with BOTH val and test F1 side by side.
"""
import os, sys

# ---- locate the code folder (nids_cic.py / nids_unsw.py) wherever this file
# sits, then chdir into it so relative data paths (unsw_processed.npz, the
# kagglehub cache) resolve exactly as the original scripts expect.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [_HERE,
               os.path.join(_HERE, "full code and csv", "fair"),
               os.path.join(_HERE, "fair"),
               os.path.join(_HERE, "..", "full code and csv", "fair"),
               os.path.join(_HERE, "..", "fair")]
for _c in _CANDIDATES:
    if os.path.exists(os.path.join(_c, "nids_unsw.py")):
        _CODE = os.path.abspath(_c); break
else:
    sys.exit("Could not find nids_cic.py / nids_unsw.py. Put this script in "
             "CONF_ML_NIDS/ or beside those files.")
sys.path.insert(0, _CODE); os.chdir(_CODE)
print(f"code dir: {_CODE}")

import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

DS         = "cic"                # hard-wired: this file is the CIC run
SEED_SPLIT = False          # <-- flip to True for the honest-variance version
FEATS      = 30
DEPTHS     = [1, 2, 3, 4, 5]

import nids_cic as G
SEEDS, DEVICE = G.SEEDS, G.DEVICE

Xtr_full, ytr, Xte_full, yte, order = G.load_cic()
print(f"test-set class balance (natural): {np.bincount(yte)}  "
      f"-> minority = {100*np.bincount(yte).min()/len(yte):.2f}%")

k  = min(FEATS, Xtr_full.shape[1]); ok = order[:k]
Xtr, Xte = Xtr_full[:, ok], Xte_full[:, ok]

rows = []
for depth in DEPTHS:
    for act in G.ACTIVATIONS:
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
            rs = seed if SEED_SPLIT else G.SEED          # the one-character bug
            Xtrn, Xval, ytrn, yval = train_test_split(
                Xtr, ytr, test_size=0.1, random_state=rs, stratify=ytr)
            m = G.build_model(depth, k, act).to(DEVICE)
            f1_val  = G.train_to_convergence(m, Xtrn, ytrn, Xval, yval, DEVICE)
            m.eval()
            f1_test = G.macro_f1(m, Xte, yte, DEVICE)    # <-- what Table 1 should report
            rows.append(dict(depth=depth, activation=act, seed=seed,
                             f1_val_balanced=round(float(f1_val), 4),
                             f1_test_natural=round(float(f1_test), 4)))
            print(f"  d{depth} {act:4s} s{seed}: val(bal)={f1_val:.4f}  TEST(nat)={f1_test:.4f}")

df = pd.DataFrame(rows)
df.to_csv(f"clean_f1_test_{DS}.csv", index=False)
piv = df.groupby(["activation", "depth"])[["f1_val_balanced", "f1_test_natural"]].agg(["mean", "std"]).round(3)
print(f"\n=== {DS.upper()}  (SEED_SPLIT={SEED_SPLIT}) ===\n{piv}")
print(f"\nSaved clean_f1_test_{DS}.csv -- use f1_test_natural for Table 1.")
