# hsj_only.py -- run ONLY HopSkipJump (PGD-initialised, eps-bounded variant) and
# nothing else, reusing the published pipeline unchanged.
#
# Everything that determines the numbers -- dataset download and preprocessing,
# the fixed split / scaler / pool at seed 42, model architectures, the training
# protocol, adversarial training, the capability masks and the ASR definition --
# is imported from your own scripts (nids_cic.py / nids_unsw.py via
# capability_cic.py / capability_unsw.py). This file only chooses WHICH models to
# train and runs a_hsj_bounded() on them. Nothing is re-implemented here.
#
# The CIC data is downloaded automatically through kagglehub by G.load_cic() the
# first time it is needed (or read from ~/.cache/kagglehub if already present).
# The UNSW data is read from unsw_processed.npz, which prepare_unsw.py builds.
#
# Usage, in the folder that contains capability_cic.py / capability_unsw.py:
#     python hsj_only.py cic
#     python hsj_only.py unsw
#
# Output: results_hsj_<ds>.csv, one row per (defence, level, depth, activation,
# seed, eps). The script RESUMES: rows already in the CSV are skipped, so it can
# be stopped and restarted (HopSkipJump is slow, especially on UNSW).
#
# What to run is set in the CONFIG block below. Defaults reproduce the missing
# HopSkipJump evaluation of the adversarially-trained models (Section 4.6):
# depth 4, three activations, three seeds, standard + PGD-AT, level 0, three
# budgets. Set DEPTHS = [1,2,3,4,5] and DEFENCES = ["standard"] to instead
# re-run the full grid with the bounded HopSkipJump variant.

import os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- CONFIG
DS = (sys.argv[1] if len(sys.argv) > 1 else "cic").lower()   # "cic" | "unsw"
DEPTHS      = [4]                       # [1,2,3,4,5] for the full grid
ACTIVATIONS = ["relu", "elu", "tanh"]
SEEDS       = [1, 7, 42]
FEATS       = 30
EPS         = [0.1, 0.3, 0.5]
LEVELS      = [0]                       # 0 unconstrained; add 2 for realizable
DEFENCES    = ["standard", "pgd_at"]    # ["standard"] for the plain grid
HSJ_MAX_EVAL = None                     # None -> the published cap (3000 CIC / 1000 UNSW);
                                        # set e.g. 3000 to match the cap on both datasets
OUT = f"results_hsj_{DS}.csv"
# -------------------------------------------------------------------------

if DS == "cic":
    import capability_cic as A
else:
    import capability_unsw as A
G = A.G
DEVICE = G.DEVICE
if HSJ_MAX_EVAL is not None:
    G.HSJ_MAX_EVAL = HSJ_MAX_EVAL      # a_hsj_bounded reads G.HSJ_MAX_EVAL


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.defence, d.level, d.depth, d.activation, d.seed, d.eps))


def main():
    print(f"dataset={DS}  device={DEVICE}  depths={DEPTHS}  defences={DEFENCES}  "
          f"levels={LEVELS}  eps={EPS}  HSJ cap={G.HSJ_MAX_EVAL}", flush=True)

    # --- data: identical to the published grid (download if needed) ---
    Xtr_full, ytr, Xte_full, yte, order = (G.load_cic() if DS == "cic"
                                           else G.load_unsw())
    k = min(FEATS, Xtr_full.shape[1]); ok = order[:k]
    Xtr, Xte = Xtr_full[:, ok], Xte_full[:, ok]

    # same fixed pool as every other table: malicious test flows, seed 42
    rng = np.random.default_rng(G.SEED)
    pool = rng.choice(np.where(yte != 0)[0],
                      size=min(G.N_EVAL, int((yte != 0).sum())), replace=False)
    Xe, ye = Xte_full[pool][:, ok], yte[pool]

    masks = {0: None}
    if any(lv > 0 for lv in LEVELS):
        names = A.load_feature_names(G, DS, order, k)
        for lv in LEVELS:
            if lv > 0:
                masks[lv] = A.build_mask(names, DS, lv)

    have = done_keys(OUT)
    rows = []
    t0 = time.time()
    for depth in DEPTHS:
        for act in ACTIVATIONS:
            for seed in SEEDS:
                todo = [(df, lv, eps) for df in DEFENCES for lv in LEVELS for eps in EPS
                        if (df, lv, depth, act, seed, eps) not in have]
                if not todo:
                    print(f"skip d{depth} {act} s{seed}: already done", flush=True)
                    continue

                torch.manual_seed(seed); np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                Xtrn, Xval, ytrn, yval = train_test_split(
                    Xtr, ytr, test_size=0.1,
                    random_state=(G.SEED if A.MATCH_PUBLISHED_SPLIT else seed),
                    stratify=ytr)

                for defence in DEFENCES:
                    if not any(df == defence for df, _, _ in todo):
                        continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    model = G.build_model(depth, k, act).to(DEVICE)
                    if defence == "standard":
                        G.train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
                    else:
                        A.train_adversarial(model, Xtrn, ytrn, Xval, yval, mask=None)
                    model.eval()
                    f1 = G.macro_f1(model, Xte, yte, DEVICE)
                    clf = G._art_clf(model, Xtrn, DEVICE)

                    for lv in LEVELS:
                        for eps in EPS:
                            if (defence, lv, depth, act, seed, eps) in have:
                                continue
                            t1 = time.time()
                            print(f"  {defence:8s} L{lv} d{depth} {act:4s} s{seed} "
                                  f"eps={eps} :: hsj_bounded ...", end="", flush=True)
                            try:
                                adv = A.a_hsj_bounded(clf, Xe, eps, masks[lv])
                                asr = G._asr(model, adv, Xe, ye, DEVICE, eps=eps)
                                print(f" ASR={asr:.4f}  ({time.time()-t1:.0f}s)", flush=True)
                            except Exception as e:
                                print(f" FAILED {type(e).__name__}: {e}", flush=True)
                                asr = float("nan")
                            rows.append((defence, lv, depth, act, k, seed, "hsj_bounded",
                                         eps, round(asr, 4) if asr == asr else None,
                                         round(f1, 4)))
                            # append immediately so the run can resume
                            pd.DataFrame(rows[-1:], columns=[
                                "defence", "level", "depth", "activation", "features",
                                "seed", "attack", "eps", "asr", "clean_f1"]
                            ).to_csv(OUT, mode="a", index=False,
                                     header=not os.path.exists(OUT))
                            have.add((defence, lv, depth, act, seed, eps))

    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    d = pd.read_csv(OUT)
    print("\n=== mean HopSkipJump ASR over seeds ===")
    print(d.groupby(["defence", "level", "depth", "activation", "eps"]).asr
           .agg(["mean", lambda s: s.std(ddof=0)]).round(3).to_string())


if __name__ == "__main__":
    main()
