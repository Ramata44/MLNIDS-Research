# xai_grad_unsw.py -- the gradient-margin test, UNSW-NB15 ONLY.
#
# This is the UNSW-NB15 counterpart of xai_grad.py. The dataset is hard-set here,
# so there is no command-line argument to get wrong and nothing to edit before
# running. It imports capability_unsw.py (never capability_cic.py) and reads the UNSW
# data from unsw_processed.npz, which prepare_unsw.py builds.
#
# WHAT IT MEASURES
#
# Section 4.7 reports that the activation which spreads attribution over more
# features is the less evaded one, then gives two reasons not to read that
# causally: (a) for an L_inf adversary the achievable change in the logit scales
# with the L1 norm of the input gradient, which GROWS as weight is spread over
# more features, so "breadth" is arguably the wrong quantity; and (b) the ordering
# is confounded with goodness of fit, since a better-fitting model has a larger
# decision margin. This script measures the quantity that settles both at once.
#
# For a locally linear model an L_inf perturbation of size eps changes the margin
# by at most eps * ||g||_1, where g is the gradient of the margin with respect to
# the input. A flow therefore flips when
#
#       eps  >=  margin / ||g||_1  =:  eps_hat
#
# so eps_hat is the predicted perturbation needed to evade, per flow. It holds
# both quantities: the margin (the fit confound) in the numerator and the gradient
# norm (which breadth of reliance drives) in the denominator. The fraction of
# originally-correct flows with eps_hat <= eps is a PREDICTED ASR, directly
# comparable to the measured ASR in the paper's tables.
#
# Restricting g to the features a realizable adversary controls gives the L2
# version of the same prediction, so the test also speaks to Section 4.5. On
# UNSW-NB15 that is the point of interest: this is the dataset on which the
# realizable adversary places ReLU last, and this test either explains that
# quantitatively or contradicts it.
#
# Everything that determines the numbers -- data loading and preprocessing, the
# fixed split/scaler/pool at seed 42, model architectures, the training protocol
# and the capability masks -- is imported from your own scripts (nids_unsw.py
# via capability_unsw.py). Nothing is re-implemented here; this file trains the
# standard models and differentiates them.
#
# It runs NO attacks. Cost is 3 activations x 3 seeds of clean training plus one
# backward pass over the 500-flow evaluation pool.
#
# Usage, in the folder that contains capability_unsw.py:
#     python xai_grad_unsw.py
#
# Output: results_gradmargin_unsw.csv, one row per (activation, seed, level, eps).
# The script RESUMES: rows already in the CSV are skipped.

import os, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- CONFIG
DS          = "unsw"                     # hard-set: this file is UNSW-NB15 only
DEPTH       = 4
FEATS       = 30
ACTIVATIONS = ["relu", "elu", "tanh"]
SEEDS       = [1, 7, 42]
LEVELS      = [0, 2]                     # 0 unconstrained, 2 realizable
EPS         = [0.1, 0.3, 0.5]
OUT         = "results_gradmargin_unsw.csv"
# -------------------------------------------------------------------------

import capability_unsw as A
G = A.G
DEVICE = G.DEVICE


def margin_and_grad(model, X, y, device, bs=2048):
    """Margin of the true class over the other, and its input gradient.

    The model has two output logits. For a flow with true label y the margin is
    logit[y] - logit[1-y]; it is positive exactly when the flow is classified
    correctly, and an attack succeeds when a perturbation drives it below zero.
    """
    model.eval()
    M, Gr = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32,
                          device=device, requires_grad=True)
        yb = torch.tensor(y[i:i + bs], dtype=torch.long, device=device)
        out = model(xb)
        other = 1 - yb
        m = out.gather(1, yb[:, None]).squeeze(1) - out.gather(1, other[:, None]).squeeze(1)
        g, = torch.autograd.grad(m.sum(), xb)
        M.append(m.detach().cpu().numpy())
        Gr.append(g.detach().cpu().numpy())
    return np.concatenate(M, 0), np.concatenate(Gr, 0)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.activation, d.seed, d.level, d.eps))


def main():
    print(f"dataset={DS} (hard-set)  device={DEVICE}  depth={DEPTH}  "
          f"levels={LEVELS}  eps={EPS}  (no attacks)", flush=True)

    Xtr_full, ytr, Xte_full, yte, order = G.load_unsw()
    k = min(FEATS, Xtr_full.shape[1]); ok = order[:k]
    Xtr = Xtr_full[:, ok]

    # the same fixed evaluation pool as every attack table
    rng = np.random.default_rng(G.SEED)
    pool = rng.choice(np.where(yte != 0)[0],
                      size=min(G.N_EVAL, int((yte != 0).sum())), replace=False)
    Xe, ye = Xte_full[pool][:, ok], yte[pool].astype(int)
    print(f"evaluation pool: {len(ye)} malicious flows, {k} features", flush=True)

    # capability masks, built by your own classifier (1 = perturbable)
    names = A.load_feature_names(G, DS, order, k)
    masks = {lv: (np.ones(k, dtype=np.float32) if lv == 0
                  else A.build_mask(names, DS, lv)) for lv in LEVELS}
    for lv in LEVELS:
        print(f"  L{lv}: {int(masks[lv].sum())} of {k} features perturbable",
              flush=True)

    have = done_keys(OUT)
    t0 = time.time()
    for act in ACTIVATIONS:
        for seed in SEEDS:
            if all((act, seed, lv, e) in have for lv in LEVELS for e in EPS):
                print(f"skip {act} s{seed}: already done", flush=True)
                continue

            torch.manual_seed(seed); np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            Xtrn, Xval, ytrn, yval = train_test_split(
                Xtr, ytr, test_size=0.1,
                random_state=(G.SEED if A.MATCH_PUBLISHED_SPLIT else seed),
                stratify=ytr)

            torch.manual_seed(seed); np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = G.build_model(DEPTH, k, act).to(DEVICE)
            G.train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
            model.eval()

            m, g = margin_and_grad(model, Xe, ye, DEVICE)

            # ASR is defined over originally-correct flows only (the _asr helper
            # of the grid module), so the prediction uses the same denominator.
            correct = m > 0
            n_corr = int(correct.sum())

            for lv in LEVELS:
                gl1 = np.abs(g[correct] * masks[lv][None, :]).sum(1)
                # a flow whose perturbable gradient is zero can never be flipped
                # by this linearization: eps_hat = inf
                with np.errstate(divide="ignore", invalid="ignore"):
                    eps_hat = np.where(gl1 > 0, m[correct] / gl1, np.inf)

                finite = np.isfinite(eps_hat)
                for e in EPS:
                    if (act, seed, lv, e) in have:
                        continue
                    row = {
                        "activation": act, "seed": seed, "level": lv, "eps": e,
                        "depth": DEPTH, "features": k,
                        "n_correct": n_corr,
                        "pred_asr": round(float((eps_hat <= e).mean()), 4),
                        "median_eps_hat": round(float(np.median(eps_hat[finite]))
                                                if finite.any() else float("nan"), 4),
                        "mean_margin": round(float(m[correct].mean()), 4),
                        "mean_grad_l1": round(float(gl1.mean()), 4),
                        "frac_zero_grad": round(float((gl1 == 0).mean()), 4),
                    }
                    pd.DataFrame([row]).to_csv(OUT, mode="a", index=False,
                                               header=not os.path.exists(OUT))
                    have.add((act, seed, lv, e))
                    print(f"  {act:4s} s{seed} L{lv} eps={e}: "
                          f"pred_ASR={row['pred_asr']:.4f}  "
                          f"median_eps_hat={row['median_eps_hat']:.4f}  "
                          f"margin={row['mean_margin']:.3f}  "
                          f"|g|_1={row['mean_grad_l1']:.3f}", flush=True)

    d = pd.read_csv(OUT)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\n=== predicted ASR, mean over seeds (compare with the measured ASR) ===")
    p = d.groupby(["level", "eps", "activation"]).pred_asr.agg(
        ["mean", lambda s: s.std(ddof=0)]).round(3)
    p.columns = ["mean", "sd"]
    print(p.to_string())
    print("\n=== margin and gradient norm separately, mean over seeds ===")
    print(d.groupby(["level", "activation"])[["mean_margin", "mean_grad_l1",
                                              "median_eps_hat"]]
           .mean().round(4).to_string())
    print("\nThe test: does the ordering of pred_asr across activations match the "
          "ordering of the measured ASR at the same level and budget? On UNSW-NB15 "
          "the measured L2 ordering places ReLU last, so the question is whether "
          "the restricted gradient norm reproduces that.")


if __name__ == "__main__":
    main()
