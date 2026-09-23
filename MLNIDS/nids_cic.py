"""
run_full_grid.py -- unified fair-pool re-run that regenerates ALL tables/figures.

ONE shared evaluation pool of N_EVAL malicious flows is used by all six attacks
(FGSM, PGD, BIM, HopSkipJump, ZOO, SPSA), so white-box and black-box ASR are
directly comparable. Sweeps the full grid with 3 seeds:

  * depth x activation  (depths 1-5 x {relu,elu,tanh}, features=30)  -> Tables 1-5, Fig 1-2
  * feature sweep       (depth-1 relu x feature counts)              -> Table 6
  * clean F1 recorded for every (depth, activation, features, seed)  -> Table 2

Black-box attacks (HSJ, ZOO) are generated ONCE per model and projected into the
L_inf eps-ball for each eps; white-box and SPSA are natively eps-bounded.

Set DATASET below and run once per dataset. Results are appended incrementally to
results_full_<dataset>.csv; hyperparameters are saved to hyperparams.json.

Requires:
  pip install torch adversarial-robustness-toolbox scikit-learn imbalanced-learn pandas numpy kagglehub
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================ CONFIG =========================================
DATASET   = "cic"          # "cic" (CSE-CIC-IDS2018 parquet) or "unsw" (unsw_processed.npz)

SEED      = 42             # fixed split/scaler/pool seed
SEEDS     = [1, 7, 42]     # training seeds swept everywhere
EPS_LIST  = [0.1, 0.3, 0.5]
N_EVAL    = 2000           # SINGLE shared evaluation pool for ALL six attacks

ARCHS = {                  # depth -> hidden layer widths (matches source paper Fig. 3)
    1: [64],
    2: [64, 128],
    3: [64, 128, 128],
    4: [64, 128, 512, 128],
    5: [64, 128, 512, 128, 64],
}
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
ACTIVATIONS = ["relu", "elu", "tanh"]

FEATURES_CIC  = [10, 20, 30, 50, 78]
FEATURES_UNSW = [10, 20, 30, 50, 100, 188]

# training / optimizer
MAX_EPOCHS = 120
CHUNK      = 10
PATIENCE   = 2
TOL        = 1e-3
BATCH_SIZE = 16384
LR         = 1e-3

# attack hyperparameters
FGSM_EPS_STEP_DIV = 10     # PGD/BIM eps_step = eps / 10
PGD_BIM_MAX_ITER  = 40
PGD_RANDOM_INIT   = 1
HSJ_MAX_ITER      = 50
HSJ_MAX_EVAL      = 3000
HSJ_INIT_EVAL     = 100
ZOO_MAX_ITER      = 20
ZOO_BSS           = 3      # binary_search_steps
ZOO_LR            = 1e-1
ZOO_NB_PARALLEL   = 10     # capped below feature count to avoid ZOO crash
SPSA_STEPS        = 20
SPSA_NDIRS        = 8
SPSA_C            = 0.01

SAMPLE_PER_FILE = 300000
DATASET_SLUG    = "dhoogla/distrinetcsecicids2018"
NPZ_UNSW        = "unsw_processed.npz"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV         = f"results_full_{DATASET}.csv"
FRESH           = True     # True = delete any old CSV and run from scratch (no accidental resume)
# =============================================================================


def save_hyperparams():
    hp = {
        "dataset": DATASET, "seed_fixed": SEED, "training_seeds": SEEDS,
        "eps_list": EPS_LIST, "n_eval_shared_pool": N_EVAL,
        "architectures": {str(k): v for k, v in ARCHS.items()},
        "activations": ACTIVATIONS,
        "feature_counts": FEATURES_CIC if DATASET == "cic" else FEATURES_UNSW,
        "optimizer": "Adam", "learning_rate": LR, "batch_size": BATCH_SIZE,
        "loss": "cross_entropy",
        "convergence": {"max_epochs": MAX_EPOCHS, "chunk": CHUNK,
                        "patience": PATIENCE, "tol": TOL,
                        "monitor": "validation macro-F1 (10% of train)"},
        "preprocessing": {"labels": "binary (benign/normal=0, else=1)",
                          "scaling": "StandardScaler fit on train only",
                          "balancing": "RandomUnderSampler on train only",
                          "feature_selection": "top-k ANOVA F-value on train",
                          "split": "80/20 stratified, random_state=42"},
        "attacks": {
            "fgsm": {"lib": "ART", "eps": EPS_LIST},
            "pgd":  {"lib": "ART", "eps_step": "eps/10",
                     "max_iter": PGD_BIM_MAX_ITER, "num_random_init": PGD_RANDOM_INIT},
            "bim":  {"lib": "ART", "eps_step": "eps/10", "max_iter": PGD_BIM_MAX_ITER},
            "hopskipjump": {"lib": "ART", "max_iter": HSJ_MAX_ITER,
                            "max_eval": HSJ_MAX_EVAL, "init_eval": HSJ_INIT_EVAL,
                            "note": "generated once, projected to each eps"},
            "zoo": {"lib": "ART", "max_iter": ZOO_MAX_ITER,
                    "binary_search_steps": ZOO_BSS, "learning_rate": ZOO_LR,
                    "nb_parallel": ZOO_NB_PARALLEL, "batch_size": 1,
                    "note": "generated once, projected to each eps"},
            "spsa": {"lib": "custom (forward-pass only)", "steps": SPSA_STEPS,
                     "n_dirs": SPSA_NDIRS, "c": SPSA_C,
                     "note": "natively eps-bounded, L_inf sign steps"},
        },
        "asr_definition": "fraction of originally-correct malicious flows that "
                          "flip to benign after perturbation",
        "budget_matching": "all six attacks evaluated on the SAME N_EVAL pool; "
                           "black-box adversarials projected into the L_inf eps-ball",
    }
    with open("hyperparams.json", "w") as f:
        json.dump(hp, f, indent=2)
    print("Saved hyperparams.json")


# ----------------------------- data -----------------------------------------
def _find_label_col(df):
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    return df.columns[-1]


def load_cic():
    home = os.path.expanduser("~")
    hits = glob.glob(os.path.join(home, ".cache", "kagglehub", "datasets",
                                  "dhoogla", "distrinetcsecicids2018", "**",
                                  "*.parquet"), recursive=True)
    if hits:
        data_dir = os.path.dirname(hits[0])
    else:
        import kagglehub
        p = kagglehub.dataset_download(DATASET_SLUG)
        data_dir = os.path.dirname(glob.glob(os.path.join(p, "**", "*.parquet"),
                                             recursive=True)[0])
    print(f"Using Distrinet data: {data_dir}")
    paths = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df.columns = [str(c).strip() for c in df.columns]
        if SAMPLE_PER_FILE and len(df) > SAMPLE_PER_FILE:
            df = df.sample(SAMPLE_PER_FILE, random_state=SEED)
        frames.append(df)
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common = [c for c in frames[0].columns if c in common]
    df = pd.concat([f[common] for f in frames], ignore_index=True)
    label_col = _find_label_col(df)
    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & (y_raw.str.lower() != "label")
    X, y_raw = X[mask], y_raw[mask]
    y = (~y_raw.str.lower().isin({"benign", "normal"})).astype(np.int64).values
    X = X.to_numpy(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    try:
        from imblearn.under_sampling import RandomUnderSampler
        X_tr, y_tr = RandomUnderSampler(random_state=SEED).fit_resample(X_tr, y_tr)
    except ImportError:
        pass
    scores = np.nan_to_num(SelectKBest(f_classif, k="all").fit(X_tr, y_tr).scores_)
    order = np.argsort(scores)[::-1]
    print(f"CIC loaded: train={X_tr.shape} test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te, order


def load_unsw():
    if not os.path.exists(NPZ_UNSW):
        raise SystemExit(f"{NPZ_UNSW} not found -- run the UNSW foundation step first.")
    d = np.load(NPZ_UNSW, allow_pickle=True)
    print(f"UNSW loaded: train={d['X_train'].shape} test={d['X_test'].shape}")
    return (d["X_train"], d["y_train"], d["X_test"], d["y_test"], d["feature_order"])


# ----------------------------- model / train --------------------------------
def build_model(depth, n_features, activation):
    layers, prev = [], n_features
    for h in ARCHS[depth]:
        layers += [nn.Linear(prev, h), ACTS[activation]()]
        prev = h
    layers += [nn.Linear(prev, 2)]
    return nn.Sequential(*layers)


def train_clean(model, Xtr, ytr, device, epochs):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx].to(device)), yt[idx].to(device)).backward()
            opt.step()


@torch.no_grad()
def _probs(model, X, device, bs=16384):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(F.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def macro_f1(model, X, y, device):
    return f1_score(y, _probs(model, X, device).argmax(1), average="macro", zero_division=0)


def train_to_convergence(model, Xtr, ytr, Xval, yval, device):
    best, stale, done = -1.0, 0, 0
    while done < MAX_EPOCHS:
        train_clean(model, Xtr, ytr, device, CHUNK)
        done += CHUNK
        f1 = macro_f1(model, Xval, yval, device)
        if f1 > best + TOL:
            best, stale = f1, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best


# ----------------------------- attacks --------------------------------------
class _F32(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, x): return self.model(x.float())


def _art_clf(model, X, device):
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(
        model=_F32(model).to(device), loss=nn.CrossEntropyLoss(),
        input_shape=(X.shape[1],), nb_classes=2,
        clip_values=(float(X.min()), float(X.max())),
        device_type=("gpu" if str(device).startswith("cuda") else "cpu"))


def _wb_adv(clf, X, attack, eps):
    from art.attacks.evasion import (FastGradientMethod, BasicIterativeMethod,
                                     ProjectedGradientDescent)
    if attack == "fgsm":
        atk = FastGradientMethod(clf, eps=eps, batch_size=BATCH_SIZE)
    elif attack == "bim":
        atk = BasicIterativeMethod(clf, eps=eps, eps_step=eps/FGSM_EPS_STEP_DIV,
                                   max_iter=PGD_BIM_MAX_ITER, batch_size=BATCH_SIZE)
    elif attack == "pgd":
        atk = ProjectedGradientDescent(clf, eps=eps, eps_step=eps/FGSM_EPS_STEP_DIV,
                                       max_iter=PGD_BIM_MAX_ITER,
                                       num_random_init=PGD_RANDOM_INIT, batch_size=BATCH_SIZE)
    return atk.generate(x=X.astype(np.float32))


def _hsj_once(clf, X):
    from art.attacks.evasion import HopSkipJump
    return HopSkipJump(classifier=clf, targeted=False, max_iter=HSJ_MAX_ITER,
                       max_eval=HSJ_MAX_EVAL, init_eval=HSJ_INIT_EVAL
                       ).generate(x=np.ascontiguousarray(X, np.float32)).astype(np.float32)


def _zoo_once(clf, X):
    from art.attacks.evasion import ZooAttack
    Xf = np.ascontiguousarray(X, np.float32)
    atk = ZooAttack(classifier=clf, targeted=False, max_iter=ZOO_MAX_ITER,
                    binary_search_steps=ZOO_BSS, learning_rate=ZOO_LR,
                    use_resize=False, use_importance=False,
                    nb_parallel=min(Xf.shape[1], ZOO_NB_PARALLEL), batch_size=1,
                    variable_h=0.2, abort_early=True)
    adv = atk.generate(x=Xf).astype(np.float32)
    bad = ~np.isfinite(adv).all(axis=1); adv[bad] = Xf[bad]
    return adv


def _ce(model, X, y, device):
    p = np.clip(_probs(model, X, device), 1e-8, 1.0)
    return -np.log(p[np.arange(len(y)), y])


def _spsa_adv(model, X, y, eps, device, seed):
    rng = np.random.default_rng(seed)
    x0 = np.ascontiguousarray(X, np.float32); x = x0.copy()
    alpha = eps / SPSA_STEPS * 2.5
    for _ in range(SPSA_STEPS):
        g = np.zeros_like(x)
        for _ in range(SPSA_NDIRS):
            d = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
            g += ((_ce(model, x + SPSA_C*d, y, device) -
                   _ce(model, x - SPSA_C*d, y, device)) / (2*SPSA_C))[:, None] * d
        x = x + alpha * np.sign(g)
        x = np.clip(x, x0 - eps, x0 + eps)
    return x.astype(np.float32)


def _asr(model, adv, X, y, device, eps=None):
    """ASR = flips among originally-correct. If eps given, project adv to eps-ball."""
    if eps is not None:
        adv = (X + np.clip(adv - X, -eps, eps)).astype(np.float32)
    correct = _probs(model, X, device).argmax(1) == y
    flip = correct & (_probs(model, adv, device).argmax(1) != y)
    return float(flip.sum() / max(correct.sum(), 1))


# ----------------------------- config grid ----------------------------------
def build_configs(feature_list):
    cfgs = []
    for depth in [1, 2, 3, 4, 5]:                 # depth x activation @ feat=30
        for act in ACTIVATIONS:
            cfgs.append((depth, act, 30))
    for feat in feature_list:                     # feature sweep @ depth-1 relu
        cfgs.append((1, "relu", feat))
    seen, uniq = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


# ----------------------------- main -----------------------------------------
def main():
    save_hyperparams()
    print("=" * 72)
    print(f" UNIFIED FAIR-POOL FULL GRID  |  dataset={DATASET}  |  N_EVAL={N_EVAL}")
    print(f" seeds={SEEDS}  eps={EPS_LIST}  activations={ACTIVATIONS}")
    print("=" * 72)

    if DATASET == "cic":
        Xtr_full, ytr, Xte_full, yte, order = load_cic()
        feature_list = FEATURES_CIC
    else:
        Xtr_full, ytr, Xte_full, yte, order = load_unsw()
        feature_list = FEATURES_UNSW

    # fixed evaluation-pool ROW indices (same flows for every config/seed)
    mal_idx = np.where(yte != 0)[0]
    rng = np.random.default_rng(SEED)
    pool_rows = rng.choice(mal_idx, size=min(N_EVAL, len(mal_idx)), replace=False)
    print(f"Shared pool: {len(pool_rows)} malicious test flows (fixed across all configs)")
    print(f"Train class balance (post-undersample): {np.bincount(ytr)}")

    configs = build_configs(feature_list)
    print(f"Configs: {len(configs)}  x  {len(SEEDS)} seeds = "
          f"{len(configs)*len(SEEDS)} model trainings\n")

    # FRESH=True removes any old CSV so results never mix across runs
    if FRESH and os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        print(f"FRESH=True -> removed old {OUT_CSV}, running from scratch.")

    # resume support (only active when FRESH=False): skip cells already in CSV
    done = set()
    if not FRESH and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev.depth, prev.activation, prev.features, prev.seed))
        print(f"Resuming: {len(done)} (config,seed) cells already done.")

    for (depth, act, feat) in configs:
        k = min(feat, Xtr_full.shape[1])
        ok = order[:k]
        Xtr, Xte = Xtr_full[:, ok], Xte_full[:, ok]
        Xe = Xte_full[pool_rows][:, ok]
        ye = yte[pool_rows]
        # stratified 90/10 train/val split -> val always has BOTH classes
        Xtrn, Xval, ytrn, yval = train_test_split(
            Xtr, ytr, test_size=0.1, random_state=SEED, stratify=ytr)

        for seed in SEEDS:
            if (depth, act, k, seed) in done:
                continue
            torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
            print(f"--- depth={depth} act={act} feat={k} seed={seed} ---")
            print(f"    val class balance: {np.bincount(yval)}")
            model = build_model(depth, k, act).to(DEVICE)
            f1 = train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
            model.eval()
            clf = _art_clf(model, Xtrn, DEVICE)

            # generate black-box adversarials ONCE, project per eps
            hsj_adv = _hsj_once(clf, Xe)
            zoo_adv = _zoo_once(clf, Xe)

            rows = []
            for eps in EPS_LIST:
                # white-box: generate per eps
                for a in ("fgsm", "pgd", "bim"):
                    asr = _asr(model, _wb_adv(clf, Xe, a, eps), Xe, ye, DEVICE)
                    rows.append((depth, act, k, seed, a, eps, round(asr, 4), round(f1, 4)))
                # black-box: project the once-generated adversarials
                rows.append((depth, act, k, seed, "hopskipjump", eps,
                             round(_asr(model, hsj_adv, Xe, ye, DEVICE, eps=eps), 4), round(f1, 4)))
                rows.append((depth, act, k, seed, "zoo", eps,
                             round(_asr(model, zoo_adv, Xe, ye, DEVICE, eps=eps), 4), round(f1, 4)))
                # spsa: natively eps-bounded, generate per eps
                rows.append((depth, act, k, seed, "spsa", eps,
                             round(_asr(model, _spsa_adv(model, Xe, ye, eps, DEVICE, seed),
                                        Xe, ye, DEVICE), 4), round(f1, 4)))

            df = pd.DataFrame(rows, columns=["depth", "activation", "features", "seed",
                                             "attack", "eps", "asr", "clean_f1"])
            df.to_csv(OUT_CSV, mode="a", index=False, header=not os.path.exists(OUT_CSV))
            wb = df[df.attack.isin(["fgsm","pgd","bim"]) & (df.eps==0.3)].asr.mean()
            bb = df[df.attack.isin(["hopskipjump","zoo","spsa"]) & (df.eps==0.3)].asr.mean()
            print(f"    clean_F1={f1:.4f}  eps0.3: WB={wb:.3f} BB={bb:.3f}")

    print(f"\nDone. Granular results in {OUT_CSV}; hyperparameters in hyperparams.json")


if __name__ == "__main__":
    main()
