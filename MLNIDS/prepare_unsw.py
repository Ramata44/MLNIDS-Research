"""
prepare_unsw.py -- download & process the cleaned UNSW-NB15 dataset
(dhoogla/unswnb15) into the SAME .npz format used by the CSE-CIC-IDS2018
pipeline, so the existing experiment scripts run on it unchanged.

Produces: unsw_processed.npz  (X_train, y_train, X_test, y_test,
                               feature_order, feature_names, scores)

Run:  python prepare_unsw.py

This is the file that was sitting in "full code and csv/fair/second_foundation.py";
it is the step RUN_ORDER.md and nids_unsw.load_unsw() mean by "the UNSW
foundation step". Two changes from that copy, both marked CHANGED below:
  * the .npz is written next to nids_unsw.py rather than into whatever
    directory the kernel happened to start in, so load_unsw() finds it;
  * a missing imbalanced-learn is now a hard error instead of a printed note.
    The published preprocessing applies RandomUnderSampler to the training
    split; skipping it silently produces a DIFFERENT dataset, and every number
    computed from it would be quietly incomparable with the paper.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
OUT = "unsw_processed.npz"

# ---- CHANGED: write the .npz where load_unsw() looks for it -----------------
# nids_unsw.load_unsw() opens "unsw_processed.npz" as a bare filename, so the
# file has to sit in the folder that holds nids_unsw.py. Same anchor search as
# advtrain_unsw_s10.py: no absolute path is written anywhere in this file.
import sys

_ANCHOR = "nids_unsw.py"


def _find_project_dir(start):
    seen, d = [], os.path.abspath(start)
    while True:                                   # upwards
        seen.append(d)
        if os.path.exists(os.path.join(d, _ANCHOR)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for d in seen:                                # then one level down
        try:
            for name in sorted(os.listdir(d)):
                sub = os.path.join(d, name)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, _ANCHOR)):
                    return sub
        except OSError:
            continue
    raise SystemExit(
        f"Could not find {_ANCHOR}.\n"
        f"Searched upwards from {os.path.abspath(start)!r} and one level below "
        f"each parent.\n"
        f"Start the kernel in the project folder, or cd into it first.")


_HERE = _find_project_dir(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
    else os.getcwd())
os.chdir(_HERE)
print(f"Project directory: {_HERE}")

# UNSW-NB15 non-feature / identifier / leakage columns to drop if present
DROP = {"id", "attack_cat", "label", "srcip", "dstip", "sport", "dsport",
        "stime", "ltime", "src ip", "dst ip"}
LABEL_CANDIDATES = ["label", "Label", "attack_cat"]   # binary label preferred


def _find_data():
    """Locate the cleaned UNSW-NB15 files, downloading via kagglehub if needed."""
    # already downloaded?
    for base in [os.path.expanduser("~/.cache/kagglehub"), "."]:
        hits = glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True)
        hits = [h for h in hits if "unsw" in h.lower()]
        if hits:
            print(f"Found {len(hits)} local UNSW parquet files.")
            return hits
    print("Downloading dhoogla/unswnb15 via kagglehub ...")
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub is not installed:\n    pip install kagglehub")
    path = kagglehub.dataset_download("dhoogla/unswnb15")
    hits = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    print(f"Downloaded to {path} ({len(hits)} data files).")
    return hits


def _load(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        print(f"  loaded {os.path.basename(p)}: {df.shape}")
    # align on shared columns
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common = [c for c in frames[0].columns if c in common]
    df = pd.concat([f[common] for f in frames], ignore_index=True)
    print(f"  combined on {len(common)} shared columns: {df.shape}")
    return df


def main():
    paths = _find_data()
    if not paths:
        raise SystemExit("No UNSW-NB15 data files found.")
    df = _load(paths)

    # ---- label (binary: attack=1, normal=0) ----
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise SystemExit(f"No label column found among {LABEL_CANDIDATES}")
    if label_col == "label" or df[label_col].dropna().isin([0, 1]).all():
        y = df[label_col].astype(int).values
    else:  # attack_cat style: 'Normal' vs attack names
        y = (~df[label_col].astype(str).str.strip().str.lower()
             .isin({"normal", "benign"})).astype(int).values
    print(f"  labels: normal={int((y==0).sum())} attack={int((y==1).sum())}")

    # ---- features ----
    X = df.drop(columns=[c for c in df.columns if c.strip().lower() in DROP])
    # one-hot the genuine categoricals (proto/service/state)
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat:
        X = pd.get_dummies(X, columns=cat, dtype=np.float32)
        print(f"  one-hot encoded: {cat}")
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    names = X.columns.tolist()
    X = X.to_numpy(np.float32)
    print(f"  feature matrix: {X.shape}")

    # ---- split / scale / undersample (train only) ----
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    # ---- CHANGED: undersampling is mandatory, not best-effort ---------------
    # The original swallowed the ImportError and carried on. The paper's
    # preprocessing spec is "RandomUnderSampler on train only"; without it the
    # class balance differs, every model trained on the result differs, and the
    # three published seeds would no longer reproduce -- but nothing would say
    # so. Better to stop here than to discover it after a multi-hour grid.
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        raise SystemExit(
            "imbalanced-learn is not installed, but the published preprocessing "
            "applies RandomUnderSampler to the training split. Building the .npz "
            "without it would produce a dataset that is NOT comparable with the "
            "paper.\n    pip install imbalanced-learn")
    Xtr, ytr = RandomUnderSampler(random_state=SEED).fit_resample(Xtr, ytr)
    print(f"  after undersampling: {Xtr.shape}  balance={np.bincount(ytr)}")

    scores = np.nan_to_num(SelectKBest(f_classif, k="all").fit(Xtr, ytr).scores_)
    order = np.argsort(scores)[::-1]
    np.savez_compressed(OUT, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                        feature_order=order, feature_names=np.array(names),
                        scores=scores)
    print(f"\nSaved {OUT}: train={Xtr.shape} test={Xte.shape} "
          f"features={len(names)}")
    print("  top-12 features:", [names[i] for i in order[:12]])


if __name__ == "__main__":
    main()
