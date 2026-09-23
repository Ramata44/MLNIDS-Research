"""
run_xai.py -- Feature-impact (SHAP) analysis for the activation comparison,
matching the main experiment EXACTLY: same data pipeline, same depth-4 /
30-feature config, same fixed evaluation pool, same three seeds (1, 7, 42).

For each activation (ReLU, ELU, Tanh) it trains the depth-4 model (as in
run_full_grid.py), computes per-feature attribution on the SAME N_EVAL pool the
attacks used, averages the mean |attribution| over the three seeds, and reports
the top features per activation with +/- std across seeds.

Attribution = GradientSHAP-style expected-gradients (integrated gradients from
random baselines), implemented in pure PyTorch so no `shap` package is needed.

Set DATASET; run once per dataset. Outputs:
  xai_<dataset>.csv          (feature, activation, mean_abs_attr, std_attr)
  xai_<dataset>.png / .pdf   (top-feature impact per activation)

Requires: torch scikit-learn imbalanced-learn pandas numpy matplotlib kagglehub
"""
import os, glob, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---- CONFIG: identical to run_full_grid.py ----
# dataset: pass "cic" or "unsw" on the command line (default "cic")
_args = [a for a in sys.argv[1:] if a.lower() in ("cic", "unsw")]
DATASET   = (_args[0] if _args else "cic").lower()
SEED      = 42
SEEDS     = [1, 7, 42]            # SAME three seeds as the main experiment
N_EVAL    = 2000                  # CIC pool; set to 500 for UNSW just below
DEPTH     = 4                     # depth-4 (ARCHS[4])
FEATS     = 30                    # 30 features
ARCHS     = {1:[64], 2:[64,128], 3:[64,128,128], 4:[64,128,512,128], 5:[64,128,512,128,64]}
ACTS      = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
ACTIVATIONS = ["relu", "elu", "tanh"]
MAX_EPOCHS, CHUNK, PATIENCE, TOL = 120, 10, 2, 1e-3
BATCH_SIZE, LR = 16384, 1e-3
N_BASELINE = 64      # random baselines for expected-gradients
N_STEPS    = 32      # interpolation steps
TOPK_PLOT  = 12      # top features to show
SAMPLE_PER_FILE = 300000
NPZ_UNSW = "unsw_processed.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DATASET == "unsw":
    N_EVAL = 500     # match the UNSW run's pool


# ---------- data (identical pipeline) ----------
def _label_col(df):
    for c in df.columns:
        if c.strip().lower() == "label": return c
    return df.columns[-1]

def load_cic():
    home = os.path.expanduser("~")
    hits = glob.glob(os.path.join(home,".cache","kagglehub","datasets","dhoogla",
                     "distrinetcsecicids2018","**","*.parquet"), recursive=True)
    data_dir = os.path.dirname(hits[0])
    paths = sorted(glob.glob(os.path.join(data_dir,"*.parquet")))
    frames=[]
    for p in paths:
        df=pd.read_parquet(p); df.columns=[str(c).strip() for c in df.columns]
        if SAMPLE_PER_FILE and len(df)>SAMPLE_PER_FILE: df=df.sample(SAMPLE_PER_FILE,random_state=SEED)
        frames.append(df)
    common=set(frames[0].columns)
    for f in frames[1:]: common&=set(f.columns)
    common=[c for c in frames[0].columns if c in common]
    df=pd.concat([f[common] for f in frames],ignore_index=True)
    lc=_label_col(df); yr=df[lc].astype(str).str.strip()
    X=df.drop(columns=[lc]).apply(pd.to_numeric,errors="coerce").astype("float32").replace([np.inf,-np.inf],np.nan)
    m=X.notna().all(axis=1)&(yr.str.lower()!="label"); X,yr=X[m],yr[m]
    y=(~yr.str.lower().isin({"benign","normal"})).astype(np.int64).values
    names=list(X.columns); X=X.to_numpy(np.float32)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    from imblearn.under_sampling import RandomUnderSampler
    Xtr,ytr=RandomUnderSampler(random_state=SEED).fit_resample(Xtr,ytr)
    sc2=np.nan_to_num(SelectKBest(f_classif,k="all").fit(Xtr,ytr).scores_)
    order=np.argsort(sc2)[::-1]
    return Xtr,ytr,Xte,yte,order,names

def load_unsw():
    d=np.load(NPZ_UNSW,allow_pickle=True)
    names=list(d["feature_names"]) if "feature_names" in d else [f"f{i}" for i in range(d["X_train"].shape[1])]
    return d["X_train"],d["y_train"],d["X_test"],d["y_test"],d["feature_order"],names


# ---------- model / train (identical) ----------
def build(depth,nf,act):
    layers,prev=[],nf
    for h in ARCHS[depth]: layers+=[nn.Linear(prev,h),ACTS[act]()]; prev=h
    layers+=[nn.Linear(prev,2)]; return nn.Sequential(*layers)

def train_clean(model,X,y,ep):
    model.to(DEVICE).train(); opt=torch.optim.Adam(model.parameters(),lr=LR)
    X=torch.tensor(X,dtype=torch.float32); y=torch.tensor(y,dtype=torch.long); n=len(X)
    for _ in range(ep):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]; opt.zero_grad()
            F.cross_entropy(model(X[idx].to(DEVICE)),y[idx].to(DEVICE)).backward(); opt.step()

@torch.no_grad()
def f1_(model,X,y):
    model.eval(); out=[]
    for i in range(0,len(X),16384):
        out.append(F.softmax(model(torch.tensor(X[i:i+16384],dtype=torch.float32,device=DEVICE)),1).cpu().numpy())
    return f1_score(y,np.concatenate(out).argmax(1),average="macro",zero_division=0)

def converge(model,Xtr,ytr,Xv,yv):
    best,stale,done=-1,0,0
    while done<MAX_EPOCHS:
        train_clean(model,Xtr,ytr,CHUNK); done+=CHUNK; f=f1_(model,Xv,yv)
        if f>best+TOL: best,stale=f,0
        else:
            stale+=1
            if stale>=PATIENCE: break
    return best


# ---------- expected-gradients attribution (GradientSHAP-style) ----------
def expected_gradients(model, X_eval, X_ref, n_baseline=N_BASELINE, n_steps=N_STEPS):
    """Mean |attribution| per feature for class 1 (malicious), integrated from
    random baselines drawn from the reference set (training data)."""
    model.eval()
    Xe = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
    nfeat = Xe.shape[1]
    attr_sum = torch.zeros(nfeat, device=DEVICE)
    rng = np.random.default_rng(SEED)
    for _ in range(n_baseline):
        base = torch.tensor(X_ref[rng.integers(0, len(X_ref))], dtype=torch.float32, device=DEVICE)
        alphas = torch.rand(n_steps, device=DEVICE)
        for a in alphas:
            x = (base + a * (Xe - base)).clone().detach().requires_grad_(True)
            out = model(x)[:, 1].sum()
            g, = torch.autograd.grad(out, x)
            attr_sum += (g * (Xe - base)).abs().mean(0)
    return (attr_sum / (n_baseline * n_steps)).detach().cpu().numpy()


# ---------- main ----------
def main():
    print(f"XAI feature-impact | dataset={DATASET} | depth={DEPTH} feats={FEATS} "
          f"| seeds={SEEDS} | pool={N_EVAL}")
    if DATASET=="cic":
        Xtr_f,ytr,Xte_f,yte,order,names=load_cic()
    else:
        Xtr_f,ytr,Xte_f,yte,order,names=load_unsw()

    k=min(FEATS,Xtr_f.shape[1]); ok=order[:k]
    feat_names=[names[i] for i in ok]
    Xtr,Xte=Xtr_f[:,ok],Xte_f[:,ok]

    # SAME fixed pool of malicious flows as the attacks used
    mal=np.where(yte!=0)[0]; rng=np.random.default_rng(SEED)
    pool=rng.choice(mal,size=min(N_EVAL,len(mal)),replace=False)
    Xe=Xte[pool]

    def eff_num_features(attr):
        """inverse Simpson index of the normalized |attribution| vector."""
        p = attr / attr.sum()
        return float(1.0 / np.sum(p**2))

    # per-activation attribution, averaged over the 3 seeds
    rows=[]; eff_rows=[]
    per_act_mean={}; per_act_std={}
    for act in ACTIVATIONS:
        seed_attrs=[]; seed_eff=[]
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
            Xtrn,Xv,ytrn,yv=train_test_split(Xtr,ytr,test_size=0.1,random_state=SEED,stratify=ytr)
            model=build(DEPTH,k,act).to(DEVICE)
            converge(model,Xtrn,ytrn,Xv,yv)
            attr=expected_gradients(model, Xe, Xtrn)
            seed_attrs.append(attr)
            seed_eff.append(eff_num_features(attr))   # effective #features THIS seed
            print(f"  {act} seed {seed}: done  (eff#feat={seed_eff[-1]:.2f})")
        A=np.vstack(seed_attrs)
        per_act_mean[act]=A.mean(0); per_act_std[act]=A.std(0)
        eff_rows.append({"activation":act,
                         "eff_features_mean":round(float(np.mean(seed_eff)),3),
                         "eff_features_std": round(float(np.std(seed_eff)),3)})
        for fi,fn in enumerate(feat_names):
            rows.append({"feature":fn,"activation":act,
                         "mean_abs_attr":round(float(A.mean(0)[fi]),6),
                         "std_attr":round(float(A.std(0)[fi]),6)})

    pd.DataFrame(rows).to_csv(f"xai_{DATASET}.csv",index=False)
    pd.DataFrame(eff_rows).to_csv(f"xai_efffeatures_{DATASET}.csv",index=False)
    print(f"Saved xai_{DATASET}.csv and xai_efffeatures_{DATASET}.csv")
    print("Effective #features (mean +/- std over seeds):")
    for r in eff_rows:
        print(f"  {r['activation']:5s}: {r['eff_features_mean']:.2f} +/- {r['eff_features_std']:.2f}")

    # ---- figure: top-K features by ReLU impact, compared across activations ----
    ref=per_act_mean["relu"]
    topidx=np.argsort(ref)[::-1][:TOPK_PLOT]
    labels=[feat_names[i] for i in topidx]
    x=np.arange(len(topidx)); w=0.25
    colors={"relu":"#3b6fb0","elu":"#c0504d","tanh":"#e2915f"}
    fig,ax=plt.subplots(figsize=(9.5,4.6))
    for j,act in enumerate(ACTIVATIONS):
        vals=[per_act_mean[act][i] for i in topidx]
        errs=[per_act_std[act][i] for i in topidx]
        ax.bar(x+(j-1)*w,vals,w,yerr=errs,capsize=2,label=act.upper(),color=colors[act],error_kw={"elinewidth":0.7})
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=7)
    ax.set_ylabel("Mean |attribution|")
    ax.set_title(f"Top-{TOPK_PLOT} feature impact by activation ({DATASET.upper()}, depth-{DEPTH})")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"xai_{DATASET}.png",dpi=300,bbox_inches="tight")
    fig.savefig(f"xai_{DATASET}.pdf",bbox_inches="tight")
    print(f"Saved xai_{DATASET}.png/.pdf")


if __name__=="__main__":
    main()