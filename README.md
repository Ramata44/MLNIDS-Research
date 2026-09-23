# MLNIDS — adversarial robustness of ML-NIDS: budget, architecture and realizability

Code for the paper *"How Perturbation Budget and Architecture Shape the Adversarial
Robustness of ML-NIDS: A Multi-Attack, Realizability and Explainability Study."*

This repository contains the code only. Each script writes its own CSV into the working
directory and the later stages read those files, so run everything from this folder.
Result files are not committed.

## Data

* **CSE-CIC-IDS2018** — the cleaned (Distrinet) release, downloaded automatically through
  `kagglehub` on first use. No preparation step is needed.
* **UNSW-NB15** — run `python prepare_unsw.py` once to build `unsw_processed.npz`.

## Run order

| # | CSE-CIC-IDS2018 | UNSW-NB15 | Writes | Paper |
|---|-----------------|-----------|--------|-------|
| 0 | – (auto-download) | `prepare_unsw.py` | `unsw_processed.npz` | preprocessing |
| 1 | `nids_cic.py` | `nids_unsw.py` | `results_full_*.csv` | Tables 1, 3, 4, B.1 |
| 2 | `clean_f1_cic.py` | `clean_f1_unsw.py` | `clean_f1_test_*.csv` | Table 2 |
| 3 | `attacks_cic.py` | `attacks_unsw.py` | `results_extra_*_all.csv` | Tables 5–8, Figure 1 |
| 4 | `capability_cic.py` | `capability_unsw.py` | `results_advtrain_*.csv` | Tables 9, 10 |
| 5 | `hsj_defended.py cic` | `hsj_defended.py unsw` | `results_hsj_*.csv` | Section 4.6 |
| 6 | `fpr_cic.py` | `fpr_unsw.py` | `results_fpr_*.csv` | Table 11, Figure 2 |
| 7 | `attributions.py cic` | `attributions.py unsw` | `xai_*.csv`, `xai_efffeatures_*.csv` | Table 12, Figure 3 |
| 8 | `gradmargin_cic.py` | `gradmargin_unsw.py` | `results_gradmargin_*.csv` | Tables 13, 14 |
| 9 | `cnn_cic.py` | `cnn_unsw.py` | `results_cnn_*.csv` | Table 8, CNN column |
| – | `fig_asr.py` (both datasets) | | `fig2.pdf`, `fig2.png` | Figure 1 |
| – | `fig_fpr.py` (both datasets) | | `fpr_figure.pdf/.png` | Figure 2 |

Each script is run as `python <name>.py`; the two that take an argument are shown with it.

## What the files are

* `nids_cic.py`, `nids_unsw.py` — the shared library and the depth × activation ×
  feature-count grid: data loading, the fixed split, scaler and evaluation pool at seed 42,
  the model definitions, the training protocol, the attack wrappers and the ASR definition.
* `capability_cic.py`, `capability_unsw.py` — the adversary-capability masks (L0/L1/L2),
  PGD adversarial training and the worst-case ASR used throughout. Imported by the later
  scripts rather than reimplemented.
* `attacks_*`, `hsj_defended`, `fpr_*`, `attributions`, `gradmargin_*`, `cnn_*` — the
  individual experiments, each reusing the two layers above unchanged.
* `fig_asr.py`, `fig_fpr.py` — the two figures, each reading both datasets' results.

## Reproducibility

Every configuration is repeated over seeds 1, 7 and 42; the data split, scaler, class
balancing and evaluation pool are fixed at seed 42, so all configurations are compared on
identical data. Scripts that write a CSV resume from it: rows already present are skipped,
so a run can be stopped and restarted. Training runs on a GPU, whose convolution and
reduction kernels are not bit-for-bit deterministic, so a repeated run reproduces the
reported values to within the seed-to-seed variation rather than exactly.

Note that `nids_cic.py` and `nids_unsw.py` also record a `clean_f1` column during the grid;
that value is measured on the balanced validation split and is **not** the figure reported
in Table 2, which comes from `clean_f1_*.py` on the natural-distribution test set.

## Requirements

`pip install -r requirements.txt` (Python 3.10+).
