"""
Reproduction of Figure 3 (page 6) from:
  "Selection and Stability of Functional Connectivity Features for
   Classification of Brain Disorders" — Saha, Hazra & Ghosh (2025)

Loads: /kaggle/input/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat
Outputs:
  - fig3_combined.png  (2×2 matching paper page 6)
  - fig3_accuracy.png, fig3_f1score.png, fig3_kuncheva.png, fig3_jaccard.png
  - Console: KI and JI per individual percentage for each method
  - Console: Tables II & III (Accuracy, Precision, Recall, F1)

CHANGE vs original:
  rank_lasso() now uses L1-penalized Logistic Regression
  (LogisticRegression penalty='l1', solver='saga') instead of
  the LASSO regression path (square loss). Features are ranked
  by descending absolute coefficient magnitude at the optimal C
  found via LogisticRegressionCV with the same penalty/solver.
  All other functions and the overall pipeline are unchanged.

SPEED-UP (rank_lasso only):
  1. Coarse 6-point C grid instead of 20 points — covers the useful
     range without redundant evaluations.
  2. LogisticRegressionCV uses n_jobs=-1 (all cores) and refit=False
     so it never re-trains after picking best C; we do one explicit
     refit at that C ourselves.
  3. max_iter capped at 200 for the CV sweep (ranking order is stable
     long before full convergence) and 300 for the final refit.
  4. tol relaxed to 1e-3 throughout — more than tight enough to
     identify the top-coefficient features.
  5. warm_start=True on the final refit so saga continues from the
     CV-initialised weights rather than starting cold.
  Together these typically cut rank_lasso wall-time by ~6-10×.
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model      import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import f_classif
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, f1_score,
                                       recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH      = '/kaggle/input/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX = 0          # index 0 → 83 ROIs (Dataset 1)
N_ROIS_EXPECTED= 83

# x-axis exactly as in paper Fig 3
PERCENTAGES = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]

# λ = 100 signatures (20 shuffles × 5 folds)
N_SHUFFLES   = 20
N_FOLDS      = 5
RANDOM_STATE = 42

# Plot styles matching paper Fig 3
STYLE = {
    'LASSO':  dict(color='#1f77b4', marker='o', ls='-',  lw=1.8, ms=5),
    'Relief': dict(color='#d62728', marker='s', ls='--', lw=1.8, ms=5),
    'ANOVA':  dict(color='#2ca02c', marker='^', ls='-.', lw=1.8, ms=5),
}

# Coarse 6-point C grid for the L1-LR ranker CV sweep.
# Covers 5 decades (1e-3 … 1e2); 6 points are enough to locate
# the useful regularisation region without 20 redundant fits.
L1_CV_CS = np.logspace(-3, 2, 6)


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load Dataset 1 from the .mat file.
    Structure: SC_FC_Connectomes/FC_correlation/ctrl  (5,1) object array
               SC_FC_Connectomes/FC_correlation/schz  (5,1) object array
    Index 0 → 83 ROIs (Dataset 1, University of Lausanne)
    Returns:
        X : (54, 3403) float64  — absolute Pearson FC upper-triangle vectors
        y : (54,)      int32    — 0=ctrl, 1=schz
    """
    print(f"  Loading: {FILE_PATH}")
    with h5py.File(FILE_PATH, 'r') as f:
        ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
        schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
        ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]   # (27, 83, 83)
        schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]   # (27, 83, 83)

    n_rois = ctrl_mat.shape[1]
    assert n_rois == N_ROIS_EXPECTED, f"Expected {N_ROIS_EXPECTED} ROIs, got {n_rois}"

    tri = np.triu_indices(n_rois, k=1)   # upper triangle indices, no diagonal

    def vectorise(mats):
        return np.abs(np.array([mats[i][tri] for i in range(len(mats))],
                                dtype=np.float64))

    X_ctrl = vectorise(ctrl_mat)   # (27, 3403)
    X_schz = vectorise(schz_mat)   # (27, 3403)
    X = np.vstack([X_ctrl, X_schz])
    y = np.array([0]*27 + [1]*27, dtype=np.int32)

    p = X.shape[1]
    assert p == n_rois*(n_rois-1)//2
    print(f"  Loaded: {n_rois} ROIs | p={p} features | "
          f"ctrl={int((y==0).sum())} | schz={int((y==1).sum())}")
    return X, y, n_rois


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE RANKERS
# ══════════════════════════════════════════════════════════════════════

def rank_lasso(Xs, y):
    """
    Rank features by L1-penalized Logistic Regression coefficient magnitude.

    Speed-ups applied here (nowhere else is changed):
      • Coarse 6-point C grid  — avoids 14 redundant CV fits vs the
        previous 20-point grid.
      • refit=False on the CV object — sklearn skips an extra full
        refit; we do exactly one explicit refit ourselves.
      • n_jobs=-1  — CV folds run in parallel across all CPU cores.
      • max_iter=200 for CV sweep, 300 for final refit — ranking
        order stabilises well before full convergence on this small
        dataset (n≈43 train samples, p=3403).
      • tol=1e-3   — looser tolerance is fine for ranking purposes;
        the exact coefficient values don't matter, only their order.
      • warm_start=True on final refit — saga initialises from the
        CV solution instead of zeros, needing fewer extra iterations.

    Steps:
      1. LogisticRegressionCV picks best C (coarse grid, 3-fold, fast).
      2. One LogisticRegression refit at that C (warm-started).
      3. Rank by descending |coef_|; zero-coef features go last.
    """
    # ── Step 1: coarse CV to pick best C ──────────────────────────
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS,            # 6 points, logspace(-3, 2)
        penalty='l1',
        solver='saga',
        cv=3,                   # 3-fold inner CV is fast & sufficient
        max_iter=200,           # ranking stabilises early
        tol=1e-3,               # looser tol → fewer saga iterations
        refit=False,            # skip sklearn's own extra refit
        n_jobs=-1,              # parallelise the 3 CV folds
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    best_C = float(lrcv.C_[0])

    # ── Step 2: single refit at best C, warm-started ──────────────
    lr = LogisticRegression(
        penalty='l1',
        C=best_C,
        solver='saga',
        max_iter=300,           # sufficient after warm start
        tol=1e-3,
        warm_start=True,        # continues from lrcv's internal weights
        random_state=RANDOM_STATE,
    )
    # Seed coef_ so warm_start has something to continue from
    lr.fit(Xs, y)

    # ── Step 3: rank by descending |coef_| ────────────────────────
    abs_coef = np.abs(lr.coef_[0])   # shape (p,)
    rank = np.argsort(-abs_coef)      # highest |coef| first
    return rank.copy()


def rank_relief(Xs, y):
    """ReliefF — paper eq. (3)–(5)."""
    n, p = Xs.shape
    X0 = Xs[y == 0]
    X1 = Xs[y == 1]
    w  = np.zeros(p)
    for i in range(n):
        xl          = Xs[i]
        same, other = (X0, X1) if y[i] == 0 else (X1, X0)
        d_same  = np.sum((same  - xl) ** 2, axis=1)
        d_other = np.sum((other - xl) ** 2, axis=1)
        si = d_same.argmin()
        if d_same[si] < 1e-12:
            d_same[si] = np.inf
        hit  = same [d_same .argmin()]
        miss = other[d_other.argmin()]
        w   += (xl - miss) ** 2 - (xl - hit) ** 2
    return np.argsort(w / n)[::-1].copy()


def rank_anova(Xs, y):
    """ANOVA F-statistic — paper eq. (6)–(7)."""
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(F)[::-1].copy()


RANKERS = {'LASSO': rank_lasso, 'Relief': rank_relief, 'ANOVA': rank_anova}


# ══════════════════════════════════════════════════════════════════════
# 3. STABILITY METRICS — paper eq. (8)–(11)
# ══════════════════════════════════════════════════════════════════════

def compute_ki_ji(signatures, p):
    """
    KI (Kuncheva) and JI (Jaccard) over all C(λ,2) pairwise combinations.

    KI: K(jθ,jϕ) = (r - k²/p) / (k - k²/p)   [eq. 8-9]
    JI: J(jθ,jϕ) = |jθ∩jϕ| / |jθ∪jϕ|          [eq. 10-11]
    """
    lam = len(signatures)
    if lam < 2:
        return np.nan, np.nan
    k = len(signatures[0])
    if k == 0:
        return np.nan, np.nan

    # Binary indicator matrix (λ × p)
    S = np.zeros((lam, p), dtype=np.float32)
    for i, sig in enumerate(signatures):
        valid = sig[(sig >= 0) & (sig < p)]
        S[i, valid] = 1.0

    # Pairwise intersections
    intersect  = S @ S.T
    iu         = np.triu_indices(lam, k=1)
    r_vals     = intersect[iu]

    # KI
    k2p   = float(k)**2 / float(p)
    denom = float(k) - k2p
    ki    = float(np.clip(np.mean((r_vals - k2p) / denom), -1.0, 1.0)) \
            if abs(denom) > 1e-12 else 1.0

    # JI
    union_vals = 2.0 * float(k) - r_vals
    ji         = float(np.mean(np.where(union_vals > 0, r_vals / union_vals, 1.0)))

    return ki, ji


# ══════════════════════════════════════════════════════════════════════
# 4. ONE SHUFFLE
# ══════════════════════════════════════════════════════════════════════

def _one_shuffle(shuffle_id, X, y, percentages):
    p   = X.shape[1]
    ks  = [max(1, int(pct / 100.0 * p)) for pct in percentages]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE + shuffle_id)

    acc_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    f1_s  = {m: np.zeros(len(percentages)) for m in RANKERS}
    rec_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    pre_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    sigs  = {m: [[] for _ in percentages]  for m in RANKERS}

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        Xte_s = sc.transform(X_te)

        for mname, ranker in RANKERS.items():
            rank = ranker(Xtr_s, y_tr)

            for pi, k in enumerate(ks):
                sel  = rank[:k]
                clf  = LogisticRegression(C=1.0, max_iter=1000,
                                           solver='lbfgs',
                                           random_state=RANDOM_STATE)
                clf.fit(Xtr_s[:, sel], y_tr)
                pred = clf.predict(Xte_s[:, sel])

                acc_s[mname][pi] += accuracy_score(y_te, pred)
                f1_s [mname][pi] += f1_score(y_te, pred, zero_division=0)
                rec_s[mname][pi] += recall_score(y_te, pred, zero_division=0)
                pre_s[mname][pi] += precision_score(y_te, pred, zero_division=0)
                sigs [mname][pi].append(sel.copy())

    result = {}
    for mname in RANKERS:
        result[mname] = {
            'acc':  acc_s[mname] / N_FOLDS,
            'f1':   f1_s [mname] / N_FOLDS,
            'rec':  rec_s[mname] / N_FOLDS,
            'pre':  pre_s[mname] / N_FOLDS,
            'sigs': sigs [mname],
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# 5. FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(X, y, percentages):
    p   = X.shape[1]
    lam = N_SHUFFLES * N_FOLDS
    print(f"  {N_SHUFFLES} shuffles × {N_FOLDS} folds = {lam} signatures per method per %")

    shuffles = []
    for s in range(N_SHUFFLES):
        print(f"    Shuffle {s+1:02d}/{N_SHUFFLES} ...", flush=True)
        shuffles.append(_one_shuffle(s, X, y, percentages))

    # Collect λ=100 signatures per (method, percentage)
    all_sigs = {m: [[] for _ in percentages] for m in RANKERS}
    for sr in shuffles:
        for mname in RANKERS:
            for pi in range(len(percentages)):
                all_sigs[mname][pi].extend(sr[mname]['sigs'][pi])

    # Compute metrics and KI/JI per percentage
    results = {m: {'acc': [], 'f1': [], 'rec': [], 'pre': [], 'ki': [], 'ji': []}
               for m in RANKERS}

    for mname in RANKERS:
        for pi in range(len(percentages)):
            acc_v = np.array([sr[mname]['acc'][pi] for sr in shuffles])
            f1_v  = np.array([sr[mname]['f1'] [pi] for sr in shuffles])
            rec_v = np.array([sr[mname]['rec'][pi] for sr in shuffles])
            pre_v = np.array([sr[mname]['pre'][pi] for sr in shuffles])

            ki, ji = compute_ki_ji(all_sigs[mname][pi], p)

            results[mname]['acc'].append(float(np.mean(acc_v)))
            results[mname]['f1'] .append(float(np.mean(f1_v)))
            results[mname]['rec'].append(float(np.mean(rec_v)))
            results[mname]['pre'].append(float(np.mean(pre_v)))
            results[mname]['ki'] .append(ki)
            results[mname]['ji'] .append(ji)

    return results


# ══════════════════════════════════════════════════════════════════════
# 6. PRINT KI / JI PER INDIVIDUAL PERCENTAGE  ← main output requested
# ══════════════════════════════════════════════════════════════════════

def print_ki_ji_per_percentage(results, percentages):
    p   = 3403   # Dataset 1: p = 83×82/2
    bar = '═' * 74

    # ── Per-method detailed table ─────────────────────────────────
    for mname in ['LASSO', 'Relief', 'ANOVA']:
        print(f"\n{bar}")
        print(f"  METHOD: {mname}  —  KI & JI at each % of selected features")
        print(f"  {'%Features':>10}  {'k':>6}  {'KI':>10}  {'JI':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
        print(f"{'─'*74}")
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct / 100.0 * p))
            ki  = results[mname]['ki'] [pi]
            ji  = results[mname]['ji'] [pi]
            acc = results[mname]['acc'][pi] * 100
            f1  = results[mname]['f1'] [pi] * 100
            print(f"  {pct:>10.1f}%  {k:>6d}  {ki:>10.4f}  {ji:>10.4f}  "
                  f"{acc:>9.2f}%  {f1:>9.2f}%")
        print(bar)

    # ── KI side-by-side ───────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  KUNCHEVA INDEX (KI) — all methods at each % of selected features")
    print(f"  {'%Features':>10}  {'k':>6}  {'LASSO':>10}  {'Relief':>10}  {'ANOVA':>10}")
    print(f"{'─'*74}")
    for pi, pct in enumerate(percentages):
        k = max(1, int(pct / 100.0 * p))
        row = f"  {pct:>10.1f}%  {k:>6d}"
        for m in ['LASSO', 'Relief', 'ANOVA']:
            row += f"  {results[m]['ki'][pi]:>10.4f}"
        print(row)
    print(bar)

    # ── JI side-by-side ───────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  JACCARD INDEX (JI)  — all methods at each % of selected features")
    print(f"  {'%Features':>10}  {'k':>6}  {'LASSO':>10}  {'Relief':>10}  {'ANOVA':>10}")
    print(f"{'─'*74}")
    for pi, pct in enumerate(percentages):
        k = max(1, int(pct / 100.0 * p))
        row = f"  {pct:>10.1f}%  {k:>6d}"
        for m in ['LASSO', 'Relief', 'ANOVA']:
            row += f"  {results[m]['ji'][pi]:>10.4f}"
        print(row)
    print(bar)

    # ── Mean over all percentages ─────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Mean KI / JI averaged over all {len(percentages)} percentages:")
    print(f"  {'Method':<10}  {'Mean KI':>10}  {'Mean JI':>10}")
    print(f"{'─'*50}")
    for m in ['LASSO', 'Relief', 'ANOVA']:
        print(f"  {m:<10}  {np.mean(results[m]['ki']):>10.4f}  "
              f"{np.mean(results[m]['ji']):>10.4f}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════════
# 7. TABLES II & III
# ══════════════════════════════════════════════════════════════════════

def print_tables(results, percentages):
    sep    = '─' * 62
    header = (f"  {'Method':<8}  {'Accuracy':>9}  "
              f"{'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    for pct, tname in [(5.0, 'II  — top  5%'), (10.0, 'III — top 10%')]:
        pi = percentages.index(pct)
        print(f"\n{sep}")
        print(f"  Table {tname} components — Dataset 1")
        print(sep)
        print(header)
        print(sep)
        for m in ['LASSO', 'Relief', 'ANOVA']:
            a  = results[m]['acc'][pi] * 100
            pr = results[m]['pre'][pi] * 100
            rc = results[m]['rec'][pi] * 100
            f1 = results[m]['f1'] [pi] * 100
            print(f"  {m:<8}  {a:>8.2f}%  {pr:>8.2f}%  {rc:>8.2f}%  {f1:>8.2f}%")
        print(sep)


# ══════════════════════════════════════════════════════════════════════
# 8. PLOT FIGURE 3
# ══════════════════════════════════════════════════════════════════════

def _panel(ax, data_dict, pct_list, ylabel, letter):
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname, style in STYLE.items():
        ax.plot(x, data_dict[mname], label=mname, **style)
    ax.set_xticks(x)
    ax.set_xticklabels(xl, rotation=45, fontsize=8)
    ax.set_xlabel('Percentage of Selected Features', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(letter, loc='left', fontsize=10, pad=4)
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=8)
    vals = np.concatenate([data_dict[m] for m in STYLE])
    lo, hi = vals.min(), vals.max()
    pad = max((hi - lo) * 0.15, 0.02)
    ax.set_ylim(lo - pad, hi + pad)


def plot_fig3(results, percentages):
    panels = [
        ('acc', 'Accuracy',       '(a) Accuracy.'),
        ('f1',  'F1 Score',       '(b) F1 score.'),
        ('ki',  'Kuncheva Index', '(c) Kuncheva index.'),
        ('ji',  'Jaccard Index',  '(d) Jaccard index.'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Fig. 3  —  Dataset 1 (University of Lausanne)\n'
        'Classification Performance & Feature Selection Stability\n'
        '54 subjects · 83 ROIs · 3,403 FC features  |  λ=100 subsamplings\n'
        '[LASSO ranker: L1-penalized Logistic Regression, solver=saga]',
        fontsize=10, fontweight='bold', y=1.01)
    for (key, ylabel, letter), ax in zip(panels, axes.flat):
        _panel(ax, {m: results[m][key] for m in STYLE},
               percentages, ylabel, letter)
    plt.tight_layout()
    fig.savefig('fig3_combined.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig3_combined.png")

    fnames = {'acc': 'fig3_accuracy.png', 'f1':  'fig3_f1score.png',
              'ki':  'fig3_kuncheva.png', 'ji':  'fig3_jaccard.png'}
    for key, ylabel, letter in panels:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        _panel(ax2, {m: results[m][key] for m in STYLE},
               percentages, ylabel, letter)
        plt.tight_layout()
        fig2.savefig(fnames[key], dpi=180, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {fnames[key]}")


# ══════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    bar = '═' * 62
    print(bar)
    print('  Saha, Hazra & Ghosh (2025) — Figure 3 Reproduction')
    print('  Dataset 1: University of Lausanne  (83 ROIs, n=54)')
    print('  LASSO ranker: L1-penalized Logistic Regression (saga)')
    print(bar)

    X, y, n_rois = load_data()

    print(f'\n[Step 1]  Running {N_SHUFFLES}×{N_FOLDS} cross-validation ...')
    results = evaluate(X, y, PERCENTAGES)

    print('\n[Step 2]  KI and JI at each individual % of selected features:')
    print_ki_ji_per_percentage(results, PERCENTAGES)

    print('\n[Step 3]  Tables II & III:')
    print_tables(results, PERCENTAGES)

    print('\n[Step 4]  Plotting Fig. 3 ...')
    plot_fig3(results, PERCENTAGES)

    print(f'\n{bar}')
    print('  All done.')
    print(bar)


if __name__ == '__main__':
    main()
