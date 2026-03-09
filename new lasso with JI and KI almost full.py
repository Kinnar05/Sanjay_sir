import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, f1_score,
                                       recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH       = '/kaggle/input/datasets/kinnarhalder/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX  = 0          # index 0 → 83 ROIs (Dataset 1)
N_ROIS_EXPECTED = 83

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

# ──────────────────────────────────────────────────────────────────────
# InFusedLasso hyper-parameters  (Algorithm 1 / Eq.6 of the paper)
# ──────────────────────────────────────────────────────────────────────
INFL_LAMBDA1   = 0.01   # sparsity penalty on β  (λ1)
INFL_LAMBDA2   = 0.01   # fused / smoothness penalty on Cβ  (λ2)
INFL_LAMBDA3   = 0.1    # structural interaction weight  (λ3)
INFL_MU1       = 1.0    # augmented-Lagrangian penalty for p=β  (μ1)
INFL_MU2       = 1.0    # augmented-Lagrangian penalty for q=Cβ  (μ2)
INFL_DELTA1    = 1.0    # dual step size for u  (δ1)
INFL_DELTA2    = 1.0    # dual step size for v  (δ2)
INFL_MAX_ITER  = 150    # Algorithm 1 iterations (paper: ~150 to converge)
INFL_TOL       = 1e-4   # convergence tolerance on ‖β^{k+1} − β^k‖


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

# ── InFusedLasso helpers ───────────────────────────────────────────────

def _kernel_adjacency(vec_samples):
    """
    Build the kernel-based adjacency matrix for a single feature.

    Given M samples of feature fi arranged as a column vector
    fi = (fi1, …, fiM)^T, we first form the M×M Euclidean-distance
    adjacency A where A[a,b] = |fia - fib|, then normalise each row
    as a dot-product kernel (Eq. 1 of paper):

        K[a,b] = <A[a,:], A[b,:]> / sqrt(<A[a,:],A[a,:]> <A[b,:],A[b,:]>)

    Returns an (M, M) float32 matrix with values in [0, 1].
    vec_samples : (M,) array — the M sample values of feature fi.
    """
    # Euclidean distance adjacency  (M × M)
    diff = vec_samples[:, None] - vec_samples[None, :]   # (M, M)
    A    = np.abs(diff).astype(np.float32)               # (M, M)

    # Row norms
    norms = np.sqrt(np.sum(A * A, axis=1, keepdims=True))  # (M, 1)
    norms = np.maximum(norms, 1e-12)
    A_norm = A / norms                                      # (M, M)

    # Normalised dot-product kernel  K[a,b] = <A_norm[a,:], A_norm[b,:]>
    K = A_norm @ A_norm.T                                   # (M, M)
    return K


def _target_kernel_adjacency(vec_samples, labels):
    """
    Build the target feature graph adjacency for feature fi (Sec 2.1).

    For each sample a, replace fia with the class-conditional mean
    µ_c(a) where c(a) is the class of sample a.  Then apply the same
    kernel procedure as _kernel_adjacency.

    vec_samples : (M,)     — sample values of feature fi
    labels      : (M,) int — class labels
    """
    classes     = np.unique(labels)
    hat_vec     = vec_samples.copy().astype(np.float32)
    for c in classes:
        mask        = (labels == c)
        hat_vec[mask] = hat_vec[mask].mean()
    return _kernel_adjacency(hat_vec)


def _stationary_distribution(K):
    """
    Convert a non-negative symmetric (M×M) adjacency to a probability
    distribution over M nodes via row-normalisation (sum → 1).
    Returns (M,) float32.
    """
    row_sum = K.sum(axis=1)                    # (M,)
    row_sum = np.maximum(row_sum, 1e-12)
    return (K / row_sum[:, None]).mean(axis=0) # mean of row distributions


def _shannon_entropy(p):
    """Shannon entropy H(p) = -sum p*log(p), ignoring zeros."""
    p   = np.clip(p, 1e-12, 1.0)
    return -float(np.sum(p * np.log(p)))


def _jsd_multi(probs_list):
    """
    Multi-distribution Jensen–Shannon divergence (Eq. 2 of paper)
    for n distributions with equal weights π_i = 1/n.

        DJS = H(mixture) - mean_i H(P_i)

    probs_list : list of (M,) probability vectors
    Returns scalar float.
    """
    n       = len(probs_list)
    mixture = np.mean(np.stack(probs_list, axis=0), axis=0)  # (M,)
    h_mix   = _shannon_entropy(mixture)
    h_mean  = np.mean([_shannon_entropy(p) for p in probs_list])
    return float(np.clip(h_mix - h_mean, 0.0, None))


def _IS(probs_list):
    """
    Structural similarity IS = exp(-DJS)  (Eq. 3 of paper).
    probs_list : list of probability vectors.
    """
    return float(np.exp(-_jsd_multi(probs_list)))


def _build_U_matrix(Xs, y, subsample=200):
    """
    Build the N×N structural information matrix U (Eq. 4 of paper).

        U[i,j] = (IS(Gi,Gj;Ghat_i) + IS(Gi,Gj;Ghat_j)) / IS(Gi,Gj)

    Because p=3403 makes a full N×N matrix (≈12M entries) expensive,
    we compute U only on a random subsample of `subsample` features and
    return the (subsample × subsample) sub-matrix together with the
    selected indices.  The InFusedLasso optimisation is then run on
    this feature sub-space; features outside the sub-space receive
    score 0.

    Xs         : (M, N) standardised feature matrix
    y          : (M,)   class labels
    subsample  : number of features to include (set ≤ N)
    Returns:
        U      : (k, k) float32 structural information matrix
        sel_idx: (k,)   indices of the selected features in [0, N)
    """
    M, N   = Xs.shape
    k      = min(subsample, N)
    rng    = np.random.default_rng(RANDOM_STATE)
    sel_idx = rng.choice(N, size=k, replace=False)
    sel_idx.sort()

    # Pre-compute stationary distributions for features and targets
    feat_probs   = []   # P_i  for feature graph G_i
    target_probs = []   # P_hat_i for target graph G_hat_i

    for idx in sel_idx:
        fi    = Xs[:, idx].astype(np.float32)
        K_f   = _kernel_adjacency(fi)
        K_t   = _target_kernel_adjacency(fi, y)
        feat_probs.append(_stationary_distribution(K_f))
        target_probs.append(_stationary_distribution(K_t))

    # Build U (upper-triangle; then symmetrise)
    U = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i, k):
            is_ij      = _IS([feat_probs[i], feat_probs[j]])
            is_ij_ti   = _IS([feat_probs[i], feat_probs[j], target_probs[i]])
            is_ij_tj   = _IS([feat_probs[i], feat_probs[j], target_probs[j]])
            denom      = max(is_ij, 1e-8)
            u_val      = (is_ij_ti + is_ij_tj) / denom
            U[i, j]    = u_val
            U[j, i]    = u_val

    return U.astype(np.float32), sel_idx


def _build_C_matrix(N):
    """
    Build the (N-1) × N fused-lasso difference matrix C
    (1 on diagonal, -1 on superdiagonal — paper Sec. 3.2).
    """
    C      = np.zeros((N - 1, N), dtype=np.float32)
    idx    = np.arange(N - 1)
    C[idx, idx]     =  1.0
    C[idx, idx + 1] = -1.0
    return C


def _soft_threshold(x, lam):
    """Element-wise soft-thresholding: sign(x)*max(0,|x|-λ)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _infusedlasso_solve(Xs_sub, y, U,
                        lam1=INFL_LAMBDA1, lam2=INFL_LAMBDA2,
                        lam3=INFL_LAMBDA3,
                        mu1=INFL_MU1,  mu2=INFL_MU2,
                        delta1=INFL_DELTA1, delta2=INFL_DELTA2,
                        max_iter=INFL_MAX_ITER, tol=INFL_TOL):
    """
    Solve the InFusedLasso problem (Algorithm 1 / Eq. 6–13 of paper):

        min_{β} 0.5*||y - Xs_sub β||² + λ1||β||₁ + λ2||Cβ||₁ - λ3 β^T U β

    via the augmented-Lagrangian / split-Bregman iteration.

    Xs_sub : (M, k) — feature matrix restricted to selected sub-features
    y      : (M,)   — binary labels  {0, 1}
    U      : (k, k) — structural information matrix
    Returns β* : (k,) coefficient vector.
    """
    M, k  = Xs_sub.shape
    C     = _build_C_matrix(k)              # (k-1, k)
    XT    = Xs_sub.T                        # (k, M)
    XTX   = XT @ Xs_sub                    # (k, k)
    XTy   = XT @ y.astype(np.float32)      # (k,)
    CTC   = C.T @ C                         # (k, k)

    # Pre-compute the (constant) D matrix and its inverse (Eq. 13)
    # D = X^T X - 2λ3 U + μ1 I + μ2 C^T C
    D     = XTX - 2.0 * lam3 * U + mu1 * np.eye(k, dtype=np.float32) \
            + mu2 * CTC
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        D_inv = np.linalg.pinv(D)

    # Initialise primal and dual variables
    beta  = np.zeros(k, dtype=np.float32)
    p     = np.zeros(k, dtype=np.float32)
    q     = np.zeros(k - 1, dtype=np.float32)
    u     = np.zeros(k, dtype=np.float32)
    v     = np.zeros(k - 1, dtype=np.float32)

    for _ in range(max_iter):
        beta_old = beta.copy()

        # ── Update β (Eq. 13) ────────────────────────────────────
        rhs    = XTy + mu1 * (p - u / mu1) + mu2 * (C.T @ (q - v / mu2))
        beta   = D_inv @ rhs

        # ── Update p  (soft threshold, Eq. Algorithm 1 step 3) ──
        p      = _soft_threshold(beta + u / mu1,  lam1 / mu1)

        # ── Update q  (soft threshold, Eq. Algorithm 1 step 4) ──
        q      = _soft_threshold(C @ beta + v / mu2, lam2 / mu2)

        # ── Update dual variables u, v  (steps 5–6) ─────────────
        u     += delta1 * (beta - p)
        v     += delta2 * (C @ beta - q)

        # ── Convergence check ────────────────────────────────────
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def rank_lasso(Xs, y, subsample=200):
    """
    Rank features using InFusedLasso (Bai et al., NeurIPS 2019).

    Steps
    ─────
    1. Randomly draw `subsample` features to form the tractable sub-space.
    2. Build the structural information matrix U on that sub-space
       using kernel-based feature graphs and multi-distribution JSD.
    3. Run the split-Bregman InFusedLasso solver (Algorithm 1) to
       obtain coefficient vector β*.
    4. Features in the sub-space are ranked by descending |β*_i|;
       features outside the sub-space (score = 0) are appended last.

    Parameters
    ──────────
    Xs        : (M, N) standardised float array
    y         : (M,)   integer class labels
    subsample : number of features included in the U-matrix computation
                (set lower to trade accuracy for speed; paper uses all N,
                 but p=3403 makes the full N×N U matrix expensive)

    Returns rank : (N,) integer array — feature indices sorted best-first.
    """
    M, N = Xs.shape

    # ── Step 1–2: structural information matrix ───────────────────
    U, sel_idx = _build_U_matrix(Xs, y, subsample=subsample)
    k          = len(sel_idx)

    # ── Step 3: InFusedLasso optimisation ────────────────────────
    Xs_sub = Xs[:, sel_idx].astype(np.float32)
    beta   = _infusedlasso_solve(Xs_sub, y.astype(np.float32), U)

    # ── Step 4: global ranking ────────────────────────────────────
    scores        = np.zeros(N, dtype=np.float32)
    scores[sel_idx] = np.abs(beta)

    rank = np.argsort(-scores)   # descending
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
        '[LASSO ranker: InFusedLasso — Bai et al., NeurIPS 2019]',
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
    print('  LASSO ranker: InFusedLasso (Bai et al., NeurIPS 2019)')
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
