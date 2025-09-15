# -*- coding: utf-8 -*-
"""
Ultra-precise calibration (Linke fixed at -1, AfD fixed at +1)
- Optimize 4 angles only: [GRUENE, SPD, FDP, CDU]
- Fix DIE LINKE = -1, AfD = +1 (hard constraint)
- Smooth per-angle bounds (±0.25 from initial, clipped to [-1,1]) via tanh reparam
- Circular-mean objective with row/outlet weighting + robust soft_l1 loss
- Full analytic Jacobian incl. weight derivatives
"""

import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# =========================
# Hyperparameters (tune here)
# =========================
USE_PARALLEL          = True
USE_THREADS           = True
N_WORKERS             = min(8, (os.cpu_count() or 1))
N_RESTARTS            = 7
RANDOM_SEED           = 42

POLITIC_THRESHOLD     = 0.8
RESULTS_DIR           = "results"

# Robust outer loss
ROBUST_LOSS           = "soft_l1"
F_SCALE               = 0.03

# Row/outlet weighting
BETA_ROW              = 1.0   # row weight exponent: w_j ∝ r_j**BETA_ROW
DELTA_POL             = 1.0   # multiply by politic**DELTA_POL (0 to ignore)
ALPHA_OUTLET          = 1.0   # outlet weight exponent on resultant length ρ

# Solver budgets/tolerances
MAX_ITERS             = 2000
MAX_SECS              = 180
MAX_NFEV              = 25000
XTOL, FTOL, GTOL      = 1e-14, 1e-14, 1e-12

PRINT_EVERY           = 0  # 0 disables progress printing

# Dtypes / eps
DTYPE_DATA            = np.float64
DTYPE_MATS            = np.float64
EPS_DENOM             = 1e-24
RHO_FLOOR             = 1e-12

INV_HALF_PI = 2.0 / np.pi  # equals 1/(π/2)

# =========================
# Utilities
# =========================
def convertMedienlandschaft(x: float) -> float:
    # 1..7 -> [-1,1]
    return (x - 4.0) / 3.0

def base_angle(v):
    """v = [y, x]; return atan2(y, x)/(π/2) in [-1,1]."""
    return np.arctan2(v[0], v[1]) * INV_HALF_PI

def principal_angle(delta_rad: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta_rad), np.cos(delta_rad))

# Smooth bounds mapping: θ = center + halfwidth * tanh(z)
def make_bound_mapping(initial_angles: np.ndarray, rng_half: float = 0.25):
    lo = np.maximum(-1.0, initial_angles - rng_half)
    hi = np.minimum( 1.0, initial_angles + rng_half)
    center = (lo + hi) / 2.0
    halfwidth = (hi - lo) / 2.0
    halfwidth = np.maximum(halfwidth, 1e-12)

    def unpack(z):
        t = np.tanh(z)
        return center + halfwidth * t

    def dtheta_dz(z):
        t = np.tanh(z)
        return halfwidth * (1.0 - t*t)

    def pack(theta):
        t = np.clip((theta - center) / halfwidth, -0.999999, 0.999999)
        return np.arctanh(t)

    return unpack, dtheta_dz, pack, lo, hi, center, halfwidth

# Compose a full 6-angle vector from the 4 optimized variables (indices 1..4)
def compose_full_theta(theta_vars4, fixed_vals):
    full = np.empty(6, dtype=DTYPE_DATA)
    # order: [LINKE, GRUENE, SPD, FDP, CDU, AfD]
    full[0] = fixed_vals['linke']   # -1
    full[5] = fixed_vals['afd']     # +1
    full[1:5] = theta_vars4
    return full

# =========================
# Residuals & Jacobian (weighted circular mean) wrt z (4 vars only)
# =========================
def residuals_and_jacobian_weighted_circmean_wrt_z_subset(
    z_vars4,
    mats_for_classifier,
    row_base_w_list,
    targets_scaled,
    unpack4, dtheta_dz4,
    fixed_vals
):
    """
    Returns residuals r (per outlet) and Jacobian J (m x 4) wrt z_vars4.
    Variables are the 4 middle parties: [GRUENE, SPD, FDP, CDU] (indices 1..4).
    DIE LINKE and AfD are held fixed at -1 and +1 respectively.
    """
    theta_vars4 = unpack4(z_vars4)                  # (4,)
    theta_full6 = compose_full_theta(theta_vars4, fixed_vals)  # (6,)
    ang   = theta_full6 * (np.pi / 2.0)

    sin_k = np.sin(ang) ;  cos_k = np.cos(ang)      # (6,)
    d_sin =  cos_k * (np.pi / 2.0)                  # (6,)
    d_cos = -sin_k * (np.pi / 2.0)                  # (6,)
    dthdz = dtheta_dz4(z_vars4)                     # (4,)

    var_idx = [1,2,3,4]   # GRUENE, SPD, FDP, CDU (opt variables)
    m = len(mats_for_classifier)
    r = np.empty(m, dtype=DTYPE_DATA)
    J = np.zeros((m, 4), dtype=DTYPE_DATA)

    for i, (C, base_w, target_scaled) in enumerate(zip(mats_for_classifier, row_base_w_list, targets_scaled)):
        N = C.shape[0]
        if N == 0:
            r[i] = -target_scaled
            continue

        # Projections per row
        y  = C @ sin_k
        x  = C @ cos_k
        rr = np.sqrt(x*x + y*y) + EPS_DENOM

        s  = y / rr
        c  = x / rr

        # Row weights (base * rr^beta)
        if DELTA_POL != 0.0:
            w = base_w * np.power(rr, BETA_ROW)
        else:
            w = np.power(rr, BETA_ROW)
        W  = w.sum(dtype=DTYPE_DATA)

        Ns = (w * s).sum(dtype=DTYPE_DATA)
        Nc = (w * c).sum(dtype=DTYPE_DATA)
        s_bar = Ns / (W + EPS_DENOM)
        c_bar = Nc / (W + EPS_DENOM)

        # Residual (scaled)
        theta_bar = np.arctan2(s_bar, c_bar)
        theta_tgt = target_scaled * (np.pi / 2.0)
        delta     = principal_angle(theta_bar - theta_tgt)
        r_i       = INV_HALF_PI * delta

        # Outlet weight rho^alpha
        rho2 = s_bar*s_bar + c_bar*c_bar + EPS_DENOM
        rho  = np.sqrt(rho2)
        w_out = np.power(rho, ALPHA_OUTLET)
        r[i] = w_out * r_i

        # Derivatives
        inv_r3 = 1.0 / (rr*rr*rr)
        ds_dx  = -x*y * inv_r3
        ds_dy  =  (x*x) * inv_r3
        dc_dx  =  (y*y) * inv_r3
        dc_dy  = -x*y * inv_r3

        denom_bar = rho2

        for j, k in enumerate(var_idx):  # only the 4 variables
            dyk  = C[:, k] * d_sin[k]
            dxk  = C[:, k] * d_cos[k]

            ds_dth = (ds_dx * dxk + ds_dy * dyk)             # (N,)
            dc_dth = (dc_dx * dxk + dc_dy * dyk)             # (N,)
            dr_dth = (x*dxk + y*dyk) / rr                    # (N,)

            dw_dth = (BETA_ROW * w / rr) * dr_dth            # (N,)

            dNs = (dw_dth * s + w * ds_dth).sum(dtype=DTYPE_DATA)
            dNc = (dw_dth * c + w * dc_dth).sum(dtype=DTYPE_DATA)
            dW  = dw_dth.sum(dtype=DTYPE_DATA)

            ds_bar = (dNs*(W+EPS_DENOM) - Ns*dW) / ((W+EPS_DENOM)**2)
            dc_bar = (dNc*(W+EPS_DENOM) - Nc*dW) / ((W+EPS_DENOM)**2)

            dtheta_bar = (c_bar * ds_bar - s_bar * dc_bar) / denom_bar
            dr_dth_k   = INV_HALF_PI * dtheta_bar

            drho_dth   = (s_bar*ds_bar + c_bar*dc_bar) / max(rho, RHO_FLOOR)
            dw_out_dth = (ALPHA_OUTLET * np.power(rho, max(ALPHA_OUTLET-1.0, 0.0))) * drho_dth

            # Chain to z via dθ/dz for this variable index
            J_theta = w_out * dr_dth_k + dw_out_dth * r_i
            J[i, j] = J_theta * dthdz[j]

    return r, J

# =========================
# Targets, parties, newspapers
# =========================
v_linke  = [-1.0, 0.0]
v_gruene = [-0.9077674463309972, 0.4194737934385177]
v_spd    = [-0.8074405688999996, 0.5899489195637577]
v_fdp    = [0.0, 1.0]
v_cdu    = [0.614758038308033, 0.788715762702673]
v_afd    = [1.0, 0.0]

# Order: [LINKE, GRUENE, SPD, FDP, CDU, AfD]
initial_angles_full = np.array([
    base_angle(v_linke),
    base_angle(v_gruene),
    base_angle(v_spd),
    base_angle(v_fdp),
    base_angle(v_cdu),
    base_angle(v_afd),
], dtype=DTYPE_DATA)

# Fix endpoints
FIXED_VALS = {"linke": -1.0, "afd": 1.0}

# Variable subset (indices 1..4)
var_initial = initial_angles_full[1:5].copy()

# Smooth bounds for the 4 variables (±0.25 clipped)
unpack4, dtheta_dz4, pack4, bounds_lo4, bounds_hi4, bounds_center4, bounds_halfwidth4 = make_bound_mapping(var_initial, 0.25)
z0_default = pack4(var_initial)  # start in unconstrained space (4,)

l_parties = ['DIE LINKE', 'BÜNDNIS 90/DIE GRÜNEN', 'SPD', 'FDP', 'CDU/CSU', 'AfD']

l_newspapers = [
    'NLP-UniBW/deutschlandfunk_de_classified',
    'NLP-UniBW/focus_de_classified',
    'NLP-UniBW/linksunten_classified',
    'NLP-UniBW/taz_de_classified',
    'NLP-UniBW/zeit_de_classified',
    'NLP-UniBW/stern_de_classified',
    'NLP-UniBW/tichyseinblick_de_classified',
    'NLP-UniBW/cicero_de_classified',
    'NLP-UniBW/spiegel_de_classified',
    'NLP-UniBW/vice_de_classified',
    'NLP-UniBW/tagesschau_de_classified',
    'NLP-UniBW/sueddeutsche_de_classified',
    'NLP-UniBW/welt_de_classified',
    'NLP-UniBW/mdr_de_classified',
    'NLP-UniBW/der_freitag_de_classified',
    'NLP-UniBW/frankfurter_rundschau_de_classified',
    'NLP-UniBW/bild_de_classified',
    'NLP-UniBW/russia_today_de_classified',
    'NLP-UniBW/tagesspiegel_de_classified',
    'NLP-UniBW/br_de_classified',
    'NLP-UniBW/achgut_de_classified',
    'NLP-UniBW/wdr_de_classified',
    'NLP-UniBW/neues_deutschland_de_classified',
    'NLP-UniBW/compact_de_classified',
    'NLP-UniBW/ndr_de_classified',
    'NLP-UniBW/nachdenkseiten_de_classified',
    'NLP-UniBW/junge_freiheit_de_classified',
    'NLP-UniBW/rtl_de_classified',
    'NLP-UniBW/junge_welt_classified',
    'NLP-UniBW/ntv_de_classified',
    'NLP-UniBW/jungle_world_classified',
    'NLP-UniBW/frankfurter_allgemeine_de_classified',
    'NLP-UniBW/mm_news_de_classified',
]

exp_targets = {
    'NLP-UniBW/deutschlandfunk_de_classified': convertMedienlandschaft(3.8),
    'NLP-UniBW/focus_de_classified':           convertMedienlandschaft(4.9),
    'NLP-UniBW/linksunten_classified':         convertMedienlandschaft(2.0),
    'NLP-UniBW/taz_de_classified':             convertMedienlandschaft(2.8),
    'NLP-UniBW/zeit_de_classified':            convertMedienlandschaft(3.6),
    'NLP-UniBW/stern_de_classified':           convertMedienlandschaft(3.8),
    'NLP-UniBW/tichyseinblick_de_classified':  convertMedienlandschaft(5.5),
    'NLP-UniBW/cicero_de_classified':          convertMedienlandschaft(4.9),
    'NLP-UniBW/spiegel_de_classified':         convertMedienlandschaft(3.5),
    'NLP-UniBW/vice_de_classified':            convertMedienlandschaft(2.8),
    'NLP-UniBW/tagesschau_de_classified':      convertMedienlandschaft(3.7),
    'NLP-UniBW/sueddeutsche_de_classified':    convertMedienlandschaft(3.5),
    'NLP-UniBW/welt_de_classified':            convertMedienlandschaft(4.8),
    'NLP-UniBW/mdr_de_classified':             convertMedienlandschaft(4.1),
    'NLP-UniBW/der_freitag_de_classified':     convertMedienlandschaft(2.7),
    'NLP-UniBW/frankfurter_rundschau_de_classified': convertMedienlandschaft(3.4),
    'NLP-UniBW/bild_de_classified':            convertMedienlandschaft(5.2),
    'NLP-UniBW/russia_today_de_classified':    convertMedienlandschaft(5.1),
    'NLP-UniBW/tagesspiegel_de_classified':    convertMedienlandschaft(3.6),
    'NLP-UniBW/br_de_classified':              convertMedienlandschaft(4.4),
    'NLP-UniBW/achgut_de_classified':          convertMedienlandschaft(5.2),
    'NLP-UniBW/wdr_de_classified':             convertMedienlandschaft(3.5),
    'NLP-UniBW/neues_deutschland_de_classified': convertMedienlandschaft(2.6),
    'NLP-UniBW/compact_de_classified':         convertMedienlandschaft(6.0),
    'NLP-UniBW/ndr_de_classified':             convertMedienlandschaft(3.7),
    'NLP-UniBW/nachdenkseiten_de_classified':  convertMedienlandschaft(3.1),
    'NLP-UniBW/junge_freiheit_de_classified':  convertMedienlandschaft(5.8),
    'NLP-UniBW/rtl_de_classified':             convertMedienlandschaft(4.5),
    'NLP-UniBW/junge_welt_classified':         convertMedienlandschaft(2.4),
    'NLP-UniBW/ntv_de_classified':             convertMedienlandschaft(4.3),
    'NLP-UniBW/jungle_world_classified':       convertMedienlandschaft(2.3),
    'NLP-UniBW/frankfurter_allgemeine_de_classified': convertMedienlandschaft(4.5),
    'NLP-UniBW/mm_news_de_classified':         convertMedienlandschaft(5.1),
}
exp_arr_global = np.array([exp_targets[name] for name in l_newspapers], dtype=DTYPE_DATA)

# =========================
# Load data & discover classifiers
# =========================
print("Discovering classifiers from base dataset...")
base_df = load_dataset(l_newspapers[0], split="train").to_pandas()
candidate_cols = [c for c in base_df.columns if any(c.endswith("_" + p) for p in l_parties)]
l_classifier = sorted({c.rsplit("_", 1)[0] for c in candidate_cols})
print(f"Found {len(l_classifier)} classifiers.")

print("Loading & filtering newspapers (politic >= 0.8)...")
d_newspapers_df = {}
for np_name in tqdm(l_newspapers):
    df = load_dataset(np_name, split="train").to_pandas()
    if "politic" in df.columns:
        df = df[df["politic"] >= POLITIC_THRESHOLD].reset_index(drop=True)
    d_newspapers_df[np_name] = df

# =========================
# Builders & helpers
# =========================
def build_mats_for_classifier(classifier: str, subsample_rows: int | None = None):
    """
    Build per-newspaper (N,6) matrices and per-row base weights (politic^DELTA_POL or ones).
    """
    cols = [f"{classifier}_{p}" for p in l_parties]
    mats, row_base_w = [], []
    rng_local = np.random.default_rng(abs(hash(classifier)) % (2**32))
    for np_name in l_newspapers:
        df = d_newspapers_df[np_name]
        if subsample_rows and len(df) > subsample_rows:
            idx = rng_local.choice(len(df), subsample_rows, replace=False)
            df = df.iloc[idx]

        if not all(c in df.columns for c in cols):
            M = np.zeros((len(df), len(l_parties)), dtype=DTYPE_MATS)
            for j, c in enumerate(cols):
                if c in df.columns:
                    M[:, j] = df[c].to_numpy(dtype=DTYPE_MATS, copy=False)
        else:
            M = df[cols].to_numpy(dtype=DTYPE_MATS, copy=False)
        mats.append(np.ascontiguousarray(M))

        if "politic" in df.columns and DELTA_POL != 0.0:
            w0 = np.power(df["politic"].to_numpy(dtype=DTYPE_MATS, copy=False), DELTA_POL)
        else:
            w0 = np.ones(len(df), dtype=DTYPE_MATS)
        row_base_w.append(w0)

    return mats, row_base_w

def has_coverage(classifier: str, min_frac=0.5) -> bool:
    cols = [f"{classifier}_{p}" for p in l_parties]
    good = sum(1 for np_name in l_newspapers
               if any(c in d_newspapers_df[np_name].columns for c in cols)
               and len(d_newspapers_df[np_name]) > 0)
    return (good / len(l_newspapers)) >= min_frac

def mk_fun_and_jac(mats, row_base_w, targets_scaled):
    """
    Caching wrapper for fun/jac at same z (4 vars).
    """
    last = {"z": None, "r": None, "J": None}
    def _compute(z_vars4):
        r, J = residuals_and_jacobian_weighted_circmean_wrt_z_subset(
            z_vars4, mats, row_base_w, targets_scaled, unpack4, dtheta_dz4, FIXED_VALS
        )
        last["z"], last["r"], last["J"] = z_vars4.copy(), r, J

    def fun(z_vars4):
        if last["z"] is None or not np.array_equal(z_vars4, last["z"]):
            _compute(z_vars4)
        return last["r"]

    def jac(z_vars4):
        if last["z"] is None or not np.array_equal(z_vars4, last["z"]):
            _compute(z_vars4)
        return last["J"]

    return fun, jac

def unweighted_circ_mse_full(theta_full6, mats, targets_scaled):
    """
    Unweighted circular-mean MSE across outlets (for reporting).
    """
    ang   = theta_full6 * (np.pi / 2.0)
    sin_k = np.sin(ang);  cos_k = np.cos(ang)
    errs  = []
    for C, tgt in zip(mats, targets_scaled):
        if len(C) == 0:
            errs.append(tgt)
            continue
        y = C @ sin_k
        x = C @ cos_k
        rr = np.sqrt(x*x + y*y) + EPS_DENOM
        s_bar = (y / rr).mean()
        c_bar = (x / rr).mean()
        theta_bar = np.arctan2(s_bar, c_bar)
        delta = principal_angle(theta_bar - tgt*(np.pi/2.0))
        errs.append(INV_HALF_PI * delta)
    errs = np.array(errs, dtype=DTYPE_DATA)
    return float(np.mean(errs*errs))

# Optional progress callback — prints full 6 angles
Nfeval = 1
def callbackF_angles(theta_full6):
    global Nfeval
    print('{0:4d}   {1: .12f}   {2: .12f}   {3: .12f}   {4: .12f}   {5: .12f}   {6: .12f}'.format(
        Nfeval, theta_full6[0], theta_full6[1], theta_full6[2],
        theta_full6[3], theta_full6[4], theta_full6[5]
    ))
    Nfeval += 1

def make_ls_callback(print_every=0, max_iter=None, deadline_ts=None):
    k = {'i': 0}
    def _cb(z_vars4, *args, **kwargs):
        k['i'] += 1
        if print_every and (k['i'] % print_every == 0):
            try:
                theta_vars4 = unpack4(z_vars4)
                theta_full6 = compose_full_theta(theta_vars4, FIXED_VALS)
                callbackF_angles(theta_full6)
            except Exception:
                pass
        if max_iter and k['i'] >= max_iter: return True
        if deadline_ts and time.time() >= deadline_ts: return True
        return False
    return _cb

# =========================
# Optimization per classifier
# =========================
os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

def solve_one(classifier: str):
    if not has_coverage(classifier):
        return classifier, {"classifier": classifier, "skipped": True, "reason": "low coverage"}

    mats, row_base_w = build_mats_for_classifier(classifier, None)
    fun, jac = mk_fun_and_jac(mats, row_base_w, exp_arr_global)

    # Multi-starts in 4D z-space
    starts = [z0_default.copy()]
    for _ in range(N_RESTARTS - 1):
        z = z0_default + rng.normal(0.0, 0.05, size=4)
        starts.append(z)

    best = None
    best_z = None
    for start in starts:
        deadline = time.time() + MAX_SECS
        cb = make_ls_callback(PRINT_EVERY, MAX_ITERS, deadline)
        res = least_squares(
            fun=fun,
            x0=start,
            jac=jac,
            method="trf",                 # bounds handled by tanh reparam
            loss=ROBUST_LOSS,
            f_scale=F_SCALE,
            xtol=XTOL, ftol=FTOL, gtol=GTOL,
            max_nfev=MAX_NFEV,
            verbose=0,
            callback=cb
        )
        if (best is None) or (res.cost < best.cost):
            best, best_z = res, res.x

    theta_vars4 = unpack4(best_z)
    theta_full6 = compose_full_theta(theta_vars4, FIXED_VALS)

    # Weighted MSE (using residuals from fun)
    r_opt, _J = residuals_and_jacobian_weighted_circmean_wrt_z_subset(
        best_z, mats, row_base_w, exp_arr_global, unpack4, dtheta_dz4, FIXED_VALS
    )
    weighted_mse = float(np.mean(r_opt*r_opt))

    # Unweighted circular-mean MSE for comparison
    unweighted_mse = unweighted_circ_mse_full(theta_full6, mats, exp_arr_global)

    out = {
        "classifier": classifier,
        "optimized_angles": {
            "linke": float(theta_full6[0]),  # will be -1.0
            "gruene": float(theta_full6[1]),
            "spd":   float(theta_full6[2]),
            "fdp":   float(theta_full6[3]),
            "cdu":   float(theta_full6[4]),
            "afd":   float(theta_full6[5]),  # will be +1.0
        },
        "fixed": {"linke": -1.0, "afd": 1.0},
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "status": int(best.status),
        "message": str(best.message),
        "weighted_mse": weighted_mse,
        "unweighted_circ_mse": unweighted_mse,
        "bounds_variables": {
            "lo": [float(v) for v in bounds_lo4],
            "hi": [float(v) for v in bounds_hi4],
            "order": ["gruene", "spd", "fdp", "cdu"]
        },
        "tolerances": {"xtol": XTOL, "ftol": FTOL, "gtol": GTOL},
        "loss": {"type": ROBUST_LOSS, "f_scale": F_SCALE},
        "weights": {"beta_row": BETA_ROW, "delta_politic": DELTA_POL, "alpha_outlet": ALPHA_OUTLET},
        "restarts": N_RESTARTS
    }
    with open(os.path.join(RESULTS_DIR, f"{classifier}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return classifier, out

def main():
    print("Optimizing per classifier (GRUENE/SPD/FDP/CDU), with LINKe=-1 and AfD=+1 fixed...")
    results = []
    if USE_PARALLEL:
        Executor = ThreadPoolExecutor if USE_THREADS else ProcessPoolExecutor
        with Executor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(solve_one, c) for c in l_classifier]
            for fut in tqdm(as_completed(futs), total=len(futs)):
                try:
                    classifier, res = fut.result()
                except Exception as e:
                    classifier, res = "unknown", {"error": str(e)}
                results.append((classifier, res))
    else:
        for c in tqdm(l_classifier):
            classifier, res = solve_one(c)
            results.append((classifier, res))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Done. Saved {len(results)} files to '{RESULTS_DIR}/'.")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main()
