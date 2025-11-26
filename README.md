"""
NSS fitting for:
1) Risk-free curve (currency-specific constraints)
2) Credit spread curve (positivity + tail control)
3) Total corporate yield = risk-free + spread

Features:
- Reads Excel with tenor, rf_zero_yield, spread_zero_yield
- Auto-detects % vs decimal
- Fits NSS for RF and Spread using nonlinear least squares
- Enforces basic economic constraints
- Exports:
    * Full smooth curves (CSV)
    * Standard tenor table 3M, 6M, 1Y, 2Y, ... 30Y (Excel)
- Prints:
    * NSS parameters for RF and Spread
    * In-sample R², Out-of-sample R², AIC, BIC, overfitting gap
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================================================
# CONFIG
# ============================================================

# TODO: adjust this to your real file path and sheet name
EXCEL_PATH = "C:/Users/FG977WW/OneDrive - EY/Automation Project/Inputs.xlsx"
DATA_SHEET = "ICE_data"
CIQ_SHEET = ""

TENOR_COL = "tenor"
RF_COL = "rf_zero_yield"
SPREAD_COL = "spread_zero_yield"
CIQ_TENOR_COL = "CIQ_tenor"
CIQ_YIELD_COL = "CIQ_yield"
 
CURRENCY = "EUR"  # affects y_min and epsilon constraints

OUT_DIR = "C:/Users/FG977WW/OneDrive - EY/Automation Project/Outputs_Curves"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def r2_score_manual(y, y_hat):
    y = np.asarray(y, float)
    y_hat = np.asarray(y_hat, float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - (ss_res / ss_tot)

def parse_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return np.nan
    try:
        return float(x)
    except Exception:
        s = str(x).replace("%","").replace(",","").strip()
        try:
            return float(s)
        except Exception:
            return np.nan

def maybe_pct(arr):
    arr = np.asarray(arr, float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return arr
    m = np.nanmax(np.abs(arr))
    # if typical values look like 2, 5, 10 -> interpret as %
    if m > 1.5:
        return arr / 100.0
    return arr

def normalize_tenor(val):
    s = str(val).upper().strip()
    num = parse_num(s)
    if np.isnan(num):
        return np.nan
    if "Y" in s:
        return num
    if "M" in s:
        return num / 12.0
    if "D" in s:
        return num / 365.0
    return num

# ============================================================
# CURRENCY CONSTRAINTS
# ============================================================

def get_constraints(currency):
    c = currency.upper()
    if c == "USD":
        return -0.001, 0.0
    if c == "EUR":
        return -0.005, 0.0005
    if c == "CHF":
        return -0.0075, 0.0005
    if c == "JPY":
        return -0.002, 0.001
    if c == "GBP":
        return 0.0, 0.0
    return -0.002, 0.0005

Y_MIN, EPSILON = get_constraints(CURRENCY)

# ============================================================
# NSS MODEL
# ============================================================

def nss_yield(t, b0, b1, b2, b3, t1, t2):
    t = np.maximum(np.asarray(t, float), 1e-8)
    f1 = (1 - np.exp(-t/t1)) / (t/t1)
    f2 = f1 - np.exp(-t/t1)
    f3 = (1 - np.exp(-t/t2)) / (t/t2) - np.exp(-t/t2)
    return b0 + b1*f1 + b2*f2 + b3*f3

def nss_forward(t, b0, b1, b2, b3, t1, t2):
    """Instantaneous forward: f(t) = y(t) + t*y'(t)."""
    t = np.maximum(np.asarray(t, float), 1e-8)
    y = nss_yield(t, b0, b1, b2, b3, t1, t2)

    e1 = np.exp(-t/t1)
    e2 = np.exp(-t/t2)

    # derivatives of the shape functions w.r.t t (approx analytical)
    dy1 = (e1/(t/t1**2)) - ((1-e1)/(t**2/t1))
    dy2 = dy1 - (-e1/t1)
    dy3 = (e2/(t/t2**2)) - ((1-e2)/(t**2/t2)) - (-e2/t2)

    dydt = b1*dy1 + b2*dy2 + b3*dy3
    return y + t*dydt

# ============================================================
# NSS FITTING
# ============================================================

def fit_nss(tenor, y, is_rf=False):
    tenor = np.asarray(tenor, float)
    y = np.asarray(y, float)

    # Initial guess
    b0_init = float(y[-1])
    b1_init = float(y[0] - y[-1])
    init = np.array([b0_init, b1_init, 0.0, 0.0, 1.5, 5.0], float)

    # Bounds: 6 parameters
    lb = np.array([-0.20, -0.20, -0.20, -0.20, 0.05, 0.05], float)
    ub = np.array([ 0.30,  0.30,  0.30,  0.30,50.0,50.0], float)

    # clip initial guess inside bounds
    init = np.minimum(np.maximum(init, lb + 1e-8), ub - 1e-8)

    def resid(p):
        return nss_yield(tenor, *p) - y

    # First pass
    res = least_squares(resid, init, bounds=(lb, ub), max_nfev=50000, method="trf")
    p = res.x.copy()

    # Enforce simple economic constraints
    def enforce(p):
        b0, b1, b2, b3, t1, t2 = p

        # ensure tau ordering
        if t2 < t1:
            t1, t2 = t2, t1

        # risk-free short-end constraint
        if is_rf:
            if b0 + b1 < Y_MIN:
                b1 = max(Y_MIN - b0, -0.20)

        # forward positivity soft check
        grid = np.linspace(0.1, max(tenor.max(), 5.0), 300)
        fw = nss_forward(grid, b0, b1, b2, b3, t1, t2)
        if fw.min() < -EPSILON:
            # damp curvature
            b2 *= 0.5
            b3 *= 0.5

        return np.array([b0, b1, b2, b3, t1, t2], float)

    p2 = enforce(p)
    # re-clip
    p2 = np.minimum(np.maximum(p2, lb + 1e-8), ub - 1e-8)

    # Second pass
    res2 = least_squares(resid, p2, bounds=(lb, ub), max_nfev=50000, method="trf")
    return res2.x

# ============================================================
# STATS HELPERS
# ============================================================

# ============================================================
# STATS HELPERS
# ============================================================

def r2_score_manual(y, yhat):
    """
    Manual R² calculation without sklearn.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def nss_predict(t, p):
    return nss_yield(t, *p)


def loo_oos_r2(tenor, y, is_rf=False):
    """
    Leave-One-Out (LOO) cross-validation R².
    For each point i:
      - fit NSS on all points except i
      - predict y_hat at tenor[i]
    Then compute R² between y and y_hat across all points.
    """
    t = np.asarray(tenor, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 4:  # too few points for a meaningful LOO
        return float("nan")

    y_hat = np.zeros_like(y)

    for i in range(n):
        # mask for all points except i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        t_train = t[mask]
        y_train = y[mask]

        # fit with correct RF / Spread flag
        p_i = fit_nss(t_train, y_train, is_rf=is_rf)

        # predict at left-out tenor
        y_hat[i] = nss_predict(t[i], p_i)

    return r2_score_manual(y, y_hat)


def compute_stats(tenor, y, p, name="Curve", is_rf=False):
    """
    Compute:
      - In-sample R² using provided parameters p
      - Out-of-sample R² using LOO cross-validation
      - AIC, BIC (on full sample, using p)
      - Overfitting gap = R²_in - R²_OOS
    """
    t = np.asarray(tenor, float)
    y = np.asarray(y, float)

    # In-sample
    y_hat = nss_predict(t, p)
    if len(y) >= 2:
        r2_in = r2_score_manual(y, y_hat)
    else:
        r2_in = float("nan")

    # LOO out-of-sample
    r2_oos = loo_oos_r2(t, y, is_rf=is_rf)

    # AIC / BIC on full sample
    resid = y - y_hat
    sse = float(np.sum(resid ** 2))
    k = 6  # NSS has 6 parameters
    n = len(t)
    if n > k and sse > 0:
        aic = 2 * k + n * math.log(sse / n)
        bic = k * math.log(n) + n * math.log(sse / n)
    else:
        aic = float("nan")
        bic = float("nan")

    gap = (r2_in - r2_oos) if (not math.isnan(r2_in) and not math.isnan(r2_oos)) else float("nan")

    print(f"\n========== {name} ==========")
    print(f"Parameters: {p}")
    print(f"In-Sample   R²: {r2_in:.6f}" if not math.isnan(r2_in) else "In-Sample   R²: NA")
    print(f"OOS R² (LOO): {r2_oos:.6f}" if not math.isnan(r2_oos) else "OOS R² (LOO): NA")
    print(f"AIC: {aic:.6f}" if not math.isnan(aic) else "AIC: NA")
    print(f"BIC: {bic:.6f}" if not math.isnan(bic) else "BIC: NA")
    print(f"Overfitting gap (IS - OOS): {gap:.6f}" if not math.isnan(gap) else "Overfitting gap (IS - OOS): NA")

    return {
        "r2_in": r2_in,
        "r2_oos": r2_oos,
        "aic": aic,
        "bic": bic,
        "gap": gap,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_excel(EXCEL_PATH, sheet_name=DATA_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    tenor_raw = df[TENOR_COL].apply(normalize_tenor).values
    rf_raw = df[RF_COL].apply(parse_num).values
    sp_raw = df[SPREAD_COL].apply(parse_num).values

    tenor = tenor_raw.astype(float)
    rf = maybe_pct(rf_raw.astype(float))
    sp = maybe_pct(sp_raw.astype(float))

    mask = np.isfinite(tenor) & np.isfinite(rf) & np.isfinite(sp) & (tenor > 0)
    tenor, rf, sp = tenor[mask], rf[mask], sp[mask]

    # sort by tenor
    idx = np.argsort(tenor)
    tenor, rf, sp = tenor[idx], rf[idx], sp[idx]

# -----------------------------
    # Optional: load CIQ curve for overlay
    # -----------------------------
    ciq_tenor = None
    ciq_yield = None

    if CIQ_SHEET:
        try:
            df_ciq = pd.read_excel(EXCEL_PATH, sheet_name=CIQ_SHEET)
            df_ciq.columns = [str(c).strip() for c in df_ciq.columns]

            if (CIQ_TENOR_COL in df_ciq.columns) and (CIQ_YIELD_COL in df_ciq.columns):
                ciq_tenor_raw = df_ciq[CIQ_TENOR_COL].apply(normalize_tenor).values
                ciq_yield_raw = df_ciq[CIQ_YIELD_COL].apply(parse_num).values

                ciq_tenor = ciq_tenor_raw.astype(float)
                ciq_yield = maybe_pct(ciq_yield_raw.astype(float))

                mask_ciq = np.isfinite(ciq_tenor) & np.isfinite(ciq_yield) & (ciq_tenor > 0)
                ciq_tenor, ciq_yield = ciq_tenor[mask_ciq], ciq_yield[mask_ciq]

                if len(ciq_tenor) > 0:
                    idx_ciq = np.argsort(ciq_tenor)
                    ciq_tenor, ciq_yield = ciq_tenor[idx_ciq], ciq_yield[idx_ciq]
                    print(f"[INFO] Loaded CIQ curve with {len(ciq_tenor)} points.")
                else:
                    ciq_tenor, ciq_yield = None, None
                    print("[INFO] CIQ data had no valid rows; skipping CIQ overlay.")
            else:
                print("[INFO] CIQ columns not found; skipping CIQ overlay.")
        except Exception as e:
            print(f"[INFO] Could not load CIQ sheet '{CIQ_SHEET}': {e}")
    # -----------------------------
    # Optional: load CIQ curve for overlay
    # -----------------------------
    ciq_tenor = None
    ciq_yield = None

    if CIQ_SHEET:
        try:
            df_ciq = pd.read_excel(EXCEL_PATH, sheet_name=CIQ_SHEET)
            df_ciq.columns = [str(c).strip() for c in df_ciq.columns]

            if CIQ_TENOR_COL in df_ciq.columns and CIQ_YIELD_COL in df_ciq.columns:
                ciq_tenor_raw = df_ciq[CIQ_TENOR_COL].apply(normalize_tenor).values
                ciq_yield_raw = df_ciq[CIQ_YIELD_COL].apply(parse_num).values

                ciq_tenor = ciq_tenor_raw.astype(float)
                ciq_yield = maybe_pct(ciq_yield_raw.astype(float))

                mask_ciq = np.isfinite(ciq_tenor) & np.isfinite(ciq_yield) & (ciq_tenor > 0)
                ciq_tenor, ciq_yield = ciq_tenor[mask_ciq], ciq_yield[mask_ciq]

                # sort by tenor
                idx_ciq = np.argsort(ciq_tenor)
                ciq_tenor, ciq_yield = ciq_tenor[idx_ciq], ciq_yield[idx_ciq]

                print(f"[INFO] Loaded CIQ curve with {len(ciq_tenor)} points.")
            else:
                print("[INFO] CIQ cols not found; skipping CIQ overlay.")
        except Exception as e:
            print(f"[INFO] Could not load CIQ sheet '{CIQ_SHEET}': {e}")

    if len(tenor) < 3:
        raise ValueError("Not enough data points to fit NSS curves. Need at least 3.")

    # -----------------------------
    # Fit NSS curves
    # -----------------------------
    p_rf = fit_nss(tenor, rf, is_rf=True)
    p_sp = fit_nss(tenor, sp, is_rf=False)

    print("\n================ NSS PARAMETERS ================")
    print("Risk-Free NSS:", p_rf)
    print("Spread NSS   :", p_sp)

    # -----------------------------
    # Build smooth curve grid
    # -----------------------------
    T_max = max(30.0, float(tenor.max()))
    T_min = min(0.1, float(tenor.min()))
    T_grid = np.linspace(T_min, T_max, 600)

    rf_curve = nss_predict(T_grid, p_rf)
    sp_curve = nss_predict(T_grid, p_sp)
    sp_curve = np.maximum(sp_curve, 0.0)  # enforce non-negative spread

    total_curve = rf_curve + sp_curve

    # -----------------------------
    # Save full curves to CSV
    # -----------------------------
    df_full = pd.DataFrame({
        "tenor_years": T_grid,
        "rf_nss_yield": rf_curve,
        "spread_nss_yield": sp_curve,
        "total_yield": total_curve
    })
    df_full.to_csv(os.path.join(OUT_DIR, "full_curves.csv"), index=False)

    pd.DataFrame({"tenor_years": T_grid, "rf_nss_yield": rf_curve}).to_csv(
        os.path.join(OUT_DIR, "risk_free_curve.csv"), index=False
    )
    pd.DataFrame({"tenor_years": T_grid, "spread_nss_yield": sp_curve}).to_csv(
        os.path.join(OUT_DIR, "spread_curve.csv"), index=False
    )
    pd.DataFrame({"tenor_years": T_grid, "total_yield": total_curve}).to_csv(
        os.path.join(OUT_DIR, "total_yield_curve.csv"), index=False
    )

    # -----------------------------
    # Standard tenor export (3M..30Y)
    # -----------------------------
    std_tenors = np.array([
        3/12, 6/12,
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        12, 15, 20, 25, 30
    ], float)

    rf_std = nss_predict(std_tenors, p_rf)
    sp_std = nss_predict(std_tenors, p_sp)
    sp_std = np.maximum(sp_std, 0.0)
    total_std = rf_std + sp_std

    df_std = pd.DataFrame({
        "tenor_years": std_tenors,
        "tenor_label": [
            "3M","6M",
            "1Y","2Y","3Y","4Y","5Y","6Y","7Y","8Y","9Y","10Y",
            "12Y","15Y","20Y","25Y","30Y"
        ],
        "rf_yield": rf_std,
        "spread_yield": sp_std,
        "total_yield": total_std,
        "rf_yield_pct": rf_std * 100.0,
        "spread_yield_pct": sp_std * 100.0,
        "total_yield_pct": total_std * 100.0
    })

    excel_out_path = os.path.join(OUT_DIR, "C:/Users/FG977WW/OneDrive - EY/Automation Project/standard_tenor_curves.xlsx")
    df_std.to_excel(excel_out_path, index=False)
    print(f"\n[OK] Standard tenor curves exported to: {excel_out_path}")
    print(df_std)

    # -----------------------------
    # Model performance statistics
    # -----------------------------
    stats_rf = compute_stats(tenor, rf, p_rf, name="Risk-Free NSS", is_rf=True)
    stats_sp = compute_stats(tenor, sp, p_sp, name="Spread NSS", is_rf=False)

    print("\n================ OVERALL SUMMARY ================")
    print(f"RF overfitting gap (R2_in - R2_oos): {stats_rf['gap']}")
    print(f"SP overfitting gap (R2_in - R2_oos): {stats_sp['gap']}")

    # -----------------------------
    # Plot curves
    # -----------------------------
    plt.figure(figsize=(12, 7))
    plt.plot(tenor, rf, "o", label="RF Data")
    plt.plot(tenor, sp, "o", label="Spread Data")
    plt.plot(T_grid, rf_curve, label="RF NSS")
    plt.plot(T_grid, sp_curve, label="Spread NSS")
    plt.plot(T_grid, total_curve, label="Total Yield")

    # CIQ Plot
    if ciq_tenor is not None and ciq_yield is not None and len(ciq_tenor) > 0:
        plt.plot(ciq_tenor, ciq_yield, "x", label="CIQ Total Yield")

    plt.grid(True)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yield (decimal)")
    plt.title(f"NSS Curves ({CURRENCY})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "curves_plot.png"), dpi=300)
    plt.close()

    print("\n[OK] Curves plot saved to:", os.path.join(OUT_DIR, "curves_plot.png"))
    print("[DONE]")

if __name__ == "__main__":
    main()
