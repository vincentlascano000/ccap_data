# app.py
# CCAP — Methods A/B/C: PS baselines + CIF & Sales/CIF uplift + Scenario Shift + Coef Diagnostics
# • A: Baseline & drivers = AVERAGE of all historical same‑quarter QoQ factors (fixed, history‑only)
# • B: Baseline & drivers = LATEST same‑quarter QoQ factor (carry‑forward)
# • C: Baseline & drivers = TRUE ROLLING same‑quarter QoQ using last K entries (history + forecasted)
#     (True rolling appends the *realized PS factor* so it diverges from Method A.)
# • Coefficients (α, β_CIF, β_SPC) pooled across banks on residual PS growth vs an expanding same‑quarter baseline
# • Scenario Shift (±ppt) adds to PS growth each projected quarter
# • Header auto‑mapping for your CCAP column names (QUARTER, BANK, Purchase Sales (in Bn), Cards in Force (in Bn), Sales / CIF ('000))
# • Charts rendered via explicit if/else (prevents stray DeltaGenerator text blocks)
# • Coefficient Diagnostics — scatter + best‑fit lines (raw & residual views)
# • Delta tables REMOVED for brevity

import re
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="CCAP — Methods A/B/C (PS baselines + CIF/SPC uplift)", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")  # inclusive
BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
    "sales_per_cif_000": "Sales / CIF ('000)",
}

# =========================
# HEADER AUTO‑MAPPING (resilient to cases/punctuation)
# =========================
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[()\[\]’'`]", "", s)
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

HEADER_MAP = {
    "quarter": "quarter",                          # QUARTER
    "bank": "bank",                                # BANK
    "purchase sales in bn": "purchase_sales_bn",   # Purchase Sales (in Bn)
    "purchase sales bn": "purchase_sales_bn",
    "cards in force in bn": "cards_in_force_bn",   # Cards in Force (in Bn)
    "cards in force bn": "cards_in_force_bn",
    "sales cif 000": "sales_per_cif_000",          # Sales / CIF ('000)
    "sales per cif 000": "sales_per_cif_000",
}

def apply_header_map(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for col in df.columns:
        key = _canon(col)
        if key in HEADER_MAP:
            ren[col] = HEADER_MAP[key]
    return df.rename(columns=ren)

# =========================
# HELPERS
# =========================
def parse_quarter_token(value: str):
    """Parse '1Q23', 'Q1 2023', '2023 Q1', or date → (label, quarter-end Timestamp)."""
    if pd.isna(value): return None, pd.NaT
    s = str(value).strip().upper().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)
    m = re.match(r"^([1-4])Q(\d{2,4})$", s)
    if m:
        q, yy = int(m.group(1)), m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        per = pd.Period(freq="Q", year=year, quarter=q)
        return f"{year}Q{q}", per.to_timestamp(how="end")
    m = re.match(r"^Q([1-4])\s+(\d{2,4})$", s)
    if m:
        q, yy = int(m.group(1)), m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        per = pd.Period(freq="Q", year=year, quarter=q)
        return f"{year}Q{q}", per.to_timestamp(how="end")
    m = re.match(r"^(\d{4})\s*Q([1-4])$", s)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        per = pd.Period(freq="Q", year=year, quarter=q)
        return f"{year}Q{q}", per.to_timestamp(how="end")
    try:
        dt = pd.to_datetime(s, errors="raise")
        per = pd.Period(dt, freq="Q")
        return str(per), per.to_timestamp(how="end")
    except Exception:
        return None, pd.NaT

def to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    is_pct = s2.str.contains("%", regex=False, na=False)
    s2 = s2.str.replace("%", "", regex=False)
    out = pd.to_numeric(s2, errors="coerce")
    return pd.Series(np.where(is_pct, out/100.0, out), index=s.index)

def qoq_factors_by_quarter(series: pd.Series, periods: pd.Series) -> Dict[int, List[float]]:
    """Dict q -> list of QoQ factors f = X_t / X_{t-1}, grouped by quarter-of-year of t."""
    s_per = periods.dt.to_period("Q")
    s = pd.Series(series.values, index=s_per).sort_index()
    f = (s / s.shift(1)).dropna()
    out = {1: [], 2: [], 3: [], 4: []}
    for p, val in f.items():
        if np.isfinite(val) and val > 0:
            out[p.quarter].append(float(val))
    return out

def latest_same_quarter_qoq(series: pd.Series, periods: pd.Series) -> Dict[int, float]:
    """Latest observed QoQ factor per quarter-of-year (fallback to latest overall)."""
    s_per = periods.dt.to_period("Q")
    s = pd.Series(series.values, index=s_per).sort_index()
    f = (s / s.shift(1)).dropna()
    latest_overall = float(f.iloc[-1]) if not f.empty and np.isfinite(f.iloc[-1]) and f.iloc[-1] > 0 else 1.0
    d = {}
    for q in (1,2,3,4):
        fq = [float(v) for p, v in f.items() if p.quarter == q and np.isfinite(v) and v > 0]
        d[q] = fq[-1] if fq else latest_overall
        if not np.isfinite(d[q]) or d[q] <= 0:
            d[q] = 1.0
    return d

# =========================
# UI
# =========================
st.title("CCAP — PS Baselines with CIF & Sales/CIF Uplift (Methods A, B & C)")
st.caption(f"Forecast end: **{str(TARGET_END)}**")

st.sidebar.header("General")
round_dec = st.sidebar.selectbox("Round decimals (levels)", options=[0,1,2], index=1)

st.sidebar.header("Scenario")
scenario = st.sidebar.radio("Scenario", ["Pessimistic","Realistic","Optimistic"], index=1, horizontal=True)
scenario_shift_ppt = st.sidebar.slider("Scenario shift (±ppt added to PS growth each projected quarter)", 0.0, 10.0, 1.5, 0.1)
scenario_adj_prop = (scenario_shift_ppt/100.0) * (1 if scenario=="Optimistic" else (-1 if scenario=="Pessimistic" else 0))

st.sidebar.header("Method C (True rolling window)")
rolling_window_years = st.sidebar.slider("Last K same‑quarter entries to average (history + forecasted)", 3, 8, 6, 1)

# =========================
# LOAD & VALIDATE DATA (with header auto‑mapping)
# =========================
try:
    raw = pd.read_csv(RAW_URL, engine="python")
except Exception:
    raw = pd.read_csv(RAW_URL, engine="python", encoding="utf-8-sig")

df = apply_header_map(raw.copy())

has_time = ("quarter_dt" in df.columns) or ("quarter" in df.columns)
required = ["bank", "purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]
missing = [c for c in required if c not in df.columns]
if missing or not has_time:
    st.error(
        "Missing required columns after auto-mapping.\n\n"
        f"Required: {required} and either 'quarter_dt' or 'quarter'.\n"
        f"Found (source): {list(raw.columns)}\n"
        f"Mapped columns: {list(df.columns)}\n\n"
        "Tip: Headers like 'QUARTER', 'BANK', 'Purchase Sales (in Bn)', "
        "'Cards in Force (in Bn)', and \"Sales / CIF ('000)\" are supported."
    )
    st.stop()

# Make quarter_dt if needed
if "quarter_dt" not in df.columns:
    parsed = df["quarter"].apply(parse_quarter_token)
    df["quarter_dt"] = parsed.apply(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)

# Coerce numerics
for c in ["purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]:
    df[c] = to_numeric(df[c])

panel = (
    df[["bank","quarter_dt","purchase_sales_bn","cards_in_force_bn","sales_per_cif_000"]]
    .dropna(subset=["bank","quarter_dt"])
    .sort_values(["bank","quarter_dt"])
    .reset_index(drop=True)
)

# Bank selection
banks_all = sorted(panel["bank"].dropna().unique().tolist(),
                   key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
banks_pick = st.multiselect("Banks to include", options=banks_all, default=banks_all)
if not banks_pick:
    st.info("Select at least one bank.")
    st.stop()

# =========================
# COEFFICIENT ESTIMATION — pooled on residual PS growth vs expanding same‑quarter baseline
# =========================
def fit_uplift_coefs(panel_bank: pd.DataFrame) -> Tuple[float,float,float,pd.DataFrame]:
    """
    Pooled OLS on residual PS growth vs Δ%CIF and Δ%SPC (Sales/CIF).
    Baseline for residualization: expanding same‑quarter average of d_ps.
    """
    g = panel_bank.sort_values(["bank","quarter_dt"]).copy()
    g["per"] = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = g["per"].apply(lambda p: p.quarter)

    # QoQ % changes
    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    # Expanding same‑quarter baseline for PS growth
    base_g = []
    for b, gb in g.groupby("bank"):
        pools = {1:[],2:[],3:[],4:[]}
        q_b   = gb["qtr"].values
        dps   = gb["d_ps"].values
        base_series = []
        for i in range(len(gb)):
            q = q_b[i]
            pool = pools[q]
            g_base = float(np.mean(pool)) if len(pool) > 0 else np.nan
            base_series.append(g_base)
            if pd.notna(dps[i]):
                pools[q].append(dps[i])
        base_g.append(pd.Series(base_series, index=gb.index))
    g["g_base"] = pd.concat(base_g).sort_index()

    # Residual growth
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps","d_cif","d_spc","per","bank"]).copy()
    if fit.empty:
        return 0.0, 0.0, 0.0, pd.DataFrame()

    # Gentle winsorize to reduce single‑quarter leverage
    for col in ["r_ps","d_cif","d_spc"]:
        fit[col] = fit[col].clip(lower=-0.5, upper=0.5)

    X = np.column_stack([np.ones(len(fit)), fit["d_cif"].to_numpy(), fit["d_spc"].to_numpy()])
    y = fit["r_ps"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, b_cif, b_spc = float(beta[0]), float(beta[1]), float(beta[2])
    return b_cif, b_spc, intercept, fit

b_cif, b_spc, b_int, fit_df = fit_uplift_coefs(panel[panel["bank"].isin(banks_pick)])

# --- Coefficient Diagnostics helper frames (raw and residual views) ---
# --- Coefficient Diagnostics helper: build from selected panel (robust) ---
def make_coef_diag_frames_from_panel(panel_sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build diagnostic frames for coefficient visuals directly from the selected banks' panel.
    Returns:
      raw_df   : bank, quarter_dt, d_ps_pct, d_cif_pct, d_spc_pct
      resid_df : bank, quarter_dt, r_ps_pct, d_cif_pct, d_spc_pct
                 (residual r_ps = d_ps - g_base where g_base is expanding same-quarter average)
    This mirrors how coefficients are estimated, but isn't limited by the regression's final sample.
    """
    if panel_sel is None or panel_sel.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = panel_sel.sort_values(["bank", "quarter_dt"]).copy()
    g["per"] = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = g["per"].apply(lambda p: p.quarter)

    # QoQ % changes (proportions)
    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    # Expanding same‑quarter baseline for PS growth (proportions)
    base_g = []
    for b, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        q_b   = gb["qtr"].to_numpy()
        dps   = gb["d_ps"].to_numpy()
        series = []
        for i in range(len(gb)):
            q = q_b[i]
            pool = pools[q]
            g_base = float(np.mean(pool)) if len(pool) > 0 else np.nan
            series.append(g_base)
            if pd.notna(dps[i]):
                pools[q].append(dps[i])
        base_g.append(pd.Series(series, index=gb.index))
    g["g_base"] = pd.concat(base_g).sort_index()

    # Residual growth
    g["r_ps"] = g["d_ps"] - g["g_base"]

    # RAW growths frame (drop NaNs; convert to % for display)
    raw_df = g[["bank", "per", "d_ps", "d_cif", "d_spc"]].dropna().copy()
    raw_df["quarter_dt"] = raw_df["per"].dt.to_timestamp()
    raw_df["d_ps_pct"]  = raw_df["d_ps"]  * 100.0
    raw_df["d_cif_pct"] = raw_df["d_cif"] * 100.0
    raw_df["d_spc_pct"] = raw_df["d_spc"] * 100.0
    raw_df = raw_df[["bank", "quarter_dt", "d_ps_pct", "d_cif_pct", "d_spc_pct"]]

    # RESIDUAL frame (what OLS fits)
    resid_df = g[["bank", "per", "r_ps", "d_cif", "d_spc"]].dropna().copy()
    resid_df["quarter_dt"] = resid_df["per"].dt.to_timestamp()
    resid_df["r_ps_pct"]  = resid_df["r_ps"] * 100.0
    resid_df["d_cif_pct"] = resid_df["d_cif"] * 100.0
    resid_df["d_spc_pct"] = resid_df["d_spc"] * 100.0
    resid_df = resid_df[["bank", "quarter_dt", "r_ps_pct", "d_cif_pct", "d_spc_pct"]]

    return raw_df, resid_df

# Build diagnostics from the currently selected banks
panel_selected = panel[panel["bank"].isin(banks_pick)].copy()
raw_df, resid_df = make_coef_diag_frames_from_panel(panel_selected)

with st.expander("Diagnostics status (data availability)", expanded=False):
    st.write(f"RAW points:      {len(raw_df)}")
    st.write(f"RESIDUAL points: {len(resid_df)}")
    if len(raw_df) == 0:
        st.info("RAW view has no rows. Check that selected banks have enough quarters for QoQ % changes.")
    if len(resid_df) == 0:
        st.info("RESIDUAL view has no rows. Often this happens when the first same‑quarter has no baseline yet; try including more banks.")

# =========================
# PROJECTIONS: Methods A/B/C
# =========================
def average_same_quarter_factor(hist_dict: Dict[int, List[float]], q: int) -> float:
    vals = hist_dict.get(q, [])
    if not vals: return 1.0
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if not vals: return 1.0
    m = float(np.mean(vals))
    return m if (np.isfinite(m) and m > 0) else 1.0

def project_method_A(bank_df: pd.DataFrame, scenario_adj_prop: float) -> pd.DataFrame:
    """Method A: Avg same‑quarter QoQ (history‑only) + uplift + scenario shift."""
    gb = bank_df.sort_values("quarter_dt").copy()
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cards_in_force_bn"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["sales_per_cif_000"], gb["quarter_dt"])

    level_ps  = float(gb.iloc[-1]["purchase_sales_bn"])
    level_cif = float(gb.iloc[-1]["cards_in_force_bn"])
    level_spc = float(gb.iloc[-1]["sales_per_cif_000"])

    rows = []
    for h in range(1, H+1):
        t = last_per + h; q = t.quarter
        prev_ps, prev_cif, prev_spc = level_ps, level_cif, level_spc

        f_ps_base = average_same_quarter_factor(hist_ps, q);     g_base = f_ps_base - 1.0
        f_cif     = average_same_quarter_factor(hist_cif, q);    d_cif = f_cif - 1.0
        f_spc     = average_same_quarter_factor(hist_spc, q);    d_spc = f_spc - 1.0

        uplift  = b_int + b_cif*d_cif + b_spc*d_spc
        g_total = g_base + uplift + scenario_adj_prop

        level_ps  *= (1.0 + g_total)
        level_cif *= (1.0 + d_cif)
        level_spc *= (1.0 + d_spc)

        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc_pct = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan

        rows.append({
            "quarter": str(t), "bank": gb["bank"].iloc[0], "method": "Method A (Avg QoQ + uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "delta_purchase_sales_pct": round(d_ps_pct,2)  if pd.notna(d_ps_pct)  else None,
            "delta_cif_pct":            round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_sales_per_cif_000_pct":  round(d_spc_pct,2) if pd.notna(d_spc_pct) else None,
            "ps_uplift_pp":             round(uplift*100.0, 2),
            "ps_uplift_multiplier":     round(1.0 + uplift, 4),
            "scenario_shift_pp":        round(scenario_adj_prop*100.0, 2),
        })
    return pd.DataFrame(rows)

def project_method_B(bank_df: pd.DataFrame, scenario_adj_prop: float) -> pd.DataFrame:
    """Method B: Latest same‑quarter QoQ + uplift + scenario shift."""
    gb = bank_df.sort_values("quarter_dt").copy()
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    latest_ps  = latest_same_quarter_qoq(gb["purchase_sales_bn"], gb["quarter_dt"])
    latest_cif = latest_same_quarter_qoq(gb["cards_in_force_bn"], gb["quarter_dt"])
    latest_spc = latest_same_quarter_qoq(gb["sales_per_cif_000"], gb["quarter_dt"])

    level_ps  = float(gb.iloc[-1]["purchase_sales_bn"])
    level_cif = float(gb.iloc[-1]["cards_in_force_bn"])
    level_spc = float(gb.iloc[-1]["sales_per_cif_000"])

    rows = []
    for h in range(1, H+1):
        t = last_per + h; q = t.quarter
        prev_ps, prev_cif, prev_spc = level_ps, level_cif, level_spc

        f_ps_base = latest_ps.get(q, 1.0);  f_ps_base = f_ps_base if (np.isfinite(f_ps_base) and f_ps_base>0) else 1.0
        g_base = f_ps_base - 1.0
        f_cif  = latest_cif.get(q, 1.0);    f_cif  = f_cif  if (np.isfinite(f_cif)  and f_cif>0)  else 1.0
        f_spc  = latest_spc.get(q, 1.0);    f_spc  = f_spc  if (np.isfinite(f_spc)  and f_spc>0)  else 1.0
        d_cif = f_cif - 1.0
        d_spc = f_spc - 1.0

        uplift  = b_int + b_cif*d_cif + b_spc*d_spc
        g_total = g_base + uplift + scenario_adj_prop

        level_ps  *= (1.0 + g_total)
        level_cif *= (1.0 + d_cif)
        level_spc *= (1.0 + d_spc)

        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc_pct = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan

        rows.append({
            "quarter": str(t), "bank": gb["bank"].iloc[0], "method": "Method B (Latest QoQ + uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "delta_purchase_sales_pct": round(d_ps_pct,2)  if pd.notna(d_ps_pct)  else None,
            "delta_cif_pct":            round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_sales_per_cif_000_pct":  round(d_spc_pct,2) if pd.notna(d_spc_pct) else None,
            "ps_uplift_pp":             round(uplift*100.0, 2),
            "ps_uplift_multiplier":     round(1.0 + uplift, 4),
            "scenario_shift_pp":        round(scenario_adj_prop*100.0, 2),
        })
    return pd.DataFrame(rows)

def project_method_C(bank_df: pd.DataFrame, scenario_adj_prop: float, K: int) -> pd.DataFrame:
    """
    Method C (True Rolling QoQ + uplift):
      • Baseline PS growth = average of last K same‑quarter *realized* PS factors (history + forecasted‑to‑date).
      • Driver growths (Δ%CIF, Δ%SPC) = average of last K same‑quarter factors (history + forecasted).
      • Append the *realized total PS factor* (PS_t / PS_{t-1}) into PS rolling pool so next same‑quarter adapts.
    """
    gb = bank_df.sort_values("quarter_dt").copy()
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cards_in_force_bn"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["sales_per_cif_000"], gb["quarter_dt"])

    fore_ps  = {1: [], 2: [], 3: [], 4: []}
    fore_cif = {1: [], 2: [], 3: [], 4: []}
    fore_spc = {1: [], 2: [], 3: [], 4: []}

    level_ps  = float(gb.iloc[-1]["purchase_sales_bn"])
    level_cif = float(gb.iloc[-1]["cards_in_force_bn"])
    level_spc = float(gb.iloc[-1]["sales_per_cif_000"])

    rows = []
    for h in range(1, H+1):
        t = last_per + h
        q = t.quarter

        prev_ps, prev_cif, prev_spc = level_ps, level_cif, level_spc

        # PS baseline factor: average of last K *realized* PS factors (hist + fore)
        pool_ps_q = hist_ps[q] + fore_ps[q]
        use_ps = pool_ps_q[-K:] if len(pool_ps_q) >= K else pool_ps_q
        f_ps_base = float(np.mean(use_ps)) if use_ps else 1.0
        f_ps_base = f_ps_base if (np.isfinite(f_ps_base) and f_ps_base > 0) else 1.0
        g_base = f_ps_base - 1.0

        # Drivers: same rolling logic on CIF/SPC factors
        pool_cif_q = hist_cif[q] + fore_cif[q]
        use_cif = pool_cif_q[-K:] if len(pool_cif_q) >= K else pool_cif_q
        f_cif = float(np.mean(use_cif)) if use_cif else 1.0
        f_cif = f_cif if (np.isfinite(f_cif) and f_cif > 0) else 1.0
        d_cif = f_cif - 1.0

        pool_spc_q = hist_spc[q] + fore_spc[q]
        use_spc = pool_spc_q[-K:] if len(pool_spc_q) >= K else pool_spc_q
        f_spc = float(np.mean(use_spc)) if use_spc else 1.0
        f_spc = f_spc if (np.isfinite(f_spc) and f_spc > 0) else 1.0
        d_spc = f_spc - 1.0

        # Uplift + Scenario
        uplift  = b_int + b_cif*d_cif + b_spc*d_spc
        g_total = g_base + uplift + scenario_adj_prop

        # Evolve levels
        level_ps  *= (1.0 + g_total)
        level_cif *= (1.0 + d_cif)
        level_spc *= (1.0 + d_spc)

        # Append REALIZED factors to rolling pools
        f_ps_realized = (level_ps / prev_ps) if prev_ps else 1.0
        if np.isfinite(f_ps_realized) and f_ps_realized > 0:
            fore_ps[q].append(float(f_ps_realized))
        if np.isfinite(1.0 + d_cif) and (1.0 + d_cif) > 0:
            fore_cif[q].append(float(1.0 + d_cif))
        if np.isfinite(1.0 + d_spc) and (1.0 + d_spc) > 0:
            fore_spc[q].append(float(1.0 + d_spc))

        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc_pct = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan

        rows.append({
            "quarter": str(t), "bank": gb["bank"].iloc[0], "method": "Method C (True Rolling QoQ + uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "delta_purchase_sales_pct": round(d_ps_pct,2)  if pd.notna(d_ps_pct)  else None,
            "delta_cif_pct":            round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_sales_per_cif_000_pct":  round(d_spc_pct,2) if pd.notna(d_spc_pct) else None,
            "ps_uplift_pp":             round(uplift*100.0, 2),
            "ps_uplift_multiplier":     round(1.0 + uplift, 4),
            "scenario_shift_pp":        round(scenario_adj_prop*100.0, 2),
        })
    return pd.DataFrame(rows)

# =========================
# RUN METHODS A/B/C
# =========================
proj_A_frames, proj_B_frames, proj_C_frames, hist_frames = [], [], [], []
for b in banks_pick:
    gb = panel[panel["bank"] == b][["bank","quarter_dt","purchase_sales_bn","cards_in_force_bn","sales_per_cif_000"]]
    if gb.shape[0] < 3:
        continue
    hist_frames.append(gb.assign(method="Actual"))
    pa = project_method_A(gb, scenario_adj_prop)
    pb = project_method_B(gb, scenario_adj_prop)
    pc = project_method_C(gb, scenario_adj_prop, K=int(rolling_window_years))
    if not pa.empty: proj_A_frames.append(pa)
    if not pb.empty: proj_B_frames.append(pb)
    if not pc.empty: proj_C_frames.append(pc)

proj_A = pd.concat(proj_A_frames, ignore_index=True) if proj_A_frames else pd.DataFrame()
proj_B = pd.concat(proj_B_frames, ignore_index=True) if proj_B_frames else pd.DataFrame()
proj_C = pd.concat(proj_C_frames, ignore_index=True) if proj_C_frames else pd.DataFrame()

# Round level columns per UI
for dfp in [proj_A, proj_B, proj_C]:
    if not dfp.empty:
        for col in ["projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn"]:
            dfp[col] = dfp[col].round(int(round_dec))

# =========================
# COEFFICIENTS PANEL
# =========================
with st.expander("Coefficients used for uplift (pooled, residualized vs PS baseline)", expanded=True):
    st.markdown(
        f"""
- **Intercept (α)**: **{b_int:.4f}**  
- **β (Δ% CIF)**: **{b_cif:.4f}**  
- **β (Δ% Sales/CIF)**: **{b_spc:.4f}**  

**Scenario:** **{scenario}** (adds **{scenario_shift_ppt:.1f} pp** to PS growth per projected quarter).  
**Method C rolling window (K):** **{int(rolling_window_years)}** same‑quarter factors (history + forecasted).
"""
    )

# =========================
# CHARTS — A/B/C overlays (explicit if/else to avoid stray DeltaGenerator repr)
# =========================
color_scale = alt.Scale(scheme='tableau10')

def chart_method(method_name: str, metric_code: str):
    metric_label = FRIENDLY.get(metric_code, metric_code)
    overlays = []

    # Actual
    hist_all = pd.concat(hist_frames, ignore_index=True)
    actual = hist_all[["bank","quarter_dt",metric_code]].dropna().copy()
    actual = actual.rename(columns={metric_code:"value"}).assign(scenario="Actual")
    overlays.append(actual[["bank","quarter_dt","value","scenario"]])

    # Projections
    mapping = {
        "Method A (Avg QoQ + uplift)": proj_A,
        "Method B (Latest QoQ + uplift)":  proj_B,
        "Method C (True Rolling QoQ + uplift)":  proj_C,
    }
    dfp = mapping.get(method_name, pd.DataFrame())
    if dfp.empty: return None

    rename_map = {
        "projected_cif_bn":"cards_in_force_bn",
        "projected_sales_per_cif_000":"sales_per_cif_000",
        "projected_purchase_sales_bn":"purchase_sales_bn",
    }
    use = dfp.rename(columns=rename_map)
    use["quarter_dt"] = use["quarter"].apply(lambda s: pd.Period(s, freq="Q").to_timestamp(how="end"))
    use = use[["bank","quarter_dt", metric_code, "method"]].rename(columns={metric_code:"value","method":"scenario"})
    overlays.append(use)

    overlay = pd.concat(overlays, ignore_index=True)

    actual_line = (
        alt.Chart(overlay).transform_filter(alt.datum.scenario == "Actual")
          .mark_line(point=True, strokeWidth=2)
          .encode(
              x=alt.X("quarter_dt:T", title="Quarter"),
              y=alt.Y("value:Q", title=metric_label),
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"),
                       alt.Tooltip("quarter_dt:T", title="Quarter"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"),
                       alt.Tooltip("scenario:N")]
          ).properties(height=360)
    )
    proj_line = (
        alt.Chart(overlay).transform_filter(alt.datum.scenario != "Actual")
          .mark_line(point=False, strokeWidth=2, strokeDash=[6,4])
          .encode(
              x="quarter_dt:T", y="value:Q",
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"),
                       alt.Tooltip("quarter_dt:T", title="Quarter"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"),
                       alt.Tooltip("scenario:N")]
          ).properties(height=360)
    )
    return alt.layer(actual_line, proj_line)

metric_pick = st.selectbox(
    "Metric to chart",
    options=[FRIENDLY["purchase_sales_bn"], FRIENDLY["cards_in_force_bn"], FRIENDLY["sales_per_cif_000"]],
    index=0
)
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_pick]

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Method A — Avg same‑quarter QoQ + uplift")
    chA = chart_method("Method A (Avg QoQ + uplift)", metric_code)
    if chA is not None:
        st.altair_chart(chA, use_container_width=True)
    else:
        st.info("No projections for Method A.")
with col2:
    st.subheader("Method B — Latest same‑quarter QoQ + uplift")
    chB = chart_method("Method B (Latest QoQ + uplift)", metric_code)
    if chB is not None:
        st.altair_chart(chB, use_container_width=True)
    else:
        st.info("No projections for Method B.")
with col3:
    st.subheader("Method C — True Rolling same‑quarter QoQ + uplift")
    chC = chart_method("Method C (True Rolling QoQ + uplift)", metric_code)
    if chC is not None:
        st.altair_chart(chC, use_container_width=True)
    else:
        st.info("No projections for Method C.")

# =========================
# COEFFICIENT DIAGNOSTICS — scatterplots with best‑fit lines
# =========================
st.subheader("Coefficient Diagnostics — Scatterplots with Best‑Fit Line")

def scatter_with_line(df: pd.DataFrame, x_col: str, y_col: str,
                      x_title: str, y_title: str, color_scale=alt.Scale(scheme='tableau10')):
    if df is None or df.empty:
        return None
    base = alt.Chart(df).mark_circle(size=60, opacity=0.55).encode(
        x=alt.X(f"{x_col}:Q", title=x_title),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        color=alt.Color("bank:N", legend=None, scale=color_scale),
        tooltip=[
            alt.Tooltip("bank:N"),
            alt.Tooltip("quarter_dt:T", title="Quarter"),
            alt.Tooltip(x_col, title=x_title, format=",.2f"),
            alt.Tooltip(y_col, title=y_title, format=",.2f"),
        ]
    )
    line = alt.Chart(df).transform_regression(x_col, y_col).mark_line(
        color="#e31a1c", strokeWidth=2
    )
    return (base + line).properties(height=320)

# Row 1: RAW growths (intuitive view)
st.markdown("**Raw growths (QoQ %): Δ% Purchase Sales vs Drivers**")
colR1, colR2 = st.columns(2)
with colR1:
    ch_raw_cif = scatter_with_line(raw_df, "d_cif_pct", "d_ps_pct",
                                   "Δ% CIF (QoQ, %)", "Δ% Purchase Sales (QoQ, %)")
    if ch_raw_cif is not None:
        st.altair_chart(ch_raw_cif, use_container_width=True)
    else:
        st.info("Not enough data to plot Δ%PS vs Δ%CIF.")
with colR2:
    ch_raw_spc = scatter_with_line(raw_df, "d_spc_pct", "d_ps_pct",
                                   "Δ% Sales/CIF (QoQ, %)", "Δ% Purchase Sales (QoQ, %)")
    if ch_raw_spc is not None:
        st.altair_chart(ch_raw_spc, use_container_width=True)
    else:
        st.info("Not enough data to plot Δ%PS vs Δ%Sales/CIF.")

# Row 2: RESIDUAL growth (what OLS actually fits)
st.markdown("**Residual growth (QoQ %) vs Drivers — what the regression fits**")
colE1, colE2 = st.columns(2)
with colE1:
    ch_resid_cif = scatter_with_line(resid_df, "d_cif_pct", "r_ps_pct",
                                     "Δ% CIF (QoQ, %)", "Residual Δ% Purchase Sales (QoQ, %)")
    if ch_resid_cif is not None:
        st.altair_chart(ch_resid_cif, use_container_width=True)
    else:
        st.info("Not enough data to plot residual Δ%PS vs Δ%CIF.")
with colE2:
    ch_resid_spc = scatter_with_line(resid_df, "d_spc_pct", "r_ps_pct",
                                     "Δ% Sales/CIF (QoQ, %)", "Residual Δ% Purchase Sales (QoQ, %)")
    if ch_resid_spc is not None:
        st.altair_chart(ch_resid_spc, use_container_width=True)
    else:
        st.info("Not enough data to plot residual Δ%PS vs Δ%Sales/CIF.")
