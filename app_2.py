# app.py
# CCAP — Methods A/B + NEW Method C (Baseline × Driver-Uplift)
# Method A: Rolling seasonal QoQ average (expanding window by quarter-of-year) for CIF & SPC and PS = scale*CIF*SPC
# Method B: Latest same-quarter QoQ carry-forward for CIF & SPC and PS = scale*CIF*SPC
# Method C: PS baseline (rolling QoQ seasonal) + uplift from Δ%CIF and Δ%Balances (pooled OLS on growth)
# - Tables include QoQ % deltas (as percent strings)
# - Charts: one per method for a chosen metric + Method C driver scatter plots with best-fit lines

import re
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")  # inclusive
BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "quarter": ["quarter", "qtr", "period", "quarter_str", "q", "date"],
    "bank":    ["bank", "issuer", "bank_name", "issuer_name", "issuer bank"],

    "purchase_sales_bn": [
        "purchase_sales_bn","purchase_sales","sales",
        "purchase volume (bn)","purchase sales (in bn)","purchase_sales_bil","purchase_sales_billion"
    ],
    "cards_in_force_bn": [
        "cards_in_force_bn","cards_in_force","cif (bn)","cif","cif_in_bn","cif_bn","cards","c.i.f."
    ],
    "balances_bn": [
        "balances_bn","balances","balance (bn)","balances (in bn)","total_balance_bn","balances_bil","balances_billion"
    ],
    "sales_per_cif_000": [
        "sales_per_cif_000","sales_per_cif","sales per cif (000)","sales/cif (000)","sales_cif"
    ],
}

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
    "balances_bn": "Balances (Bn)",
    "sales_per_cif_000": "Sales / CIF ('000)",
}

# =========================
# HELPERS
# =========================
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[()\[\]%]", "", s)
    s = re.sub(r"[\\/]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort auto-map headers to canonical names."""
    csrc = {_canon(c): c for c in df.columns}
    ren = {}
    for target, syns in COLUMN_SYNONYMS.items():
        for s in syns:
            sc = _canon(s)
            if sc in csrc and target not in ren:
                ren[csrc[sc]] = target
                break
    return df.rename(columns=ren)

def parse_quarter_token(value: str):
    """Parse common quarter strings -> (label, quarter-end Timestamp)."""
    if pd.isna(value): return None, pd.NaT
    s = str(value).strip().upper().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)
    m = re.match(r"^([1-4])Q(\d{2,4})$", s)  # 1Q23 / 4Q2025
    if m:
        q, yy = int(m.group(1)), m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        per = pd.Period(freq="Q", year=year, quarter=q)
        return f"{year}Q{q}", per.to_timestamp(how="end")
    m = re.match(r"^Q([1-4])\s+(\d{2,4})$", s)  # Q1 2023
    if m:
        q, yy = int(m.group(1)), m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        per = pd.Period(freq="Q", year=year, quarter=q)
        return f"{year}Q{q}", per.to_timestamp(how="end")
    m = re.match(r"^(\d{4})\s*Q([1-4])$", s)  # 2023 Q1
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
    if pd.api.types.is_numeric_dtype(s):  # already numeric
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    is_pct = s2.str.contains("%", regex=False, na=False)
    s2 = s2.str.replace("%", "", regex=False)
    out = pd.to_numeric(s2, errors="coerce")
    return pd.Series(np.where(is_pct, out/100.0, out), index=s.index)

@st.cache_data(ttl=600, show_spinner=False)
def load_raw_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, engine="python")
    except Exception:
        return pd.read_csv(url, engine="python", encoding="utf-8-sig")

def candidate_cols(df: pd.DataFrame, names_or_keywords: List[str]) -> List[str]:
    want = {_canon(x) for x in names_or_keywords}
    out, seen = [], set()
    for col in df.columns:
        cc = _canon(col)
        if cc in want or any(k in cc for k in want):
            if col not in seen:
                out.append(col); seen.add(col)
    return out

def format_percent_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Render numeric percent columns as strings with '%' and 2 decimals."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: (f"{x:.2f}%" if pd.notna(x) else None))
    return out

# QoQ factor helpers
def qoq_factors_by_quarter(series: pd.Series, periods: pd.Series) -> Dict[int, List[float]]:
    """Return dict quarter->list of QoQ factors f = X_t / X_{t-1} for quarter(t)==q."""
    s_per = periods.dt.to_period("Q")
    s = pd.Series(series.values, index=s_per).sort_index()
    f = (s / s.shift(1)).dropna()
    out = {1: [], 2: [], 3: [], 4: []}
    for p, val in f.items():
        if np.isfinite(val) and val > 0:
            out[p.quarter].append(float(val))
    return out

def latest_same_quarter_qoq(series: pd.Series, periods: pd.Series) -> Dict[int, float]:
    """Latest observed QoQ factor per quarter-of-year; fallback to latest overall."""
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

def robust_mean(vals: List[float], median=False) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 1.0
    return float(np.median(arr) if median else np.mean(arr))

# =========================
# UI
# =========================
st.set_page_config(page_title="CCAP — QoQ Methods + Method C", layout="wide")
st.title("CCAP — CIF × Sales/CIF and Baseline×Uplift: Methods A/B/C")
st.caption(f"Forecast end: **{str(TARGET_END)}**")

st.sidebar.header("Data & Output")
round_dec = st.sidebar.selectbox("Round decimals (levels)", options=[0,1,2], index=1)

st.sidebar.header("Method A (Rolling QoQ)")
base_window_years = st.sidebar.slider("Initial lookback (years) per quarter", 1, 4, 2, 1)
use_median_A = st.sidebar.checkbox("Use median (instead of mean) for A", value=False)

st.sidebar.header("Method C (Baseline×Uplift)")
uplift_cap_pct_pts = st.sidebar.slider("Cap uplift contribution (±ppt)", 2.0, 15.0, 8.0, 1.0)

# =========================
# LOAD & NORMALIZE
# =========================
try:
    raw = load_raw_csv(RAW_URL)
except Exception as e:
    st.error(f"Failed to read CSV from GitHub: {e}")
    st.stop()

df = map_columns(raw)

# Quarter parsing
if "quarter_dt" not in df.columns:
    qsrc = "quarter" if "quarter" in df.columns else None
    if qsrc is None:
        st.error("Missing 'quarter' or 'quarter_dt' column.")
        st.write("Detected columns:", list(df.columns))
        st.stop()
    df["quarter_dt"] = df[qsrc].apply(parse_quarter_token).apply(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)

# Bank
if "bank" not in df.columns:
    st.error("Missing 'bank' column after mapping.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Column mapping for CIF/SPC/Balances/Purchase Sales
st.sidebar.header("Column mapping (if needed)")
def ensure_col(name_key: str, label: str) -> str:
    if name_key in df.columns:
        return name_key
    cands = candidate_cols(df, COLUMN_SYNONYMS[name_key]) or list(df.columns)
    return st.sidebar.selectbox(f"Select column for {label}", options=cands, index=0)

cif_col = ensure_col("cards_in_force_bn", FRIENDLY["cards_in_force_bn"])
spc_col = ensure_col("sales_per_cif_000", FRIENDLY["sales_per_cif_000"])
bal_col = ensure_col("balances_bn", FRIENDLY["balances_bn"])
if "purchase_sales_bn" not in df.columns:
    cands = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"])
    if cands:
        sel = st.sidebar.selectbox("Select column for Purchase Sales", options=cands, index=0)
        df["purchase_sales_bn"] = df[sel]

# Coerce numeric
for c in [cif_col, spc_col, bal_col, "purchase_sales_bn"]:
    df[c] = to_numeric(df[c])

panel = (df[["bank","quarter_dt", cif_col, spc_col, bal_col, "purchase_sales_bn"]]
         .dropna(subset=["bank","quarter_dt"])
         .sort_values(["bank","quarter_dt"])
         .reset_index(drop=True))

# Bank selection
banks_all = sorted(panel["bank"].dropna().unique().tolist(),
                   key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
banks_pick = st.multiselect("Banks to include", options=banks_all, default=banks_all)
if not banks_pick:
    st.info("Select at least one bank.")
    st.stop()

# =========================
# METHOD B — Latest same-quarter QoQ (CIF & SPC → PS via identity)
# =========================
def project_latest_qoq_identity(bank_df: pd.DataFrame) -> pd.DataFrame:
    g = bank_df.sort_values("quarter_dt").copy()
    per = g["quarter_dt"].dt.to_period("Q")
    last_dt  = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0:
        return pd.DataFrame()

    # splice scaling for PS identity
    cif_last = float(g.iloc[-1][cif_col]); spc_last = float(g.iloc[-1][spc_col])
    denom = cif_last * spc_last if (cif_last and spc_last) else None
    scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom and denom != 0 else 1.0

    latest_cif = latest_same_quarter_qoq(g[cif_col], g["quarter_dt"])
    latest_spc = latest_same_quarter_qoq(g[spc_col], g["quarter_dt"])
    level_cif, level_spc = cif_last, spc_last

    rows = []
    for h in range(1, H + 1):
        t = last_per + h; q = t.quarter
        prev_ps = scale * level_cif * level_spc
        prev_cif, prev_spc = level_cif, level_spc

        f_cif = latest_cif.get(q, 1.0); f_spc = latest_spc.get(q, 1.0)
        level_cif *= f_cif; level_spc *= f_spc
        level_ps = scale * level_cif * level_spc

        d_cif = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan
        d_ps  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan

        rows.append({"quarter": str(t), "bank": g["bank"].iloc[0], "method": "Latest QoQ",
                     "projected_cif_bn": level_cif,
                     "projected_sales_per_cif_000": level_spc,
                     "projected_purchase_sales_bn": level_ps,
                     "delta_cif_pct": round(d_cif,2) if pd.notna(d_cif) else None,
                     "delta_sales_per_cif_pct": round(d_spc,2) if pd.notna(d_spc) else None,
                     "delta_purchase_sales_pct": round(d_ps,2) if pd.notna(d_ps) else None})
    return pd.DataFrame(rows)

# =========================
# METHOD A — Rolling seasonal QoQ (CIF & SPC → PS via identity)
# =========================
def project_rolling_qoq_identity(bank_df: pd.DataFrame) -> pd.DataFrame:
    g = bank_df.sort_values("quarter_dt").copy()
    per = g["quarter_dt"].dt.to_period("Q")
    last_dt  = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    # splice scaling
    cif_last = float(g.iloc[-1][cif_col]); spc_last = float(g.iloc[-1][spc_col])
    denom = cif_last * spc_last if (cif_last and spc_last) else None
    scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom and denom != 0 else 1.0

    hist_cif = qoq_factors_by_quarter(g[cif_col], g["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(g[spc_col], g["quarter_dt"])
    fore_cif, fore_spc = {1:[],2:[],3:[],4:[]},{1:[],2:[],3:[],4:[]}
    done_cnt = {1:0,2:0,3:0,4:0}

    level_cif, level_spc = cif_last, spc_last
    rows = []
    for h in range(1, H+1):
        t = last_per + h; q = t.quarter
        prev_ps = scale * level_cif * level_spc
        prev_cif, prev_spc = level_cif, level_spc

        K = base_window_years + done_cnt[q]
        pool_cif = hist_cif[q] + fore_cif[q]
        pool_spc = hist_spc[q] + fore_spc[q]
        f_cif = robust_mean(pool_cif[-K:] if len(pool_cif)>=K else pool_cif, median=use_median_A) if pool_cif else 1.0
        f_spc = robust_mean(pool_spc[-K:] if len(pool_spc)>=K else pool_spc, median=use_median_A) if pool_spc else 1.0
        f_cif = f_cif if (np.isfinite(f_cif) and f_cif>0) else 1.0
        f_spc = f_spc if (np.isfinite(f_spc) and f_spc>0) else 1.0

        level_cif*=f_cif; level_spc*=f_spc
        fore_cif[q].append(f_cif); fore_spc[q].append(f_spc); done_cnt[q]+=1
        level_ps = scale * level_cif * level_spc

        d_cif = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan
        d_ps  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan

        rows.append({"quarter": str(t), "bank": g["bank"].iloc[0], "method": "Rolling Seasonal (QoQ avg)",
                     "projected_cif_bn": level_cif,
                     "projected_sales_per_cif_000": level_spc,
                     "projected_purchase_sales_bn": level_ps,
                     "delta_cif_pct": round(d_cif,2) if pd.notna(d_cif) else None,
                     "delta_sales_per_cif_pct": round(d_spc,2) if pd.notna(d_spc) else None,
                     "delta_purchase_sales_pct": round(d_ps,2) if pd.notna(d_ps) else None})
    return pd.DataFrame(rows)

# =========================
# METHOD C — Baseline (PS) × Uplift (Δ%CIF, Δ%BAL)
# =========================
def fit_uplift_coefs(panel_bank: pd.DataFrame, window_years: int = 2) -> Tuple[float,float,float,pd.DataFrame]:
    """
    Estimate pooled OLS on PS growth vs Δ%CIF and Δ%BAL.
    - Build historical QoQ % growth for PS, CIF, BAL.
    - Build PS baseline growth per quarter as rolling QoQ avg of PS (using window_years, expanding per quarter).
    - Residual growth r = g_ps - g_base.
    - Pooled OLS: r ~ d_cif + d_bal (with intercept) via numpy lstsq.
    Returns (beta_cif, beta_bal, intercept, fit_df for diagnostics/charting).
    """
    g = panel_bank.sort_values(["bank","quarter_dt"]).copy()
    g["per"] = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = g["per"].apply(lambda p: p.quarter)

    # QoQ % changes
    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")[cif_col].pct_change()
    g["d_bal"] = g.groupby("bank")[bal_col].pct_change()

    # Baseline PS growth per quarter: rolling QoQ avg by quarter-of-year, expanding.
    base_g = []
    # Build pools per bank&quarter for past PS QoQ factors (not % but close; use % to be consistent)
    for b, gb in g.groupby("bank"):
        pools = {1:[],2:[],3:[],4:[]}
        done = {1:0,2:0,3:0,4:0}
        series = gb["purchase_sales_bn"].values
        per_b  = gb["per"].values
        q_b    = gb["qtr"].values

        # compute QoQ % for PS first (aligned)
        dps = gb["d_ps"].values
        base_series = []
        for i in range(len(gb)):
            q = q_b[i]
            # use pool (past dps for same q)
            pool = pools[q]
            K = base_window_years + done[q]  # expanding window for in-sample too
            if len(pool) == 0:
                g_base = np.nan  # cannot baseline first time; will drop later
            else:
                use = pool[-K:] if len(pool)>=K else pool
                g_base = float(np.median(use) if use_median_A else np.mean(use))
            base_series.append(g_base)
            # after computing baseline for this point, add actual dps[i] to pool if available
            if pd.notna(dps[i]):
                pools[q].append(dps[i]); done[q]+=1
        base_g.append(pd.Series(base_series, index=gb.index))

    g["g_base"] = pd.concat(base_g).sort_index()
    # Residual growth
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps","d_cif","d_bal"]).copy()
    if fit.empty:
        return 0.0, 0.0, 0.0, pd.DataFrame()

    # Winsorize big movements to reduce outlier leverage
    for col in ["r_ps","d_cif","d_bal"]:
        fit[col] = fit[col].clip(lower=-0.5, upper=0.5)

    # Design matrix: [1, d_cif, d_bal]
    X = np.column_stack([
        np.ones(len(fit)),
        fit["d_cif"].to_numpy(),
        fit["d_bal"].to_numpy()
    ])
    y = fit["r_ps"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, b_cif, b_bal = float(beta[0]), float(beta[1]), float(beta[2])

    return b_cif, b_bal, intercept, fit

def project_method_c(bank_df: pd.DataFrame, b_cif: float, b_bal: float, intercept: float) -> pd.DataFrame:
    """
    Forecast PS with: g_ps = g_base + (intercept + b_cif*d_cif + b_bal*d_bal), then compound.
    - g_base from PS rolling QoQ seasonal (expanding window).
    - d_cif, d_bal from CIF & Balances rolling QoQ seasonal (expanding window).
    """
    gb = bank_df.sort_values("quarter_dt").copy()
    per = gb["quarter_dt"].dt.to_period("Q")
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    # Initialize last levels
    ps_last  = float(gb.iloc[-1]["purchase_sales_bn"])
    cif_last = float(gb.iloc[-1][cif_col])
    bal_last = float(gb.iloc[-1][bal_col])

    # Build historical QoQ % pools per quarter for PS, CIF, BAL
    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])    # factors
    hist_cif = qoq_factors_by_quarter(gb[cif_col], gb["quarter_dt"])
    hist_bal = qoq_factors_by_quarter(gb[bal_col], gb["quarter_dt"])
    fore_ps, fore_cif, fore_bal = {1:[],2:[],3:[],4:[]}, {1:[],2:[],3:[],4:[]}, {1:[],2:[],3:[],4:[]}
    done_cnt = {1:0,2:0,3:0,4:0}

    rows = []
    level_ps, level_cif, level_bal = ps_last, cif_last, bal_last

    ppt_cap = uplift_cap_pct_pts / 100.0  # convert ppt to proportion for cap

    for h in range(1, H+1):
        t = last_per + h; q = t.quarter

        # Previous quarter levels for deltas
        prev_ps  = level_ps
        prev_cif = level_cif
        prev_bal = level_bal

        # --- Baseline g_base from PS rolling QoQ seasonal
        Kp = base_window_years + done_cnt[q]
        pool_ps = hist_ps[q] + fore_ps[q]
        f_ps_base = robust_mean(pool_ps[-Kp:] if len(pool_ps)>=Kp else pool_ps, median=use_median_A) if pool_ps else 1.0
        f_ps_base = f_ps_base if (np.isfinite(f_ps_base) and f_ps_base>0) else 1.0
        g_base = f_ps_base - 1.0  # baseline growth (proportion)

        # --- Driver % changes from rolling QoQ seasonal (as proportions)
        Kd = base_window_years + done_cnt[q]
        # CIF
        pool_cif = hist_cif[q] + fore_cif[q]
        f_cif = robust_mean(pool_cif[-Kd:] if len(pool_cif)>=Kd else pool_cif, median=use_median_A) if pool_cif else 1.0
        f_cif = f_cif if (np.isfinite(f_cif) and f_cif>0) else 1.0
        d_cif = f_cif - 1.0
        # BAL
        pool_bal = hist_bal[q] + fore_bal[q]
        f_bal = robust_mean(pool_bal[-Kd:] if len(pool_bal)>=Kd else pool_bal, median=use_median_A) if pool_bal else 1.0
        f_bal = f_bal if (np.isfinite(f_bal) and f_bal>0) else 1.0
        d_bal = f_bal - 1.0

        # --- Uplift (capped)
        uplift = intercept + b_cif*d_cif + b_bal*d_bal
        uplift = max(min(uplift, ppt_cap), -ppt_cap)

        # --- Total PS growth & evolve
        g_total = g_base + uplift
        level_ps  = level_ps  * (1.0 + g_total)
        level_cif = level_cif * (1.0 + d_cif)
        level_bal = level_bal * (1.0 + d_bal)

        # Append forecast factors for expanding windows
        fore_ps[q].append(1.0 + g_base)   # store baseline factor, not total
        fore_cif[q].append(1.0 + d_cif)
        fore_bal[q].append(1.0 + d_bal)
        done_cnt[q] += 1

        # Deltas (%)
        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_bal_pct = (level_bal/prev_bal - 1)*100 if prev_bal else np.nan

        rows.append({
            "quarter": str(t),
            "bank": gb["bank"].iloc[0],
            "method": "Method C (Baseline×Uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_balances_bn": level_bal,
            "delta_purchase_sales_pct": round(d_ps_pct,2) if pd.notna(d_ps_pct) else None,
            "delta_cif_pct": round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_balances_pct": round(d_bal_pct,2) if pd.notna(d_bal_pct) else None,
        })

    return pd.DataFrame(rows)

# =========================
# RUN ALL METHODS
# =========================
proj_A_frames, proj_B_frames, proj_C_frames, hist_frames = [], [], [], []
for b in banks_pick:
    gb = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col, bal_col, "purchase_sales_bn"]]
    if gb.shape[0] < 3:
        continue
    # Actuals cache for charts
    hist_frames.append(gb.assign(method="Actual"))

    # A & B for CIF/SPC identity
    pa = project_rolling_qoq_identity(gb)
    pb = project_latest_qoq_identity(gb)

    # Fit uplift coeffs on pooled (we can fit once on all banks, but for simplicity fit pooled per selection)
    # We'll concatenate selected banks to fit one pooled model.
    proj_A_frames.append(pa) if not pa.empty else None
    proj_B_frames.append(pb) if not pb.empty else None

# Fit uplift coefficients pooled on selected banks
fit_panel = panel[panel["bank"].isin(banks_pick)].copy()
b_cif, b_bal, b_int, fit_df = fit_uplift_coefs(fit_panel, window_years=base_window_years)

# Now project Method C per bank
for b in banks_pick:
    gb = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col, bal_col, "purchase_sales_bn"]]
    if gb.shape[0] < 3:
        continue
    pc = project_method_c(gb, b_cif, b_bal, b_int)
    if not pc.empty:
        proj_C_frames.append(pc)

proj_A = pd.concat(proj_A_frames, ignore_index=True) if proj_A_frames else pd.DataFrame()
proj_B = pd.concat(proj_B_frames, ignore_index=True) if proj_B_frames else pd.DataFrame()
proj_C = pd.concat(proj_C_frames, ignore_index=True) if proj_C_frames else pd.DataFrame()

# Round levels per UI (keep deltas numeric for formatting later)
for dfp in [proj_A, proj_B]:
    if not dfp.empty:
        dfp["projected_cif_bn"] = dfp["projected_cif_bn"].round(int(round_dec))
        dfp["projected_sales_per_cif_000"] = dfp["projected_sales_per_cif_000"].round(int(round_dec))
        dfp["projected_purchase_sales_bn"] = dfp["projected_purchase_sales_bn"].round(int(round_dec))
if not proj_C.empty:
    proj_C["projected_cif_bn"]       = proj_C["projected_cif_bn"].round(int(round_dec))
    proj_C["projected_balances_bn"]  = proj_C["projected_balances_bn"].round(int(round_dec))
    proj_C["projected_purchase_sales_bn"] = proj_C["projected_purchase_sales_bn"].round(int(round_dec))

# =========================
# CHARTS — A/B/C metric overlay
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

    # Projections (map by method)
    mapping = {
        "Rolling Seasonal (QoQ avg)": proj_A,
        "Latest QoQ": proj_B,
        "Method C (Baseline×Uplift)": proj_C,
    }
    dfp = mapping.get(method_name, pd.DataFrame())
    if dfp.empty: return None

    rename_map = {
        "projected_cif_bn":"cards_in_force_bn",
        "projected_sales_per_cif_000":"sales_per_cif_000",
        "projected_balances_bn":"balances_bn",
        "projected_purchase_sales_bn":"purchase_sales_bn",
    }
    use = dfp.rename(columns=rename_map)
    use["quarter_dt"] = use["quarter"].apply(lambda s: pd.Period(s, freq="Q").to_timestamp(how="end"))
    use = use[["bank","quarter_dt", metric_code, "method"]].rename(columns={metric_code:"value","method":"scenario"})
    overlays.append(use)

    overlay = pd.concat(overlays, ignore_index=True)

    # Build chart
    actual_line = (
        alt.Chart(overlay).transform_filter(alt.datum.scenario == "Actual")
          .mark_line(point=True, strokeWidth=2)
          .encode(
              x=alt.X("quarter_dt:T", title="Quarter", axis=alt.Axis(format="%Y Q%q")),
              y=alt.Y("value:Q", title=metric_label),
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"), alt.Tooltip("quarter_dt:T", format="%Y Q%q"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"), alt.Tooltip("scenario:N")]
          ).properties(height=360)
    )
    proj_line = (
        alt.Chart(overlay).transform_filter(alt.datum.scenario != "Actual")
          .mark_line(point=False, strokeWidth=2, strokeDash=[6,4])
          .encode(
              x="quarter_dt:T", y="value:Q",
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"), alt.Tooltip("quarter_dt:T", format="%Y Q%q"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"), alt.Tooltip("scenario:N")]
          ).properties(height=360)
    )
    return alt.layer(actual_line, proj_line)

metric_pick = st.selectbox(
    "Metric to chart",
    options=[FRIENDLY["purchase_sales_bn"], FRIENDLY["cards_in_force_bn"], FRIENDLY["balances_bn"], FRIENDLY["sales_per_cif_000"]],
    index=0
)
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_pick]

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Method A — Rolling Seasonal (QoQ avg)")
    chA = chart_method("Rolling Seasonal (QoQ avg)", metric_code)
    st.altair_chart(chA, use_container_width=True) if chA else st.info("No projections for Method A.")
with col2:
    st.subheader("Method B — Latest same‑quarter QoQ")
    chB = chart_method("Latest QoQ", metric_code)
    st.altair_chart(chB, use_container_width=True) if chB else st.info("No projections for Method B.")
with col3:
    st.subheader("Method C — Baseline × Driver‑Uplift")
    chC = chart_method("Method C (Baseline×Uplift)", metric_code)
    st.altair_chart(chC, use_container_width=True) if chC else st.info("No projections for Method C.")

# =========================
# DRIVER DIAGNOSTICS — Best-fit lines (Method C)
# =========================
st.subheader("Method C — Driver diagnostics (best‑fit lines)")
if fit_df.empty:
    st.info("Not enough data to estimate uplift coefficients.")
else:
    # Scatter of Δ%PS vs Δ%CIF and Δ%PS vs Δ%BAL with regression lines (simple, not residuals for viewing clarity)
    diag = fit_df.dropna(subset=["d_ps","d_cif","d_bal"]).copy()
    # Convert to percent for display
    diag["d_ps_pct"]  = diag["d_ps"]*100
    diag["d_cif_pct"] = diag["d_cif"]*100
    diag["d_bal_pct"] = diag["d_bal"]*100

    def scatter_with_line(df, x, y, x_title, y_title):
        base = alt.Chart(df).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=alt.Color("bank:N", legend=None, scale=color_scale),
            tooltip=["bank","quarter_dt", alt.Tooltip(x, format=",.2f"), alt.Tooltip(y, format=",.2f")]
        )
        # Regression line via transform_regression
        line = alt.Chart(df).transform_regression(x, y).mark_line(color="#e31a1c", strokeWidth=2)
        return (base + line).properties(height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Δ% Purchase Sales vs Δ% CIF (QoQ)")
        st.altair_chart(scatter_with_line(diag.assign(quarter_dt=diag["per"].dt.to_timestamp()),
                                          "d_cif_pct","d_ps_pct",
                                          "Δ% CIF (QoQ, %)", "Δ% Purchase Sales (QoQ, %)"), use_container_width=True)
    with c2:
        st.caption("Δ% Purchase Sales vs Δ% Balances (QoQ)")
        st.altair_chart(scatter_with_line(diag.assign(quarter_dt=diag["per"].dt.to_timestamp()),
                                          "d_bal_pct","d_ps_pct",
                                          "Δ% Balances (QoQ, %)", "Δ% Purchase Sales (QoQ, %)"), use_container_width=True)

# =========================
# TABLES — A/B/C with % delta formatting
# =========================
def tidy_sort_percent(dfp: pd.DataFrame, cols_pct: List[str]) -> pd.DataFrame:
    if dfp.empty: return dfp
    bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
    dfp["_b_sort"] = dfp["bank"].map(lambda x: bank_order_map.get(x, 999))
    dfp["_q_sort"] = dfp["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
    dfp = dfp.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])
    dfp = format_percent_cols(dfp, cols_pct)
    return dfp

st.subheader("Projections Table — Method A (Rolling Seasonal QoQ avg)")
if not proj_A.empty:
    show_A = tidy_sort_percent(
        proj_A[["quarter","bank","method","projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
                "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]],
        ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]
    )
    st.dataframe(show_A, use_container_width=True)
else:
    st.info("No projections to show for Method A.")

st.subheader("Projections Table — Method B (Latest same‑quarter QoQ)")
if not proj_B.empty:
    show_B = tidy_sort_percent(
        proj_B[["quarter","bank","method","projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
                "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]],
        ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]
    )
    st.dataframe(show_B, use_container_width=True)
else:
    st.info("No projections to show for Method B.")

st.subheader("Projections Table — Method C (Baseline × Driver‑Uplift)")
if not proj_C.empty:
    show_C = tidy_sort_percent(
        proj_C[["quarter","bank","method","projected_purchase_sales_bn","projected_cif_bn","projected_balances_bn",
                "delta_purchase_sales_pct","delta_cif_pct","delta_balances_pct"]],
        ["delta_purchase_sales_pct","delta_cif_pct","delta_balances_pct"]
    )
    st.dataframe(show_C, use_container_width=True)
else:
    st.info("No projections to show for Method C.")
