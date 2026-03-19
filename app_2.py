# app.py
# CCAP — CIF × Sales/CIF → Purchase Sales, with TWO QoQ seasonal projection methods
# • Method A: Rolling seasonal QoQ average (expanding window by quarter-of-year)
# • Method B: Latest same-quarter QoQ carry-forward
# • Purchase Sales = scale × CIF × Sales/CIF  (scale fitted at splice to reconcile units)
# • Two charts (one per method), and tables with QoQ % deltas (two decimals as percent strings)
# • Forecast horizon: to 2028 Q4 (inclusive), per bank

import re
from typing import Dict, List

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
    "sales_per_cif_000": [
        "sales_per_cif_000","sales_per_cif","sales per cif (000)","sales/cif (000)","sales_cif"
    ],
}

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
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
    """Parse '1Q23', 'Q1 2023', '2023 Q1', or a date → (label, quarter-end Timestamp)."""
    if pd.isna(value):
        return None, pd.NaT
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

    # Fallback: datetime
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

@st.cache_data(ttl=600, show_spinner=False)
def load_raw_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, engine="python")
    except Exception:
        return pd.read_csv(url, engine="python", encoding="utf-8-sig")

def candidate_cols(df: pd.DataFrame, names_or_keywords: List[str]) -> List[str]:
    want = {_canon(x) for x in names_or_keywords}
    out = []
    for col in df.columns:
        cc = _canon(col)
        if cc in want or any(k in cc for k in want):
            out.append(col)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def format_percent_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Render numeric percent columns as strings with '%' and 2 decimals."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: (f"{x:.2f}%" if pd.notna(x) else None))
    return out

# =========================
# UI
# =========================
st.set_page_config(page_title="CCAP — Two QoQ TS Methods + Charts", layout="wide")
st.title("CCAP — CIF × Sales/CIF → Purchase Sales: Rolling QoQ vs Latest QoQ")
st.caption(f"Forecast end: **{str(TARGET_END)}**")

st.sidebar.header("Data & Output")
round_dec = st.sidebar.selectbox("Round decimals (levels)", options=[0,1,2], index=1)

st.sidebar.header("Method A — Rolling QoQ seasonal average")
base_window_years = st.sidebar.slider("Initial lookback (years) per quarter", 1, 4, 2, 1)
use_median_A = st.sidebar.checkbox("Use median (instead of mean) for A", value=False)

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

# Column mapping for CIF & Sales/CIF
st.sidebar.header("Column mapping (if needed)")
def ensure_col(name_key: str, label: str) -> str:
    if name_key in df.columns:
        return name_key
    cands = candidate_cols(df, COLUMN_SYNONYMS[name_key]) or list(df.columns)
    return st.sidebar.selectbox(f"Select column for {label}", options=cands, index=0)

cif_col = ensure_col("cards_in_force_bn", FRIENDLY["cards_in_force_bn"])
spc_col = ensure_col("sales_per_cif_000", FRIENDLY["sales_per_cif_000"])

# Optional: purchase sales (for scale)
if "purchase_sales_bn" not in df.columns:
    cands = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"])
    if cands:
        sel = st.sidebar.selectbox("Select column for Purchase Sales (optional — splice scaling)", options=["<none>"] + cands, index=0)
        if sel != "<none>":
            df["purchase_sales_bn"] = df[sel]

# Coerce numeric
df[cif_col] = to_numeric(df[cif_col])
df[spc_col] = to_numeric(df[spc_col])
if "purchase_sales_bn" in df.columns:
    df["purchase_sales_bn"] = to_numeric(df["purchase_sales_bn"])

panel = (df[["bank","quarter_dt", cif_col, spc_col] + (["purchase_sales_bn"] if "purchase_sales_bn" in df.columns else [])]
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
# QoQ FACTORS (by quarter-of-year)
# =========================
def historical_qoq_factors_by_quarter(series: pd.Series, periods: pd.Series) -> Dict[int, List[float]]:
    """
    Compute QoQ factor f_t = X_t / X_{t-1}, then group by quarter-of-year of t (1..4).
    Returns dict: {1: [f for Q1s], 2: [...], 3: [...], 4: [...]}
    """
    s_per = periods.dt.to_period("Q")
    s = pd.Series(series.values, index=s_per).sort_index()
    f = (s / s.shift(1)).dropna()
    out = {1: [], 2: [], 3: [], 4: []}
    for p, val in f.items():
        if np.isfinite(val) and val > 0:
            out[p.quarter].append(float(val))
    return out

def latest_same_quarter_qoq_factor(series: pd.Series, periods: pd.Series) -> Dict[int, float]:
    """
    For each q in {1..4}, return the latest observed QoQ factor for that quarter-of-year:
        f_q = X_{t}/X_{t-1} where quarter(t)=q, from the most recent such t with valid data.
    Fallback to the latest overall QoQ factor if none for that q.
    """
    s_per = periods.dt.to_period("Q")
    s = pd.Series(series.values, index=s_per).sort_index()
    f = (s / s.shift(1)).dropna()
    latest_overall = float(f.iloc[-1]) if not f.empty and np.isfinite(f.iloc[-1]) and f.iloc[-1] > 0 else 1.0

    latest = {}
    for q in (1,2,3,4):
        # pick last factor whose period has quarter==q
        fq = [float(v) for p, v in f.items() if p.quarter == q and np.isfinite(v) and v > 0]
        latest[q] = (fq[-1] if fq else latest_overall)
        if not np.isfinite(latest[q]) or latest[q] <= 0:
            latest[q] = 1.0
    return latest

# =========================
# METHOD B — Latest same-quarter QoQ carry-forward
# =========================
def project_bank_latest_same_qtr_qoq(bank_df: pd.DataFrame, target_end: pd.Period) -> pd.DataFrame:
    """
    For each future quarter t (in order):
      CIF_t = CIF_{t-1} * f_qoq_cif[q(t)], where f_qoq_cif[q] is the latest observed QoQ factor for that quarter-of-year.
      SPC_t = SPC_{t-1} * f_qoq_spc[q(t)],  same idea.
      PS_t  = scale * CIF_t * SPC_t
    """
    g = bank_df.sort_values("quarter_dt").copy()
    per = g["quarter_dt"].dt.to_period("Q")
    last_dt  = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (target_end.year - last_per.year) * 4 + (target_end.quarter - last_per.quarter)
    H = int(max(0, H))
    if H == 0:
        return pd.DataFrame(columns=[
            "quarter","bank","method",
            "projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn",
            "delta_cif_pct","delta_sales_per_cif_pct","delta_purchase_sales_pct"
        ])

    # Splice scaling
    cif_last = float(g.iloc[-1][cif_col]); spc_last = float(g.iloc[-1][spc_col])
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1].get("purchase_sales_bn", np.nan)) and cif_last and spc_last:
        denom = cif_last * spc_last
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    # Latest same-quarter QoQ factors (constants for future quarters)
    latest_cif = latest_same_quarter_qoq_factor(g[cif_col], g["quarter_dt"])
    latest_spc = latest_same_quarter_qoq_factor(g[spc_col], g["quarter_dt"])

    # Start from last actual levels; iterate sequentially by quarter
    level_cif = cif_last; level_spc = spc_last
    rows = []
    for h in range(1, H + 1):
        t = last_per + h
        q = t.quarter

        prev_cif = level_cif
        prev_spc = level_spc
        prev_ps  = scale * prev_cif * prev_spc

        f_cif = latest_cif.get(q, 1.0)
        f_spc = latest_spc.get(q, 1.0)

        # evolve by QoQ
        level_cif *= f_cif
        level_spc *= f_spc
        level_ps   = scale * level_cif * level_spc

        # deltas (%)
        d_cif = ((level_cif / prev_cif) - 1.0) * 100 if prev_cif not in (0, np.nan) else np.nan
        d_spc = ((level_spc / prev_spc) - 1.0) * 100 if prev_spc not in (0, np.nan) else np.nan
        d_ps  = ((level_ps  / prev_ps ) - 1.0) * 100 if prev_ps  not in (0, np.nan) else np.nan

        rows.append({
            "quarter": str(t),
            "bank": g["bank"].iloc[0],
            "method": "Latest QoQ",
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "projected_purchase_sales_bn": level_ps,
            "delta_cif_pct": None if pd.isna(d_cif) else round(float(d_cif), 2),
            "delta_sales_per_cif_pct": None if pd.isna(d_spc) else round(float(d_spc), 2),
            "delta_purchase_sales_pct": None if pd.isna(d_ps)  else round(float(d_ps),  2),
        })
    return pd.DataFrame(rows)

# =========================
# METHOD A — Rolling seasonal QoQ average (expanding window)
# =========================
def robust_mean(vals: List[float], median=False) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 1.0
    return float(np.median(arr) if median else np.mean(arr))

def rolling_seasonal_projection_qoq(bank_df: pd.DataFrame,
                                    base_window: int,
                                    use_median: bool,
                                    target_end: pd.Period) -> pd.DataFrame:
    """
    For each future quarter t (in order), with q = quarter(t):
      1) Build pools of historical QoQ factors for q (X_t/X_{t-1} where quarter(t)==q) for CIF and SPC.
      2) The working pool grows by appending each forecast factor as we go.
      3) Let K = base_window + (# times q has been forecast so far).
      4) Choose factor_q = average of the last K factors from the (historical + forecasted) pool for quarter q.
      5) Evolve sequentially: X_t = X_{t-1} * factor_q.
      6) PurchaseSales_t = scale * CIF_t * SPC_t
    """
    g = bank_df.sort_values("quarter_dt").copy()
    last_dt  = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (target_end.year - last_per.year) * 4 + (target_end.quarter - last_per.quarter)
    H = int(max(0, H))
    if H == 0:
        return pd.DataFrame(columns=[
            "quarter","bank","method",
            "projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn",
            "delta_cif_pct","delta_sales_per_cif_pct","delta_purchase_sales_pct"
        ])

    # Splice scaling
    cif_last = float(g.iloc[-1][cif_col]); spc_last = float(g.iloc[-1][spc_col])
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1].get("purchase_sales_bn", np.nan)) and cif_last and spc_last:
        denom = cif_last * spc_last
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    # Historical QoQ factor pools by quarter
    hist_cif = historical_qoq_factors_by_quarter(g[cif_col], g["quarter_dt"])
    hist_spc = historical_qoq_factors_by_quarter(g[spc_col], g["quarter_dt"])

    # Growing forecast factor pools and counters
    fore_cif = {1: [], 2: [], 3: [], 4: []}
    fore_spc = {1: [], 2: [], 3: [], 4: []}
    done_cnt = {1: 0, 2: 0, 3: 0, 4: 0}

    level_cif = cif_last
    level_spc = spc_last

    rows = []
    for h in range(1, H + 1):
        t = last_per + h
        q = t.quarter

        prev_cif = level_cif
        prev_spc = level_spc
        prev_ps  = scale * prev_cif * prev_spc

        # window K for this quarter-of-year
        K = base_window + done_cnt[q]

        # CIF factor: use last K of (hist + fore) pool
        pool_cif = hist_cif[q] + fore_cif[q]
        if len(pool_cif) == 0:
            f_cif = 1.0
        else:
            use = pool_cif[-K:] if len(pool_cif) >= K else pool_cif
            f_cif = robust_mean(use, median=use_median)
            if not np.isfinite(f_cif) or f_cif <= 0:
                f_cif = 1.0

        # SPC factor similarly
        pool_spc = hist_spc[q] + fore_spc[q]
        if len(pool_spc) == 0:
            f_spc = 1.0
        else:
            use = pool_spc[-K:] if len(pool_spc) >= K else pool_spc
            f_spc = robust_mean(use, median=use_median)
            if not np.isfinite(f_spc) or f_spc <= 0:
                f_spc = 1.0

        # evolve levels by QoQ factors
        level_cif *= f_cif
        level_spc *= f_spc
        level_ps   = scale * level_cif * level_spc

        # append forecast factors into pools (so windows expand year after year)
        fore_cif[q].append(f_cif)
        fore_spc[q].append(f_spc)
        done_cnt[q] += 1

        # deltas (%)
        d_cif = ((level_cif / prev_cif) - 1.0) * 100 if prev_cif not in (0, np.nan) else np.nan
        d_spc = ((level_spc / prev_spc) - 1.0) * 100 if prev_spc not in (0, np.nan) else np.nan
        d_ps  = ((level_ps  / prev_ps ) - 1.0) * 100 if prev_ps  not in (0, np.nan) else np.nan

        rows.append({
            "quarter": str(t),
            "bank": g["bank"].iloc[0],
            "method": "Rolling Seasonal (QoQ avg)",
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "projected_purchase_sales_bn": level_ps,
            "delta_cif_pct": None if pd.isna(d_cif) else round(float(d_cif), 2),
            "delta_sales_per_cif_pct": None if pd.isna(d_spc) else round(float(d_spc), 2),
            "delta_purchase_sales_pct": None if pd.isna(d_ps)  else round(float(d_ps),  2),
        })

    return pd.DataFrame(rows)

# =========================
# RUN BOTH METHODS
# =========================
proj_A_frames, proj_B_frames, hist_frames = [], [], []
for b in banks_pick:
    gbank = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col] + (["purchase_sales_bn"] if "purchase_sales_bn" in panel.columns else [])]
    if gbank.shape[0] < 2:
        continue
    # Actuals
    hist = gbank.rename(columns={cif_col:"cif", spc_col:"spc"})
    hist_frames.append(hist.assign(method="Actual"))

    # Projections
    proj_A = rolling_seasonal_projection_qoq(gbank, base_window=base_window_years, use_median=use_median_A, target_end=TARGET_END)
    proj_B = project_bank_latest_same_qtr_qoq(gbank, TARGET_END)
    if not proj_A.empty: proj_A_frames.append(proj_A)
    if not proj_B.empty: proj_B_frames.append(proj_B)

if not (proj_A_frames or proj_B_frames):
    st.info("No projections available (check mapping or banks).")
    st.stop()

proj_A = pd.concat(proj_A_frames, ignore_index=True) if proj_A_frames else pd.DataFrame()
proj_B = pd.concat(proj_B_frames, ignore_index=True) if proj_B_frames else pd.DataFrame()

# Round level columns per UI
for dfp in [proj_A, proj_B]:
    if not dfp.empty:
        dfp["projected_cif_bn"] = dfp["projected_cif_bn"].round(int(round_dec))
        dfp["projected_sales_per_cif_000"] = dfp["projected_sales_per_cif_000"].round(int(round_dec))
        dfp["projected_purchase_sales_bn"] = dfp["projected_purchase_sales_bn"].round(int(round_dec))

# =========================
# CHARTS (two separate charts: one per method)
# =========================
color_scale = alt.Scale(scheme='tableau10')

def chart_method(method_name: str, metric_code: str):
    metric_label = FRIENDLY.get(metric_code, metric_code)
    overlays = []

    # Actual
    hist_all = pd.concat(hist_frames, ignore_index=True)
    actual = hist_all[["bank","quarter_dt",metric_code]].dropna().copy()
    actual = actual.rename(columns={metric_code:"value"})
    actual["scenario"] = "Actual"
    overlays.append(actual[["bank","quarter_dt","value","scenario"]])

    # Projections
    proj = proj_A if method_name.startswith("Rolling") else proj_B
    if proj.empty:
        return None
    proj_use = proj.rename(columns={
        "projected_cif_bn":"cards_in_force_bn",
        "projected_sales_per_cif_000":"sales_per_cif_000",
        "projected_purchase_sales_bn":"purchase_sales_bn"
    })
    proj_use["quarter_dt"] = proj_use["quarter"].apply(lambda s: pd.Period(s, freq="Q").to_timestamp(how="end"))
    proj_use = proj_use[["bank","quarter_dt", metric_code, "method"]].rename(columns={metric_code:"value","method":"scenario"})
    overlays.append(proj_use)

    overlay = pd.concat(overlays, ignore_index=True)

    # Build chart
    actual_line = (
        alt.Chart(overlay)
          .transform_filter(alt.datum.scenario == "Actual")
          .mark_line(point=True, strokeWidth=2)
          .encode(
              x=alt.X("quarter_dt:T", title="Quarter", axis=alt.Axis(format="%Y Q%q")),
              y=alt.Y("value:Q", title=metric_label),
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"), alt.Tooltip("quarter_dt:T", format="%Y Q%q"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"), alt.Tooltip("scenario:N")]
          )
          .properties(height=360)
    )

    proj_line = (
        alt.Chart(overlay)
          .transform_filter(alt.datum.scenario != "Actual")
          .mark_line(point=False, strokeWidth=2, strokeDash=[6,4])
          .encode(
              x="quarter_dt:T",
              y="value:Q",
              color=alt.Color("bank:N", title="Bank", sort=banks_pick, scale=color_scale),
              tooltip=[alt.Tooltip("bank:N"), alt.Tooltip("quarter_dt:T", format="%Y Q%q"),
                       alt.Tooltip("value:Q", title=metric_label, format=",.2f"), alt.Tooltip("scenario:N")]
          )
          .properties(height=360)
    )
    return alt.layer(actual_line, proj_line)

# Metric choice for charts
metric_pick = st.selectbox(
    "Metric to chart",
    options=[FRIENDLY["purchase_sales_bn"], FRIENDLY["cards_in_force_bn"], FRIENDLY["sales_per_cif_000"]],
    index=0
)
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_pick]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Method A — Rolling Seasonal (QoQ avg)")
    chA = chart_method("Rolling Seasonal (QoQ avg)", metric_code)
    if chA is None:
        st.info("No projections for Method A.")
    else:
        st.altair_chart(chA, use_container_width=True)

with col2:
    st.subheader("Method B — Latest same‑quarter QoQ")
    chB = chart_method("Latest QoQ", metric_code)
    if chB is None:
        st.info("No projections for Method B.")
    else:
        st.altair_chart(chB, use_container_width=True)

# =========================
# TABLES (with % delta columns rendered as percent strings)
# =========================
def tidy_sort(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty: return dfp
    bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
    dfp["_b_sort"] = dfp["bank"].map(lambda x: bank_order_map.get(x, 999))
    dfp["_q_sort"] = dfp["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
    dfp = dfp.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])
    # Preferred column order
    cols = ["quarter","bank","method",
            "projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
            "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]
    cols = [c for c in cols if c in dfp.columns]
    return dfp[cols]

st.subheader("Projections Table — Method A (Rolling Seasonal QoQ avg)")
if not proj_A.empty:
    show_A = tidy_sort(proj_A)
    show_A = format_percent_cols(show_A, ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"])
    st.dataframe(show_A, use_container_width=True)
else:
    st.info("No projections to show for Method A.")

st.subheader("Projections Table — Method B (Latest same‑quarter QoQ)")
if not proj_B.empty:
    show_B = tidy_sort(proj_B)
    show_B = format_percent_cols(show_B, ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"])
    st.dataframe(show_B, use_container_width=True)
else:
    st.info("No projections to show for Method B.")
