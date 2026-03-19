# app.py
# CCAP — Simple Time Series Baseline (QoQ + Seasonality), tables only (no graphs)
# • Forecasts Purchase Sales (Bn) per bank
# • Trend = average QoQ % change over a window you choose
# • Seasonality = average Q1..Q4 deviation from trend (+ optional Q4 holiday lift)
# • Projections extend to 2028 Q4 (inclusive), per bank
# • Output: numerical projections table only

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# 0) SOURCE & LABELS
# =========================================
RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"

FRIENDLY = {"purchase_sales_bn": "Purchase Sales (Bn)"}
BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "quarter": ["quarter", "qtr", "period", "quarter_str", "q", "date"],
    "bank":    ["bank", "issuer", "bank_name", "issuer_name", "issuer bank"],
    "purchase_sales_bn": [
        "purchase_sales_bn","purchase_sales","sales",
        "purchase volume (bn)","purchase sales (in bn)","purchase_sales_bil","purchase_sales_billion"
    ],
}

# Forecast target (inclusive)
TARGET_END = pd.Period("2028Q4", freq="Q")

# =========================================
# 1) HELPERS
# =========================================
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[()\[\]%]", "", s)
    s = re.sub(r"[\\/]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort auto-map common header variants to canonical names."""
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
    """Parse '1Q23', 'Q1 2023', '2023 Q1', or a date → quarter-end Timestamp."""
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

    # Fallback: parse date
    try:
        dt = pd.to_datetime(s, errors="raise")
        per = pd.Period(dt, freq="Q")
        return str(per), per.to_timestamp(how="end")
    except Exception:
        return None, pd.NaT

def drop_duplicate_names_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    out, seen, keep = df.copy(), set(), []
    for c in out.columns:
        if c not in seen:
            keep.append(c); seen.add(c)
    return out.loc[:, keep]

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

# =========================================
# 2) APP UI
# =========================================
st.set_page_config(page_title="CCAP — TS Baseline (QoQ + Seasonality) — Tables", layout="wide")
st.title("CCAP — Time Series Projections (Trend + Q4 Seasonality) — Tables Only")

st.sidebar.header("Forecast settings")
window_q   = st.sidebar.slider("Window for QoQ stats (quarters)", 4, 16, 8, 1)
season_wt  = st.sidebar.slider("Seasonality strength (0 = off, 1 = full)", 0.0, 1.0, 1.0, 0.1)
holiday_q4 = st.sidebar.slider("Q4 holiday lift (additional % points)", 0.0, 0.10, 0.02, 0.01)
use_median = st.sidebar.checkbox("Use median (instead of mean) for QoQ & seasonal", value=False)
round_dec  = st.sidebar.selectbox("Round projections to (decimals)", options=[0,1,2], index=1)

st.caption(f"Forecast end: **{str(TARGET_END)}**")

# =========================================
# 3) LOAD & NORMALIZE
# =========================================
try:
    raw = load_raw_csv(RAW_URL)
except Exception as e:
    st.error(f"Failed to read CSV from GitHub: {e}")
    st.stop()

df = map_columns(drop_duplicate_names_keep_first(raw))

# Quarter parsing
if "quarter_dt" not in df.columns:
    qsrc = "quarter" if "quarter" in df.columns else None
    if qsrc is None:
        st.error("Missing quarter column. Please ensure your file has 'quarter' or 'quarter_dt'.")
        st.write("Detected columns:", list(df.columns))
        st.stop()
    df["quarter_dt"] = df[qsrc].apply(parse_quarter_token).apply(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)

# Ensure bank
if "bank" not in df.columns:
    st.error("Missing 'bank' column after mapping. Please ensure a bank/issuer column is present.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Purchase Sales mapping (if needed)
if "purchase_sales_bn" not in df.columns:
    st.sidebar.header("Column mapping")
    sugg = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"]) or list(df.columns)
    sel_sales = st.sidebar.selectbox("Select Purchase Sales column", options=sugg, index=0)
    df["purchase_sales_bn"] = to_numeric(df[sel_sales])
else:
    df["purchase_sales_bn"] = to_numeric(df["purchase_sales_bn"])

panel = (df[["bank","quarter_dt","purchase_sales_bn"]]
         .dropna(subset=["bank","quarter_dt"])
         .sort_values(["bank","quarter_dt"])
         .reset_index(drop=True))

# Bank selection
banks_all = sorted(panel["bank"].dropna().unique().tolist(),
                   key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
banks_pick = st.multiselect("Banks to include", options=banks_all, default=banks_all)

if not banks_pick:
    st.info("Select at least one bank to proceed.")
    st.stop()

# =========================================
# 4) CORE TIME SERIES LOGIC (QoQ + Seasonality → TARGET_END)
# =========================================
def robust_stat(x: pd.Series, median=False):
    return float(x.median()) if median else float(x.mean())

def seasonal_forecast_to_target(gbank: pd.DataFrame,
                                window_q: int,
                                season_strength: float,
                                q4_extra: float,
                                use_median: bool,
                                target_end: pd.Period) -> pd.DataFrame:
    """
    Baseline per bank to a fixed target quarter (e.g., 2028Q4):
      1) QoQ % change r_t = (X_t/X_{t-1}) - 1
      2) Trend g = mean(r_t) over last window_q quarters
      3) Seasonal deviation s_k for k in {Q1..Q4}:
            s_k = mean(r_t | quarter==k) - g   (over same window)
         Shrink: s_k <- season_strength * s_k
         Add holiday bump to Q4: s_4 <- s_4 + q4_extra
      4) Forecast H quarters where:
            H = max(0, (Y_target - Y_last)*4 + (Q_target - Q_last))
         For h = 1..H:
            growth = g + s_{quarter(next)}
            X_{t+1} = X_t * (1 + growth)
    """
    gb = gbank.sort_values("quarter_dt").copy()
    if gb.empty or gb["quarter_dt"].isna().all():
        return pd.DataFrame(columns=["quarter","bank","projected_purchase_sales_bn"])

    gb["qtr"] = gb["quarter_dt"].dt.quarter
    gb["pct"] = gb["purchase_sales_bn"].pct_change()

    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")

    # Normalize target_end
    if not isinstance(target_end, pd.Period) or target_end.freqstr is None or not target_end.freqstr.startswith("Q"):
        target_end = pd.Period(str(target_end), freq="Q")

    # Quarter distance H
    H = (target_end.year - last_per.year) * 4 + (target_end.quarter - last_per.quarter)
    H = int(max(0, H))
    if H == 0:
        return pd.DataFrame(columns=["quarter","bank","projected_purchase_sales_bn"])

    # Window subset
    winmask = gb["quarter_dt"] > (last_dt - pd.offsets.QuarterEnd(window_q))
    gwin = gb.loc[winmask].copy()

    # Trend g
    g_series = gwin["pct"].dropna()
    g = robust_stat(g_series, median=use_median) if g_series.size > 0 else 0.0

    # Seasonality s_k
    season_map = {}
    for k in (1, 2, 3, 4):
        sk = gwin.loc[gwin["qtr"] == k, "pct"].dropna()
        s_k = (robust_stat(sk, median=use_median) - g) if sk.size >= 1 else 0.0
        season_map[k] = season_strength * s_k
    # Add Q4 holiday lift
    season_map[4] = season_map.get(4, 0.0) + q4_extra

    # Forecast loop
    level = float(gb.iloc[-1]["purchase_sales_bn"])
    rows  = []
    for h in range(1, H + 1):
        next_per = last_per + h
        next_q   = next_per.quarter
        growth   = g + season_map.get(next_q, 0.0)
        growth   = max(growth, -0.9)   # safety clamp
        level    = level * (1.0 + growth)
        rows.append({"quarter": str(next_per), "projected_purchase_sales_bn": level})

    out = pd.DataFrame(rows)
    out["bank"] = gb["bank"].iloc[0]
    return out[["quarter","bank","projected_purchase_sales_bn"]]

# =========================================
# 5) COMPUTE PROJECTIONS (TABLES ONLY)
# =========================================
proj_frames = []
for b in banks_pick:
    gbank = panel[panel["bank"] == b]
    if gbank.shape[0] < 2:
        continue
    proj_b = seasonal_forecast_to_target(
        gbank=gbank,
        window_q=window_q,
        season_strength=season_wt,
        q4_extra=holiday_q4,
        use_median=use_median,
        target_end=TARGET_END,
    )
    if not proj_b.empty:
        proj_frames.append(proj_b)

if not proj_frames:
    st.info("Not enough history or already at/after 2028 Q4 for selected banks — no projections to show.")
    st.stop()

projections = pd.concat(proj_frames, ignore_index=True)
projections["projected_purchase_sales_bn"] = projections["projected_purchase_sales_bn"].round(int(round_dec))

# Order by quarter then bank (keep preferred bank order)
# Create a sort key for banks
bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
projections["_b_sort"] = projections["bank"].map(lambda x: bank_order_map.get(x, 999))
projections["_q_sort"] = projections["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))  # year*4 + Q

projections = projections.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])

st.subheader("Projected Purchase Sales (Bn) — Baseline (trend + seasonality) to 2028 Q4")
st.dataframe(projections, use_container_width=True)
