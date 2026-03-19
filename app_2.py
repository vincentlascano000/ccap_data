# app.py
# CCAP — Time-series projections where: Purchase Sales = (CIF) × (Sales/CIF)
# Rolling seasonal averages by quarter-of-year for growth factors (no charts).
# Extends per bank to 2028 Q4. Outputs a single projections table.

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ============== CONFIG ==============
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

# ============== HELPERS ==============
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

    # Fallback: try datetime
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

# ============== UI ==============
st.set_page_config(page_title="CCAP — TS: CIF × Sales/CIF (Rolling Seasonal Averages)", layout="wide")
st.title("CCAP — Projections via CIF × Sales/CIF (Rolling Seasonal Averages)")

st.sidebar.header("Projection settings")
base_window = st.sidebar.slider("Initial lookback (years) per quarter", 1, 4, 2, 1)  # Q1-2026 uses last 'base_window' Q1s
use_median  = st.sidebar.checkbox("Use median instead of mean for growth", value=False)
round_dec   = st.sidebar.selectbox("Round decimals", options=[0,1,2], index=1)

st.caption(f"Forecast end: **{str(TARGET_END)}** — per bank, Purchase Sales = CIF × Sales/CIF (with auto scale at splice).")

# ============== LOAD & NORMALIZE ==============
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

# Ensure CIF & SPC columns (with manual mapping fallback)
st.sidebar.header("Column mapping (if needed)")
def ensure_col(name_key: str, label: str) -> str:
    if name_key in df.columns:
        return name_key
    cands = candidate_cols(df, COLUMN_SYNONYMS[name_key]) or list(df.columns)
    return st.sidebar.selectbox(f"Select column for {label}", options=cands, index=0)

cif_col = ensure_col("cards_in_force_bn", FRIENDLY["cards_in_force_bn"])
spc_col = ensure_col("sales_per_cif_000", FRIENDLY["sales_per_cif_000"])

# Optional: purchase sales if present (for scaling)
has_sales = "purchase_sales_bn" in df.columns
if not has_sales:
    cands = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"])
    if cands:
        sel = st.sidebar.selectbox("Select column for Purchase Sales (optional — for better scaling)", options=["<none>"] + cands, index=0)
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

# ============== CORE LOGIC ==============
def robust_mean(x: pd.Series, median=False) -> float:
    return float(x.median()) if median else float(x.mean())

def rolling_seasonal_projection(bank_df: pd.DataFrame,
                                base_window: int,
                                target_end: pd.Period,
                                use_median: bool) -> pd.DataFrame:
    """
    Project CIF and Sales/CIF individually using rolling seasonal averages of QoQ growth factors.
    Then Purchase Sales = scale * CIF * Sales/CIF, where scale calibrates to last actual quarter.
    """
    g = bank_df.sort_values("quarter_dt").copy()
    g["qtr"] = g["quarter_dt"].dt.quarter

    # Last actual period
    last_dt  = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")

    # Horizon in quarters to TARGET_END
    H = (target_end.year - last_per.year) * 4 + (target_end.quarter - last_per.quarter)
    H = int(max(0, H))
    if H == 0:
        return pd.DataFrame(columns=["quarter","bank","projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn"])

    # Current levels (last actual)
    cif_level = float(g.iloc[-1][cif_col])
    spc_level = float(g.iloc[-1][spc_col])

    # Optional scale to match last actual purchase sales (if present)
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1]["purchase_sales_bn"]) and (cif_level is not None) and (spc_level is not None) and cif_level != 0:
        # If SPC is in '000, units may not match; learn a scalar so identity holds at splice.
        denom = cif_level * spc_level
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    # Build historical QoQ factors for each quarter-of-year for both CIF and SPC
    # Factor f_t = X_t / X_{t-1}; we collect by quarter-of-year of time t (i.e., the realized quarter).
    def quarter_factors(series: pd.Series, qtrs: pd.Series) -> Dict[int, List[float]]:
        # compute f_t per row
        f = series / series.shift(1)
        data = {}
        for k in (1,2,3,4):
            vals = f[qtrs == k].dropna().tolist()
            data[k] = vals
        return data

    hist_factors_cif = quarter_factors(g[cif_col], g["qtr"])
    hist_factors_spc = quarter_factors(g[spc_col], g["qtr"])

    # Keep forecasted factors we will generate (to implement growing rolling window)
    fore_factors_cif = {k: [] for k in (1,2,3,4)}
    fore_factors_spc = {k: [] for k in (1,2,3,4)}
    # Track how many times we have forecasted each quarter
    forecasts_done = {k: 0 for k in (1,2,3,4)}

    rows = []
    for h in range(1, H + 1):
        next_per = last_per + h
        next_q   = next_per.quarter

        # Window size grows: base_window + number of prior forecasts for this quarter
        win_k = base_window + forecasts_done[next_q]

        def next_factor(hist: Dict[int, List[float]], fore: Dict[int, List[float]], q: int, fallback: float = 1.0) -> float:
            pool = hist[q] + fore[q]
            if len(pool) == 0:
                return fallback
            # use last 'win_k' elements (if fewer available, use all)
            use = pool[-win_k:] if len(pool) >= win_k else pool
            # convert to growth then average, or directly average factors — both are similar here
            # we will average factors to stay in multiplicative space
            avg_f = robust_mean(pd.Series(use), median=use_median)
            # sanity: avoid negative or zero factors
            if not np.isfinite(avg_f) or avg_f <= 0.0:
                return 1.0
            return float(avg_f)

        # Fallback if no pool yet: use overall average factor from *all* quarters (recent ones if possible)
        def overall_fallback(series: pd.Series) -> float:
            f = (series / series.shift(1)).dropna()
            if f.empty:
                return 1.0
            return float(f.median() if use_median else f.mean())

        # CIF factor
        cif_fallback = overall_fallback(g[cif_col])
        f_cif = next_factor(hist_factors_cif, fore_factors_cif, next_q, fallback=cif_fallback)
        cif_level *= f_cif
        fore_factors_cif[next_q].append(f_cif)

        # SPC factor
        spc_fallback = overall_fallback(g[spc_col])
        f_spc = next_factor(hist_factors_spc, fore_factors_spc, next_q, fallback=spc_fallback)
        spc_level *= f_spc
        fore_factors_spc[next_q].append(f_spc)

        # Purchase Sales = scale × CIF × SPC
        ps_level = scale * cif_level * spc_level

        rows.append({
            "quarter": str(next_per),
            "projected_cif_bn": cif_level,
            "projected_sales_per_cif_000": spc_level,
            "projected_purchase_sales_bn": ps_level,
        })

        # increment "done" counter for this quarter
        forecasts_done[next_q] += 1

    out = pd.DataFrame(rows)
    out["bank"] = g["bank"].iloc[0]
    return out[["quarter","bank","projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn"]]

# ============== RUN PROJECTIONS ==============
proj_frames = []
for b in banks_pick:
    gbank = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col] + (["purchase_sales_bn"] if "purchase_sales_bn" in panel.columns else [])]
    if gbank.shape[0] < 2:
        continue
    proj_b = rolling_seasonal_projection(gbank, base_window=base_window, target_end=TARGET_END, use_median=use_median)
    if not proj_b.empty:
        proj_frames.append(proj_b)

if not proj_frames:
    st.info("Not enough history or already at/after 2028 Q4 for selected banks — no projections to show.")
    st.stop()

projections = pd.concat(proj_frames, ignore_index=True)

# Round
projections["projected_cif_bn"] = projections["projected_cif_bn"].round(int(round_dec))
projections["projected_sales_per_cif_000"] = projections["projected_sales_per_cif_000"].round(int(round_dec))
projections["projected_purchase_sales_bn"] = projections["projected_purchase_sales_bn"].round(int(round_dec))

# Order by quarter then bank
bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
projections["_b_sort"] = projections["bank"].map(lambda x: bank_order_map.get(x, 999))
# Parse quarter like "2027Q3" into sortable key
projections["_q_sort"] = projections["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
projections = projections.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])

st.subheader("Projected Numbers — CIF × Sales/CIF → Purchase Sales (to 2028 Q4)")
st.dataframe(projections, use_container_width=True)
