# app.py
# CCAP — Two projection methods + two charts + tables (to 2028 Q4)
# Method A: Rolling seasonal averages (YoY, expanding by quarter-of-year)
# Method B: Latest same-quarter YoY carry-forward
# Purchase Sales = scale × CIF × Sales/CIF (scale fitted at splice)
# NEW: Adds QoQ % change columns (2 decimals) for Purchase Sales, CIF, and Sales/CIF in projection tables.

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

# =========================
# UI
# =========================
st.set_page_config(page_title="CCAP — Two TS Methods + Charts", layout="wide")
st.title("CCAP — CIF × Sales/CIF → Purchase Sales: Rolling Seasonal vs Latest YoY")

st.caption(f"Forecast end: **{str(TARGET_END)}**")

st.sidebar.header("Data & Output")
round_dec = st.sidebar.selectbox("Round decimals", options=[0,1,2], index=1)

st.sidebar.header("Rolling seasonal (Method A)")
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
# METHOD B — Latest same-quarter YoY carry-forward
# =========================
def latest_same_quarter_yoy_factor(series: pd.Series, periods: pd.Series) -> dict[int, float]:
    """Return f_q for q=1..4 where f_q = latest available YoY factor X_{Y,q}/X_{Y-1,q}; fallback to latest QoQ."""
    per = periods.dt.to_period("Q")
    dfv = pd.DataFrame({"per": per, "val": series.astype(float)})
    dfv = dfv.dropna(subset=["per", "val"]).drop_duplicates(subset=["per"]).set_index("per").sort_index()

    yoy = {}
    for q in (1, 2, 3, 4):
        qidx = [p for p in dfv.index if p.quarter == q]
        qidx = sorted(qidx)
        f_q = None
        for p in reversed(qidx):
            prev = p - 4
            if prev in dfv.index:
                v = dfv.loc[p, "val"]
                v_prev = dfv.loc[prev, "val"]
                if pd.notna(v) and pd.notna(v_prev) and v_prev not in (0, np.nan):
                    f_q = float(v / v_prev)
                    break
        yoy[q] = f_q

    qo_q = (dfv["val"] / dfv["val"].shift(1)).dropna()
    qo_q_last = float(qo_q.iloc[-1]) if not qo_q.empty and np.isfinite(qo_q.iloc[-1]) and qo_q.iloc[-1] > 0 else 1.0

    for q in (1, 2, 3, 4):
        if yoy[q] is None or not np.isfinite(yoy[q]) or yoy[q] <= 0:
            yoy[q] = qo_q_last
    return yoy

def project_bank_latest_same_qtr_yoy(bank_df: pd.DataFrame,
                                     target_end: pd.Period) -> pd.DataFrame:
    g = bank_df.sort_values("quarter_dt").copy()
    g["qtr"] = g["quarter_dt"].dt.quarter
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

    # Splice levels
    cif_last = float(g.iloc[-1][cif_col])
    spc_last = float(g.iloc[-1][spc_col])
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1].get("purchase_sales_bn", np.nan)) and cif_last and spc_last:
        denom = cif_last * spc_last
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    yoy_cif = latest_same_quarter_yoy_factor(g[cif_col], g["quarter_dt"])
    yoy_spc = latest_same_quarter_yoy_factor(g[spc_col], g["quarter_dt"])

    qo_q_cif = (g[cif_col] / g[cif_col].shift(1)).dropna()
    qo_q_spc = (g[spc_col] / g[spc_col].shift(1)).dropna()
    f_qoq_cif = float(qo_q_cif.iloc[-1]) if not qo_q_cif.empty and qo_q_cif.iloc[-1] > 0 else 1.0
    f_qoq_spc = float(qo_q_spc.iloc[-1]) if not qo_q_spc.empty and qo_q_spc.iloc[-1] > 0 else 1.0

    per_idx = g["quarter_dt"].dt.to_period("Q")
    levels_cif = {p: float(v) for p, v in zip(per_idx, g[cif_col]) if pd.notna(v)}
    levels_spc = {p: float(v) for p, v in zip(per_idx, g[spc_col]) if pd.notna(v)}

    rows = []
    for h in range(1, H + 1):
        t = last_per + h
        q = t.quarter

        # Previous quarter levels for delta (%)
        prev_per = t - 1
        prev_cif = levels_cif.get(prev_per, cif_last)
        prev_spc = levels_spc.get(prev_per, spc_last)
        prev_ps  = scale * prev_cif * prev_spc

        # CIF
        if (t - 4) in levels_cif:
            cif_t = levels_cif[t - 4] * yoy_cif[q]
        else:
            cif_t = prev_cif * f_qoq_cif
        levels_cif[t] = cif_t

        # SPC
        if (t - 4) in levels_spc:
            spc_t = levels_spc[t - 4] * yoy_spc[q]
        else:
            spc_t = prev_spc * f_qoq_spc
        levels_spc[t] = spc_t

        # Purchase Sales
        ps_t = scale * cif_t * spc_t

        # --- NEW: deltas (QoQ %) rounded to 2 decimals
        d_cif = ((cif_t / prev_cif) - 1.0) * 100 if prev_cif not in (0, np.nan) else np.nan
        d_spc = ((spc_t / prev_spc) - 1.0) * 100 if prev_spc not in (0, np.nan) else np.nan
        d_ps  = ((ps_t  / prev_ps)  - 1.0) * 100 if prev_ps  not in (0, np.nan) else np.nan

        rows.append({
            "quarter": str(t),
            "bank": g["bank"].iloc[0],
            "method": "Latest YoY",
            "projected_cif_bn": cif_t,
            "projected_sales_per_cif_000": spc_t,
            "projected_purchase_sales_bn": ps_t,
            "delta_cif_pct": None if pd.isna(d_cif) else round(float(d_cif), 2),
            "delta_sales_per_cif_pct": None if pd.isna(d_spc) else round(float(d_spc), 2),
            "delta_purchase_sales_pct": None if pd.isna(d_ps)  else round(float(d_ps),  2),
        })
    return pd.DataFrame(rows)

# =========================
# METHOD A — Rolling seasonal averages (YoY, expanding window)
# =========================
def robust_mean(x: List[float], median=False) -> float:
    arr = np.array(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return float(np.median(arr) if median else np.mean(arr))

def rolling_seasonal_projection_yoy(bank_df: pd.DataFrame,
                                    base_window: int,
                                    use_median: bool,
                                    target_end: pd.Period) -> pd.DataFrame:
    """
    For each quarter-of-year q:
      1) Gather historical YoY factors: f_{Y,q} = X_{Y,q} / X_{Y-1,q}
      2) For forecast year Y*, use average of last K YoY factors for that quarter,
         where K starts at base_window and increases by 1 each time that quarter recurs in forecasts
      3) X_{Y*,q} = X_{Y-1,q} * avg_factor_q
    Applies separately to CIF and Sales/CIF; PS = scale × CIF × SPC.
    """
    g = bank_df.sort_values("quarter_dt").copy()
    per = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = per.apply(lambda p: p.quarter)

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
    cif_last = float(g.iloc[-1][cif_col])
    spc_last = float(g.iloc[-1][spc_col])
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1].get("purchase_sales_bn", np.nan)) and cif_last and spc_last:
        denom = cif_last * spc_last
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    # Build historical YoY factors per quarter
    def yoy_factors(series: pd.Series) -> Dict[int, List[float]]:
        s = pd.Series(series.values, index=per)
        s = s[s.notna()]
        f_by_q = {1: [], 2: [], 3: [], 4: []}
        for p in s.index:
            prev = p - 4
            if prev in s.index and s[prev] not in (0, np.nan):
                f = float(s[p] / s[prev])
                if np.isfinite(f) and f > 0:
                    f_by_q[p.quarter].append(f)
        return f_by_q

    hist_yoy_cif = yoy_factors(g[cif_col])
    hist_yoy_spc = yoy_factors(g[spc_col])

    # Forecast holders and counters per quarter
    fore_yoy_cif = {1: [], 2: [], 3: [], 4: []}
    fore_yoy_spc = {1: [], 2: [], 3: [], 4: []}
    times_forecasted = {1: 0, 2: 0, 3: 0, 4: 0}

    per_idx = per
    levels_cif = {p: float(v) for p, v in zip(per_idx, g[cif_col]) if pd.notna(v)}
    levels_spc = {p: float(v) for p, v in zip(per_idx, g[spc_col]) if pd.notna(v)}

    # QoQ fallbacks (edge start if year-ago level missing)
    qo_q_cif = (g[cif_col] / g[cif_col].shift(1)).dropna()
    qo_q_spc = (g[spc_col] / g[spc_col].shift(1)).dropna()
    f_qoq_cif = float(qo_q_cif.iloc[-1]) if not qo_q_cif.empty and qo_q_cif.iloc[-1] > 0 else 1.0
    f_qoq_spc = float(qo_q_spc.iloc[-1]) if not qo_q_spc.empty and qo_q_spc.iloc[-1] > 0 else 1.0

    rows = []
    for h in range(1, H + 1):
        t = last_per + h
        q = t.quarter

        # Previous quarter levels (for delta %)
        prev_per = t - 1
        prev_cif = levels_cif.get(prev_per, cif_last)
        prev_spc = levels_spc.get(prev_per, spc_last)
        prev_ps  = scale * prev_cif * prev_spc

        # Determine window size for this quarter
        K = base_window + times_forecasted[q]
        # Pools: historical + already-forecasted YoY factors
        pool_cif = hist_yoy_cif[q] + fore_yoy_cif[q]
        pool_spc = hist_yoy_spc[q] + fore_yoy_spc[q]

        def avg_factor(pool: List[float], fallback: float) -> float:
            if len(pool) == 0:
                return fallback
            use = pool[-K:] if len(pool) >= K else pool
            f = robust_mean(use, median=use_median_A)
            return f if (np.isfinite(f) and f > 0) else fallback

        # If we have the year-ago quarter level, use YoY; else bridge once with QoQ
        if (t - 4) in levels_cif:
            f_cif = avg_factor(pool_cif, f_qoq_cif)
            cif_t = levels_cif[t - 4] * f_cif
            fore_yoy_cif[q].append(f_cif)
        else:
            cif_t = prev_cif * f_qoq_cif

        if (t - 4) in levels_spc:
            f_spc = avg_factor(pool_spc, f_qoq_spc)
            spc_t = levels_spc[t - 4] * f_spc
            fore_yoy_spc[q].append(f_spc)
        else:
            spc_t = prev_spc * f_qoq_spc

        levels_cif[t] = cif_t
        levels_spc[t] = spc_t
        ps_t = scale * cif_t * spc_t

        # --- NEW: deltas (QoQ %) rounded to 2 decimals
        d_cif = ((cif_t / prev_cif) - 1.0) * 100 if prev_cif not in (0, np.nan) else np.nan
        d_spc = ((spc_t / prev_spc) - 1.0) * 100 if prev_spc not in (0, np.nan) else np.nan
        d_ps  = ((ps_t  / prev_ps)  - 1.0) * 100 if prev_ps  not in (0, np.nan) else np.nan

        rows.append({
            "quarter": str(t),
            "bank": g["bank"].iloc[0],
            "method": "Rolling Seasonal (YoY avg)",
            "projected_cif_bn": cif_t,
            "projected_sales_per_cif_000": spc_t,
            "projected_purchase_sales_bn": ps_t,
            "delta_cif_pct": None if pd.isna(d_cif) else round(float(d_cif), 2),
            "delta_sales_per_cif_pct": None if pd.isna(d_spc) else round(float(d_spc), 2),
            "delta_purchase_sales_pct": None if pd.isna(d_ps)  else round(float(d_ps),  2),
        })

        times_forecasted[q] += 1

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
    proj_A = rolling_seasonal_projection_yoy(gbank, base_window=base_window_years, use_median=use_median_A, target_end=TARGET_END)
    proj_B = project_bank_latest_same_qtr_yoy(gbank, TARGET_END)
    if not proj_A.empty: proj_A_frames.append(proj_A)
    if not proj_B.empty: proj_B_frames.append(proj_B)

if not (proj_A_frames or proj_B_frames):
    st.info("No projections available (check mapping or banks).")
    st.stop()

proj_A = pd.concat(proj_A_frames, ignore_index=True) if proj_A_frames else pd.DataFrame()
proj_B = pd.concat(proj_B_frames, ignore_index=True) if proj_B_frames else pd.DataFrame()

# Round levels per UI (delta columns are already rounded 2 decimals above)
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
    # Build overlay: Actual + Selected Method
    metric_label = FRIENDLY.get(metric_code, metric_code)
    overlays = []

    # Actual series
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
metric_pick = st.selectbox("Metric to chart", options=[FRIENDLY["purchase_sales_bn"], FRIENDLY["cards_in_force_bn"], FRIENDLY["sales_per_cif_000"]],
                           index=0)
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_pick]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Method A — Rolling Seasonal (YoY avg)")
    chA = chart_method("Rolling Seasonal (YoY avg)", metric_code)
    if chA is None:
        st.info("No projections for Method A.")
    else:
        st.altair_chart(chA, use_container_width=True)

with col2:
    st.subheader("Method B — Latest same‑quarter YoY")
    chB = chart_method("Latest YoY", metric_code)
    if chB is None:
        st.info("No projections for Method B.")
    else:
        st.altair_chart(chB, use_container_width=True)

# =========================
# TABLES (with delta columns)
# =========================
def tidy_sort(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty: return dfp
    bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
    dfp["_b_sort"] = dfp["bank"].map(lambda x: bank_order_map.get(x, 999))
    dfp["_q_sort"] = dfp["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
    dfp = dfp.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])
    # Column order: quarter, bank, method, levels, deltas
    cols = ["quarter","bank","method",
            "projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
            "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"]
    # keep only those that exist
    cols = [c for c in cols if c in dfp.columns]
    return dfp[cols]

st.subheader("Projections Table — Method A (Rolling Seasonal YoY avg)")
if not proj_A.empty:
    st.dataframe(tidy_sort(proj_A), use_container_width=True)
else:
    st.info("No projections to show for Method A.")

st.subheader("Projections Table — Method B (Latest same‑quarter YoY)")
if not proj_B.empty:
    st.dataframe(tidy_sort(proj_B), use_container_width=True)
else:
    st.info("No projections to show for Method B.")

