# app.py
# CCAP — Two PS baselines + CIF/SPC uplift, with Scenario Shift fixed; no lookback, no uplift cap.
# Method A: Baseline & drivers use the AVERAGE of all historical same-quarter QoQ factors (balanced).
# Method B: Baseline & drivers use the LATEST same-quarter QoQ factor (latest trend).
# Coefficients are pooled (residualized vs a simple expanding same-quarter baseline); Balances are NOT used.
# Scenario Shift is added to PS growth each projected quarter.

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

def qoq_factors_by_quarter(series: pd.Series, periods: pd.Series) -> Dict[int, List[float]]:
    """Dict quarter->list of QoQ factors f = X_t / X_{t-1} for quarter(t)==q."""
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

# =========================
# UI
# =========================
st.set_page_config(page_title="CCAP — Two PS Baselines + CIF/SPC uplift", layout="wide")
st.title("CCAP — PS Baselines with CIF & Sales/CIF Uplift (Methods A & B)")
st.caption(f"Forecast end: **{str(TARGET_END)}**")

st.sidebar.header("General")
round_dec = st.sidebar.selectbox("Round decimals (levels)", options=[0,1,2], index=1)

st.sidebar.header("Scenario")
scenario = st.sidebar.radio("Scenario", ["Pessimistic","Realistic","Optimistic"], index=1, horizontal=True)
scenario_shift_ppt = st.sidebar.slider("Scenario shift (±ppt added to PS growth each projected quarter)", 0.0, 10.0, 1.5, 0.1)
scenario_adj_prop = (scenario_shift_ppt/100.0) * (1 if scenario=="Optimistic" else (-1 if scenario=="Pessimistic" else 0))

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

# Column mapping for PS/CIF/SPC
st.sidebar.header("Column mapping (if needed)")
def ensure_col(name_key: str, label: str) -> str:
    if name_key in df.columns:
        return name_key
    cands = candidate_cols(df, COLUMN_SYNONYMS[name_key]) or list(df.columns)
    return st.sidebar.selectbox(f"Select column for {label}", options=cands, index=0)

cif_col = ensure_col("cards_in_force_bn", FRIENDLY["cards_in_force_bn"])
spc_col = ensure_col("sales_per_cif_000", FRIENDLY["sales_per_cif_000"])
if "purchase_sales_bn" not in df.columns:
    cands = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"])
    if cands:
        sel = st.sidebar.selectbox("Select column for Purchase Sales", options=cands, index=0)
        df["purchase_sales_bn"] = df[sel]

# Coerce numeric
for c in [cif_col, spc_col, "purchase_sales_bn"]:
    df[c] = to_numeric(df[c])

panel = (df[["bank","quarter_dt", cif_col, spc_col, "purchase_sales_bn"]]
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
# COEFFICIENT ESTIMATION (pooled) — residual vs simple expanding same-quarter baseline
# =========================
def fit_uplift_coefs(panel_bank: pd.DataFrame) -> Tuple[float,float,float,pd.DataFrame]:
    """
    Pooled OLS on residual PS growth vs Δ%CIF and Δ%SPC.
    Baseline for residualization: an expanding same-quarter average of d_ps (no tunable lookback).
    """
    g = panel_bank.sort_values(["bank","quarter_dt"]).copy()
    g["per"] = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = g["per"].apply(lambda p: p.quarter)

    # QoQ % changes
    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")[cif_col].pct_change()
    g["d_spc"] = g.groupby("bank")[spc_col].pct_change()

    # Build baseline growth (expanding same-quarter average WITHOUT any slider)
    base_g = []
    for b, gb in g.groupby("bank"):
        pools = {1:[],2:[],3:[],4:[]}
        q_b   = gb["qtr"].values
        dps   = gb["d_ps"].values
        base_series = []
        for i in range(len(gb)):
            q = q_b[i]
            pool = pools[q]
            if len(pool) == 0:
                g_base = np.nan
            else:
                g_base = float(np.mean(pool))
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

    # Gentle winsorize to reduce single-quarter outliers
    for col in ["r_ps","d_cif","d_spc"]:
        fit[col] = fit[col].clip(lower=-0.5, upper=0.5)

    X = np.column_stack([np.ones(len(fit)), fit["d_cif"].to_numpy(), fit["d_spc"].to_numpy()])
    y = fit["r_ps"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, b_cif, b_spc = float(beta[0]), float(beta[1]), float(beta[2])

    return b_cif, b_spc, intercept, fit

b_cif, b_spc, b_int, fit_df = fit_uplift_coefs(panel[panel["bank"].isin(banks_pick)])

# =========================
# PROJECTIONS
# =========================
def average_same_quarter_factor(hist_dict: Dict[int, List[float]], q: int) -> float:
    vals = hist_dict.get(q, [])
    if not vals:
        return 1.0
    m = float(np.mean([v for v in vals if np.isfinite(v) and v > 0]))
    return m if (np.isfinite(m) and m > 0) else 1.0

def project_method_A(bank_df: pd.DataFrame, scenario_adj_prop: float) -> pd.DataFrame:
    """
    Method A (Balanced averages):
      Baseline PS growth = mean of ALL historical same-quarter QoQ factors (fixed per quarter).
      Driver growths (Δ%CIF, Δ%SPC) = mean of ALL historical same-quarter QoQ factors (fixed per quarter).
      g_total = g_base + (intercept + b_cif*d_cif + b_spc*d_spc) + scenario_adj_prop
    """
    gb = bank_df.sort_values("quarter_dt").copy()
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    # Historical QoQ factor pools (PS/CIF/SPC)
    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb[cif_col], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb[spc_col], gb["quarter_dt"])

    level_ps  = float(gb.iloc[-1]["purchase_sales_bn"])
    level_cif = float(gb.iloc[-1][cif_col])
    level_spc = float(gb.iloc[-1][spc_col])

    rows = []
    for h in range(1, H+1):
        t = last_per + h; q = t.quarter
        prev_ps, prev_cif, prev_spc = level_ps, level_cif, level_spc

        # Baseline growth (proportion) from historical average factor
        f_ps_base = average_same_quarter_factor(hist_ps, q)
        g_base = f_ps_base - 1.0

        # Driver % changes (proportion) from historical average factors
        f_cif = average_same_quarter_factor(hist_cif, q)
        f_spc = average_same_quarter_factor(hist_spc, q)
        d_cif = f_cif - 1.0
        d_spc = f_spc - 1.0

        # Uplift (NO cap) + Scenario Shift
        uplift = b_int + b_cif*d_cif + b_spc*d_spc
        g_total = g_base + uplift + scenario_adj_prop

        # Evolve levels
        level_ps  *= (1.0 + g_total)
        level_cif *= (1.0 + d_cif)
        level_spc *= (1.0 + d_spc)

        # QoQ deltas (%)
        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc_pct = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan

        rows.append({
            "quarter": str(t),
            "bank": gb["bank"].iloc[0],
            "method": "Method A (Avg QoQ + uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "delta_purchase_sales_pct": round(d_ps_pct,2)  if pd.notna(d_ps_pct)  else None,
            "delta_cif_pct":            round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_sales_per_cif_pct":  round(d_spc_pct,2) if pd.notna(d_spc_pct) else None,
            "ps_uplift_pp":             round(uplift*100.0, 2),
            "ps_uplift_multiplier":     round(1.0 + uplift, 4),
            "scenario_shift_pp":        round(scenario_adj_prop*100.0, 2),
        })
    return pd.DataFrame(rows)

def project_method_B(bank_df: pd.DataFrame, scenario_adj_prop: float) -> pd.DataFrame:
    """
    Method B (Latest trend):
      Baseline PS growth = LATEST same-quarter QoQ factor.
      Driver % changes (Δ%CIF, Δ%SPC) = LATEST same-quarter QoQ factors.
      g_total = g_base + (intercept + b_cif*d_cif + b_spc*d_spc) + scenario_adj_prop
    """
    gb = bank_df.sort_values("quarter_dt").copy()
    last_dt  = gb["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0: return pd.DataFrame()

    latest_ps  = latest_same_quarter_qoq(gb["purchase_sales_bn"], gb["quarter_dt"])
    latest_cif = latest_same_quarter_qoq(gb[cif_col], gb["quarter_dt"])
    latest_spc = latest_same_quarter_qoq(gb[spc_col], gb["quarter_dt"])

    level_ps  = float(gb.iloc[-1]["purchase_sales_bn"])
    level_cif = float(gb.iloc[-1][cif_col])
    level_spc = float(gb.iloc[-1][spc_col])

    rows = []
    for h in range(1, H+1):
        t = last_per + h; q = t.quarter
        prev_ps, prev_cif, prev_spc = level_ps, level_cif, level_spc

        f_ps_base = latest_ps.get(q, 1.0); f_ps_base = f_ps_base if (np.isfinite(f_ps_base) and f_ps_base>0) else 1.0
        g_base = f_ps_base - 1.0

        f_cif = latest_cif.get(q, 1.0); f_cif = f_cif if (np.isfinite(f_cif) and f_cif>0) else 1.0
        f_spc = latest_spc.get(q, 1.0); f_spc = f_spc if (np.isfinite(f_spc) and f_spc>0) else 1.0
        d_cif = f_cif - 1.0
        d_spc = f_spc - 1.0

        uplift = b_int + b_cif*d_cif + b_spc*d_spc
        g_total = g_base + uplift + scenario_adj_prop

        level_ps  *= (1.0 + g_total)
        level_cif *= (1.0 + d_cif)
        level_spc *= (1.0 + d_spc)

        d_ps_pct  = (level_ps/prev_ps  - 1)*100 if prev_ps  else np.nan
        d_cif_pct = (level_cif/prev_cif - 1)*100 if prev_cif else np.nan
        d_spc_pct = (level_spc/prev_spc - 1)*100 if prev_spc else np.nan

        rows.append({
            "quarter": str(t),
            "bank": gb["bank"].iloc[0],
            "method": "Method B (Latest QoQ + uplift)",
            "projected_purchase_sales_bn": level_ps,
            "projected_cif_bn": level_cif,
            "projected_sales_per_cif_000": level_spc,
            "delta_purchase_sales_pct": round(d_ps_pct,2)  if pd.notna(d_ps_pct)  else None,
            "delta_cif_pct":            round(d_cif_pct,2) if pd.notna(d_cif_pct) else None,
            "delta_sales_per_cif_pct":  round(d_spc_pct,2) if pd.notna(d_spc_pct) else None,
            "ps_uplift_pp":             round(uplift*100.0, 2),
            "ps_uplift_multiplier":     round(1.0 + uplift, 4),
            "scenario_shift_pp":        round(scenario_adj_prop*100.0, 2),
        })
    return pd.DataFrame(rows)

# =========================
# RUN BOTH METHODS
# =========================
proj_A_frames, proj_B_frames, hist_frames = [], [], []
for b in banks_pick:
    gb = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col, "purchase_sales_bn"]]
    if gb.shape[0] < 3:
        continue
    hist_frames.append(gb.assign(method="Actual"))
    pa = project_method_A(gb, scenario_adj_prop)
    pb = project_method_B(gb, scenario_adj_prop)
    if not pa.empty: proj_A_frames.append(pa)
    if not pb.empty: proj_B_frames.append(pb)

proj_A = pd.concat(proj_A_frames, ignore_index=True) if proj_A_frames else pd.DataFrame()
proj_B = pd.concat(proj_B_frames, ignore_index=True) if proj_B_frames else pd.DataFrame()

# Round level columns per UI
for dfp in [proj_A, proj_B]:
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
"""
    )

# =========================
# CHARTS — Method A & Method B overlays (no guides below)
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
    options=[FRIENDLY["purchase_sales_bn"], FRIENDLY["cards_in_force_bn"], FRIENDLY["sales_per_cif_000"]],
    index=0
)
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_pick]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Method A — Avg same‑quarter QoQ baseline + uplift")
    chA = chart_method("Method A (Avg QoQ + uplift)", metric_code)
    st.altair_chart(chA, use_container_width=True) if chA else st.info("No projections for Method A.")
with col2:
    st.subheader("Method B — Latest same‑quarter QoQ baseline + uplift")
    chB = chart_method("Method B (Latest QoQ + uplift)", metric_code)
    st.altair_chart(chB, use_container_width=True) if chB else st.info("No projections for Method B.")

# =========================
# TABLES — Method A & Method B (with % deltas and uplift columns)
# =========================
def format_percent_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: (f"{x:.2f}%" if pd.notna(x) else None))
    return out

def tidy_sort_percent(dfp: pd.DataFrame, cols_pct: List[str]) -> pd.DataFrame:
    if dfp.empty: return dfp
    bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
    dfp["_b_sort"] = dfp["bank"].map(lambda x: bank_order_map.get(x, 999))
    dfp["_q_sort"] = dfp["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
    dfp = dfp.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])
    dfp = format_percent_cols(dfp, cols_pct)
    return dfp

st.subheader("Projections Table — Method A (Avg QoQ + uplift)")
if not proj_A.empty:
    colsA = ["quarter","bank","method",
             "projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
             "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct",
             "ps_uplift_pp","ps_uplift_multiplier","scenario_shift_pp"]
    show_A = tidy_sort_percent(proj_A[colsA], ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"])
    show_A["ps_uplift_pp"]         = show_A["ps_uplift_pp"].apply(lambda x: f"{x:.2f} pp" if pd.notna(x) else None)
    show_A["ps_uplift_multiplier"] = show_A["ps_uplift_multiplier"].apply(lambda x: f"{x:.4f} ×" if pd.notna(x) else None)
    show_A["scenario_shift_pp"]    = show_A["scenario_shift_pp"].apply(lambda x: f"{x:.2f} pp" if pd.notna(x) else None)
    st.dataframe(show_A, use_container_width=True)
else:
    st.info("No projections to show for Method A.")

st.subheader("Projections Table — Method B (Latest QoQ + uplift)")
if not proj_B.empty:
    colsB = ["quarter","bank","method",
             "projected_purchase_sales_bn","projected_cif_bn","projected_sales_per_cif_000",
             "delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct",
             "ps_uplift_pp","ps_uplift_multiplier","scenario_shift_pp"]
    show_B = tidy_sort_percent(proj_B[colsB], ["delta_purchase_sales_pct","delta_cif_pct","delta_sales_per_cif_pct"])
    show_B["ps_uplift_pp"]         = show_B["ps_uplift_pp"].apply(lambda x: f"{x:.2f} pp" if pd.notna(x) else None)
    show_B["ps_uplift_multiplier"] = show_B["ps_uplift_multiplier"].apply(lambda x: f"{x:.4f} ×" if pd.notna(x) else None)
    show_B["scenario_shift_pp"]    = show_B["scenario_shift_pp"].apply(lambda x: f"{x:.2f} pp" if pd.notna(x) else None)
    st.dataframe(show_B, use_container_width=True)
else:
    st.info("No projections to show for Method B.")
