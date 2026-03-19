# app.py
# CCAP — Simple Time Series Baseline (QoQ + Seasonality), per bank
# • Forecasts Purchase Sales (Bn) only
# • Trend = average QoQ % change over a window you choose
# • Seasonality = average quarterly (Q1..Q4) deviation from trend
# • Projections extend to 2028 Q4 (inclusive), per bank
# • No cards/balances used

import re
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# 0) SOURCE & LABELS
# =========================================
RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)"
}

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
    if pd.isna(value):
        return None, pd.NaT
    s = str(value).strip().upper().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"^([1-4])Q(\d{2,4})$", s)  # 1Q23, 4Q2025
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
st.set_page_config(page_title="CCAP — Simple TS Baseline (QoQ + Seasonality)", layout="wide")
st.title("CCAP — Simple Time Series Baseline (QoQ + Seasonality)")

# Controls
st.sidebar.header("Forecast settings")
window_q  = st.sidebar.slider("Window for stats (quarters)", 4, 16, 8, 1)
season_wt = st.sidebar.slider("Seasonality strength (0 = off, 1 = full)", 0.0, 1.0, 1.0, 0.1)
table_h_q = st.sidebar.slider("Numbers table: next N projected quarters", 4, 12, 8, 1)
use_median = st.sidebar.checkbox("Use median (instead of mean) for QoQ & seasonal", value=False)

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

# Purchase Sales column mapping (if needed)
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
# 4) CORE TIME SERIES LOGIC (QoQ + Seasonality to TARGET_END)
# =========================================
def robust_stat(x: pd.Series, median=False):
    return float(x.median()) if median else float(x.mean())

def seasonal_forecast_to_target(gbank: pd.DataFrame,
                                window_q: int,
                                season_strength: float,
                                use_median: bool,
                                target_end: pd.Period) -> pd.DataFrame:
    """
    Baseline per bank to a fixed target quarter:
      1) QoQ % change series r_t = (X_t/X_{t-1}) - 1
      2) Trend growth g = mean(r_t) over last window_q quarters
      3) Seasonal deviation by quarter k in {1..4}:
            s_k = mean(r_t | quarter==k) - g   (over same window)
         Then shrink: s_k <- season_strength * s_k
      4) Let last actual quarter be P_last; forecast for H = (target_end - P_last) quarters:
            For h in 1..H:
                q_next = quarter of P_last + h
                growth = g + s_{q_next}
                X_{t+1} = X_t * (1 + growth)
    """
    gb = gbank.sort_values("quarter_dt").copy()
    gb["qtr"] = gb["quarter_dt"].dt.quarter
    gb["pct"] = gb["purchase_sales_bn"].pct_change()

    # Window subset
    last_dt = gb["quarter_dt"].max()
    last_per = last_dt.to_period("Q")
    H = int((target_end - last_per))  # number of quarters to project (0 or more)
    if H <= 0:
        # Nothing to project; return empty projections
        return pd.DataFrame(columns=["quarter_dt","quarter","bank","scenario","purchase_sales_bn"])

    winmask = gb["quarter_dt"] > (last_dt - pd.offsets.QuarterEnd(window_q))
    gwin = gb[winmask].copy()

    # Trend (mean/median QoQ %)
    g_series = gwin["pct"].dropna()
    g = robust_stat(g_series, median=use_median) if g_series.size > 0 else 0.0

    # Seasonality (by observed quarter)
    season_map = {}
    for k in [1, 2, 3, 4]:
        sk = gwin.loc[gwin["qtr"] == k, "pct"].dropna()
        if sk.size >= 1:
            season_map[k] = robust_stat(sk, median=use_median) - g
        else:
            season_map[k] = 0.0
    for k in season_map:
        season_map[k] = season_strength * season_map[k]

    # Forecast loop to TARGET_END
    level = float(gb.iloc[-1]["purchase_sales_bn"])
    levels = []
    for h in range(1, H + 1):
        next_per = last_per + h
        next_dt  = next_per.to_timestamp(how="end")
        next_q   = next_per.quarter
        growth   = g + season_map.get(next_q, 0.0)
        growth   = max(growth, -0.9)  # avoid negative collapse
        level    = level * (1.0 + growth)
        levels.append({"quarter_dt": next_dt, "quarter": str(next_per), "purchase_sales_bn": level})

    out = pd.DataFrame(levels)
    out["bank"] = gb["bank"].iloc[0]
    out["scenario"] = "Baseline"
    return out

# Build actual + projections for selected banks
plot_frames = []
table_frames = []
for b in banks_pick:
    gbank = panel[panel["bank"] == b]
    if gbank.shape[0] < 2:
        continue
    proj_b = seasonal_forecast_to_target(
        gbank=gbank,
        window_q=window_q,
        season_strength=season_wt,
        use_median=use_median,
        target_end=TARGET_END,
    )
    # Actual
    hist_b = gbank.assign(scenario="Actual")
    hist_b = hist_b.rename(columns={"purchase_sales_bn":"value"})
    hist_b = hist_b[["bank","quarter_dt","value","scenario"]]
    # Projections
    if not proj_b.empty:
        proj_plot = proj_b.rename(columns={"purchase_sales_bn":"value"})[["bank","quarter_dt","value","scenario"]]
        plot_frames += [hist_b, proj_plot]
        table_frames.append(
            proj_b.assign(value=np.round(proj_b["purchase_sales_bn"], 1))[["quarter_dt","quarter","bank","scenario","value"]]
        )
    else:
        plot_frames += [hist_b]

if not plot_frames:
    st.info("Not enough history to build projections. Try selecting other banks or reducing the window.")
    st.stop()

overlay = pd.concat(plot_frames, ignore_index=True)
proj_table = pd.concat(table_frames, ignore_index=True) if table_frames else pd.DataFrame(
    columns=["quarter_dt","quarter","bank","scenario","value"]
)

# =========================================
# 5) CHART — Consolidated overlay (Actual solid, Baseline dashed)
# =========================================
color_scale = alt.Scale(scheme='tableau10')

line_actual = (
    alt.Chart(overlay)
      .transform_filter(alt.datum.scenario == "Actual")
      .mark_line(point=True, strokeWidth=2)
      .encode(
          x=alt.X("quarter_dt:T", title="Quarter", axis=alt.Axis(format="%Y Q%q")),
          y=alt.Y("value:Q", title=FRIENDLY["purchase_sales_bn"]),
          color=alt.Color("bank:N", title="Bank",
                          sort=banks_pick, scale=color_scale),
          tooltip=[
              alt.Tooltip("bank:N", title="Bank"),
              alt.Tooltip("quarter_dt:T", title="Quarter", format="%Y Q%q"),
              alt.Tooltip("value:Q", title=FRIENDLY["purchase_sales_bn"], format=",.2f"),
          ]
      )
      .properties(height=380)
)

line_proj = (
    alt.Chart(overlay)
      .transform_filter(alt.datum.scenario != "Actual")
      .mark_line(point=False, strokeWidth=2, strokeDash=[6,4])
      .encode(
          x="quarter_dt:T",
          y="value:Q",
          color=alt.Color("bank:N", title="Bank",
                          sort=banks_pick, scale=color_scale),
          tooltip=[
              alt.Tooltip("bank:N", title="Bank"),
              alt.Tooltip("quarter_dt:T", title="Quarter", format="%Y Q%q"),
              alt.Tooltip("value:Q", title=FRIENDLY["purchase_sales_bn"], format=",.2f"),
              alt.Tooltip("scenario:N", title="Scenario")
          ]
      )
      .properties(height=380)
)

st.subheader("Purchase Sales (Bn) — Actual vs Baseline (trend + seasonality) — Forecast to 2028 Q4")
st.altair_chart(alt.layer(line_actual, line_proj), use_container_width=True)

# =========================================
# 6) NUMBERS TABLE — rounded projected volumes
# =========================================
if not proj_table.empty:
    tbl = proj_table.sort_values(["quarter_dt","bank","scenario"]).copy()
    upcoming_quarters = sorted(tbl["quarter_dt"].unique())[:table_h_q]
    tbl = (tbl[tbl["quarter_dt"].isin(upcoming_quarters)]
           .rename(columns={"value": f"{FRIENDLY['purchase_sales_bn']} (~)"})
           .loc[:, ["quarter","bank","scenario", f"{FRIENDLY['purchase_sales_bn']} (~)"]])
    st.subheader("Projected volumes (rough estimates)")
    st.dataframe(tbl, use_container_width=True)
else:
    st.info("No projections available to tabulate (check data/window).")
