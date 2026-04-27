# app.py
# CCAP — Method C ONLY
# True rolling same-quarter baseline + CIF & Sales/CIF uplift + scenario shift
# Bank colors explicitly fixed

import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="CCAP — Method C Only", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")

BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

BANK_COLORS = {
    "UB": "#f28e2b",        # orange
    "BDO": "#4169E1",       # royal blue
    "RCBC": "#7ec8e3",      # light blue
    "SECBANK": "#2ca02c",   # green
    "MB": "#0b1c2d",        # navy (very dark blue)
    "BPI": "#d62728",       # red
}

# =========================
# HEADER AUTO-MAPPING
# =========================
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[()\[\]’'`]", "", s)
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

HEADER_MAP = {
    "quarter": "quarter",
    "bank": "bank",
    "purchase sales in bn": "purchase_sales_bn",
    "purchase sales bn": "purchase_sales_bn",
    "cards in force in bn": "cards_in_force_bn",
    "cards in force bn": "cards_in_force_bn",
    "sales cif 000": "sales_per_cif_000",
    "sales per cif 000": "sales_per_cif_000",
}

def apply_header_map(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = _canon(c)
        if k in HEADER_MAP:
            ren[c] = HEADER_MAP[k]
    return df.rename(columns=ren)

# =========================
# HELPERS
# =========================
def parse_quarter_token(x):
    if pd.isna(x):
        return None, pd.NaT
    s = str(x).upper().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)
    m = re.match(r"^([1-4])Q(\d{2,4})$", s)
    if m:
        q, yy = int(m.group(1)), m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        per = pd.Period(freq="Q", year=year, quarter=q)
        return str(per), per.to_timestamp(how="end")
    try:
        dt = pd.to_datetime(s)
        per = pd.Period(dt, freq="Q")
        return str(per), per.to_timestamp(how="end")
    except Exception:
        return None, pd.NaT

def to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(s2, errors="coerce")

def qoq_factors_by_quarter(series, dates):
    p = dates.dt.to_period("Q")
    s = pd.Series(series.values, index=p).sort_index()
    f = (s / s.shift(1)).dropna()
    out = {1: [], 2: [], 3: [], 4: []}
    for per, v in f.items():
        if np.isfinite(v) and v > 0:
            out[per.quarter].append(float(v))
    return out

# =========================
# UI
# =========================
st.title("CCAP — Method C Only (True Rolling Baseline)")
st.caption(f"Forecast end: **{TARGET_END}**")

st.sidebar.header("Scenario")
scenario = st.sidebar.radio("Scenario", ["Pessimistic", "Realistic", "Optimistic"], index=1)
scenario_shift_ppt = st.sidebar.slider("Scenario shift (±ppt)", 0.0, 10.0, 1.5, 0.1)
scenario_adj = (scenario_shift_ppt / 100.0) * (
    1 if scenario == "Optimistic" else (-1 if scenario == "Pessimistic" else 0)
)

st.sidebar.header("Method C")
K = st.sidebar.slider("Rolling same‑quarter window (K)", 3, 8, 6, 1)

# =========================
# LOAD DATA
# =========================
raw = pd.read_csv(RAW_URL)
df = apply_header_map(raw.copy())

parsed = df["quarter"].apply(parse_quarter_token)
df["quarter_dt"] = parsed.apply(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)

for c in ["purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]:
    df[c] = to_numeric(df[c])

panel = (
    df[["bank", "quarter_dt", "purchase_sales_bn",
        "cards_in_force_bn", "sales_per_cif_000"]]
    .dropna()
    .sort_values(["bank", "quarter_dt"])
)

banks_all = sorted(panel["bank"].unique(),
                   key=lambda b: BANK_ORDER_PREF.index(b)
                   if b in BANK_ORDER_PREF else 999)

banks_pick = st.multiselect("Banks", banks_all, default=banks_all)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================
# COEFFICIENT FIT
# =========================
def fit_uplift(panel):
    g = panel.copy()
    g["per"] = g["quarter_dt"].dt.to_period("Q")
    g["qtr"] = g["per"].apply(lambda p: p.quarter)

    g["d_ps"] = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    bases = []
    for b, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        out = []
        for _, r in gb.iterrows():
            pool = pools[r["qtr"]]
            base = np.mean(pool) if pool else np.nan
            out.append(base)
            if pd.notna(r["d_ps"]):
                pools[r["qtr"]].append(r["d_ps"])
        bases.append(pd.Series(out, index=gb.index))

    g["g_base"] = pd.concat(bases).sort_index()
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([np.ones(len(fit)), fit["d_cif"], fit["d_spc"]])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

alpha, beta_cif, beta_spc = fit_uplift(panel)

with st.expander("Uplift coefficients", expanded=True):
    st.markdown(f"""
- **Intercept (α)**: **{alpha:.4f}**  
- **β (Δ%CIF)**: **{beta_cif:.4f}**  
- **β (Δ% Sales/CIF)**: **{beta_spc:.4f}**  
- **Scenario shift**: **{scenario_shift_ppt:.1f} ppt / quarter**
""")

# =========================
# METHOD C PROJECTION
# =========================
def project_method_c(gb):
    gb = gb.sort_values("quarter_dt").copy()
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last.year) * 4 + (TARGET_END.quarter - last.quarter)
    if H <= 0:
        return pd.DataFrame()

    hist_ps = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cards_in_force_bn"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["sales_per_cif_000"], gb["quarter_dt"])

    fore_ps = {q: [] for q in range(1, 5)}
    fore_cif = {q: [] for q in range(1, 5)}
    fore_spc = {q: [] for q in range(1, 5)}

    ps = gb.iloc[-1]["purchase_sales_bn"]
    cif = gb.iloc[-1]["cards_in_force_bn"]
    spc = gb.iloc[-1]["sales_per_cif_000"]

    rows = []
    for h in range(1, H + 1):
        t = last + h
        q = t.quarter

        base_ps = np.mean((hist_ps[q] + fore_ps[q])[-K:]) if (hist_ps[q] + fore_ps[q]) else 1
        g_base = base_ps - 1

        d_cif = (np.mean((hist_cif[q] + fore_cif[q])[-K:]) - 1) if (hist_cif[q] + fore_cif[q]) else 0
        d_spc = (np.mean((hist_spc[q] + fore_spc[q])[-K:]) - 1) if (hist_spc[q] + fore_spc[q]) else 0

        uplift = alpha + beta_cif * d_cif + beta_spc * d_spc
        g_total = g_base + uplift + scenario_adj

        prev_ps = ps
        ps *= (1 + g_total)
        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        fore_ps[q].append(ps / prev_ps)
        fore_cif[q].append(1 + d_cif)
        fore_spc[q].append(1 + d_spc)

        rows.append({
            "bank": gb.iloc[0]["bank"],
            "quarter": str(t),
            "value": ps,
            "scenario": "Method C"
        })

    return pd.DataFrame(rows)

# =========================
# RUN & CHART
# =========================
# --- Ensure historical data has quarter_dt ---
# =========================
# =========================
# CHART — Quarter labels as YYYY-Q#
# =========================

# =========================
# CHART — Quarter labels as YYYY-Q#
# (self-contained, no dependency on pre-existing `proj`)
# =========================

# --- Build historical actuals ---
hist_plot = (
    panel
    .assign(
        value=lambda d: d["purchase_sales_bn"],
        scenario="Actual"
    )
    [["bank", "quarter_dt", "value", "scenario"]]
)

# --- Build projections locally (Method C) ---
proj_plot = pd.concat(
    [
        project_method_c(panel[panel["bank"] == b])
        for b in banks_pick
        if len(panel[panel["bank"] == b]) >= 3
    ],
    ignore_index=True
)

proj_plot["quarter_dt"] = proj_plot["quarter"].apply(
    lambda q: pd.Period(q, freq="Q").to_timestamp(how="end")
)

proj_plot = proj_plot[["bank", "quarter_dt", "value", "scenario"]]

# --- Combine actuals + projections ---
plot_df = pd.concat([hist_plot, proj_plot], ignore_index=True)

# --- Create formatted quarter labels (e.g., 2026-Q1) ---
plot_df["quarter_label"] = (
    plot_df["quarter_dt"]
    .dt.to_period("Q")
    .astype(str)
    .str.replace("Q", "-Q")
)

# --- Chart ---
chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "quarter_label:N",
            title="Quarter",
            sort=alt.SortField(
                field="quarter_dt",
                order="ascending"
            )
        ),
        y=alt.Y("value:Q", title="Purchase Sales"),
        color=alt.Color(
            "bank:N",
            scale=alt.Scale(
                domain=list(BANK_COLORS.keys()),
                range=list(BANK_COLORS.values())
            ),
            legend=alt.Legend(title="Bank")
        ),
        strokeDash=alt.condition(
            alt.datum.scenario == "Actual",
            alt.value([0]),      # solid
            alt.value([6, 4])    # dashed
        ),
        tooltip=["bank", "quarter_label", "value", "scenario"]
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)
