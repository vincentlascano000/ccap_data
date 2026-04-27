# app.py
# CCAP — Method C ONLY
# True rolling same-quarter baseline
# PS with CIF & Sales/CIF as QoQ-driven projected drivers
# Baseline QoQ projections for CIF & SPC exposed in data view

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="CCAP — Method C Only", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")

BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

BANK_COLORS = {
    "UB": "#f28e2b",
    "BDO": "#4169E1",
    "RCBC": "#7ec8e3",
    "SECBANK": "#4CAF50",
    "MB": "#3B5B8A",
    "BPI": "#d62728",
}

# -------------------------
# HEADER MAP
# -------------------------
def _canon(s):
    s = str(s).lower()
    s = re.sub(r"[()’']", "", s)
    s = s.replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()

HEADER_MAP = {
    "quarter": "quarter",
    "bank": "bank",
    "purchase sales in bn": "purchase_sales_bn",
    "cards in force in bn": "cards_in_force_bn",
    "sales cif 000": "sales_per_cif_000",
}

def apply_header_map(df):
    ren = {}
    for c in df.columns:
        k = _canon(c)
        if k in HEADER_MAP:
            ren[c] = HEADER_MAP[k]
    return df.rename(columns=ren)

# -------------------------
# HELPERS
# -------------------------
def parse_quarter(x):
    s = str(x).upper().replace("/", " ").replace("-", " ")
    m = re.match(r"^([1-4])Q(\d{2,4})$", s)
    if m:
        q, y = int(m.group(1)), m.group(2)
        y = int("20" + y) if len(y) == 2 else int(y)
        return pd.Period(year=y, quarter=q, freq="Q").to_timestamp("end")
    return pd.NaT

def qoq_factors_by_quarter(series, dates):
    p = dates.dt.to_period("Q")
    s = pd.Series(series.values, index=p).sort_index()
    f = (s / s.shift(1)).dropna()
    out = {1: [], 2: [], 3: [], 4: []}
    for per, v in f.items():
        if np.isfinite(v) and v > 0:
            out[per.quarter].append(float(v))
    return out

# -------------------------
# UI
# -------------------------
st.title("CCAP — Method C (True Rolling QoQ)")
scenario = st.sidebar.radio("Scenario", ["Pessimistic", "Realistic", "Optimistic"], index=1)
scenario_shift_ppt = st.sidebar.slider("Scenario shift (±ppt)", 0.0, 10.0, 1.5, 0.1)
scenario_adj = (scenario_shift_ppt / 100) * (1 if scenario == "Optimistic" else -1 if scenario == "Pessimistic" else 0)

K = st.sidebar.slider("Rolling window (K)", 3, 8, 6)

# -------------------------
# LOAD DATA
# -------------------------
raw = pd.read_csv(RAW_URL)
df = apply_header_map(raw)

df["quarter_dt"] = df["quarter"].apply(parse_quarter)
for c in ["purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

panel = (
    df[["bank", "quarter_dt", "purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]]
    .dropna()
    .sort_values(["bank", "quarter_dt"])
)

banks = sorted(panel["bank"].unique(), key=lambda b: BANK_ORDER_PREF.index(b) if b in BANK_ORDER_PREF else 999)
banks_pick = st.multiselect("Banks", banks, default=banks)

panel = panel[panel["bank"].isin(banks_pick)]

# -------------------------
# COEFFICIENT FIT
# -------------------------
def fit_uplift(panel):
    g = panel.copy()
    g["qtr"] = g["quarter_dt"].dt.to_period("Q").apply(lambda p: p.quarter)
    g["d_ps"] = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    bases = []
    for b, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        out = []
        for _, r in gb.iterrows():
            base = np.mean(pools[r["qtr"]]) if pools[r["qtr"]] else np.nan
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

st.markdown(f"""
**Coefficients (pooled):**
- α (Intercept): `{alpha:.4f}`
- β CIF: `{beta_cif:.4f}`
- β Sales/CIF: `{beta_spc:.4f}`
""")

# -------------------------
# METHOD C PROJECTION (OPTION 2)
# -------------------------
def project_method_c(gb):
    gb = gb.sort_values("quarter_dt")
    last_per = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)

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
        t = last_per + h
        q = t.quarter

        g_base = np.mean((hist_ps[q] + fore_ps[q])[-K:]) - 1 if (hist_ps[q] + fore_ps[q]) else 0
        d_cif = np.mean((hist_cif[q] + fore_cif[q])[-K:]) - 1 if (hist_cif[q] + fore_cif[q]) else 0
        d_spc = np.mean((hist_spc[q] + fore_spc[q])[-K:]) - 1 if (hist_spc[q] + fore_spc[q]) else 0

        uplift = alpha + beta_cif * d_cif + beta_spc * d_spc
        g_ps = g_base + uplift + scenario_adj

        prev_ps = ps
        ps *= (1 + g_ps)
        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        fore_ps[q].append(ps / prev_ps)
        fore_cif[q].append(1 + d_cif)
        fore_spc[q].append(1 + d_spc)

        rows.append({
            "bank": gb.iloc[0]["bank"],
            "quarter": str(t),
            "projected_ps_bn": ps,
            "projected_cif_bn": cif,
            "projected_sales_per_cif_000": spc,
            "delta_cif_pct": d_cif,
            "delta_sales_per_cif_000_pct": d_spc,
        })

    return pd.DataFrame(rows)

# -------------------------
# RUN PROJECTIONS
# -------------------------
proj = pd.concat(
    [project_method_c(panel[panel["bank"] == b]) for b in banks_pick if len(panel[panel["bank"] == b]) >= 3],
    ignore_index=True
)

# -------------------------
# DATA VIEW
# -------------------------
st.subheader("Projection Data View (Method C)")

view = proj.copy()
view["delta_cif_pct"] = view["delta_cif_pct"] * 100
view["delta_sales_per_cif_000_pct"] = view["delta_sales_per_cif_000_pct"] * 100

st.dataframe(view, use_container_width=True)
