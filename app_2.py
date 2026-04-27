# app_2.py
# CCAP — Method C ONLY
# Clean UI: no raw quarter column shown
# Stable quarter labels (1Q26, 4Q25, etc.)

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="CCAP — Method C", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")

REGIME_SHIFT_PPT = 6.0

ONE_TIME_REANCHOR = {
    "BDO": 1.025,  # +2.5%
    "BPI": 1.020,  # +2.0%
}

BANK_COLORS = {
    "UB": "#f28e2b",
    "BDO": "#4169E1",
    "BPI": "#d62728",
    "SECBANK": "#4CAF50",
    "MB": "#3B5B8A",
    "RCBC": "#7ec8e3",
}

# =========================
# QUARTER HELPERS
# =========================
def parse_quarter_dt(value):
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip().upper().replace(" ", "")

    # 1Q23 / 4Q25
    if s[0].isdigit() and "Q" in s:
        q = int(s[0])
        y = int(s[2:])
        y = 2000 + y if y < 100 else y
        return pd.Period(year=y, quarter=q, freq="Q").to_timestamp(how="end")

    # 2026Q1
    if len(s) == 6 and s[:4].isdigit():
        return pd.Period(
            year=int(s[:4]),
            quarter=int(s[-1]),
            freq="Q"
        ).to_timestamp(how="end")

    return pd.to_datetime(s).to_period("Q").to_timestamp(how="end")

def period_to_qyy(period):
    return f"{period.quarter}Q{str(period.year)[-2:]}"

# =========================
# QoQ FACTORS
# =========================
def qoq_factors_by_quarter(series, quarter_dt):
    per = quarter_dt.dt.to_period("Q")
    s = pd.Series(series.values, index=per).sort_index()
    f = (s / s.shift(1)).dropna()

    out = {1: [], 2: [], 3: [], 4: []}
    for p, v in f.items():
        if np.isfinite(v) and v > 0:
            out[p.quarter].append(float(v))
    return out

# =========================
# UI
# =========================
st.title("CCAP — Method C (Regime‑Adjusted, Re‑Anchored)")
K = st.sidebar.slider("Rolling same‑quarter window (K)", 3, 8, 6)

# =========================
# LOAD DATA
# =========================
raw = pd.read_csv(RAW_URL, engine="python").rename(columns={
    "QUARTER": "quarter",
    "BANK": "bank",
    "Purchase Sales (in Bn)": "purchase_sales_bn",
    "Cards in Force (in Bn)": "cards_in_force_bn",
    "Sales / CIF ('000)": "sales_per_cif_000",
})

raw = raw[
    ["quarter", "bank",
     "purchase_sales_bn",
     "cards_in_force_bn",
     "sales_per_cif_000"]
]

raw["quarter_dt"] = raw["quarter"].apply(parse_quarter_dt)

for c in ["purchase_sales_bn", "cards_in_force_bn", "sales_per_cif_000"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

panel = (
    raw.dropna()
       .sort_values(["bank", "quarter_dt"])
       .reset_index(drop=True)
)

banks = panel["bank"].unique().tolist()
banks_pick = st.multiselect("Banks", banks, default=banks)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================
# FIT COEFFICIENTS
# =========================
def fit_uplift(panel_bank):
    g = panel_bank.copy()
    g["q"] = g["quarter_dt"].dt.to_period("Q").dt.quarter

    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    bases = []
    for _, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        base = []
        for _, r in gb.iterrows():
            base.append(np.mean(pools[r["q"]]) if pools[r["q"]] else np.nan)
            if pd.notna(r["d_ps"]):
                pools[r["q"]].append(r["d_ps"])
        bases.append(pd.Series(base, index=gb.index))

    g["g_base"] = pd.concat(bases).sort_index()
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([np.ones(len(fit)), fit["d_cif"], fit["d_spc"]])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0], beta[1], beta[2]

alpha_raw, beta_cif, beta_spc = fit_uplift(panel)
alpha = alpha_raw + REGIME_SHIFT_PPT / 100

# =========================
# METHOD C — ONE‑TIME RE‑ANCHOR
# =========================
def project_method_C(gb):
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year-last.year)*4 + (TARGET_END.quarter-last.quarter)
    if H <= 0:
        return pd.DataFrame()

    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cards_in_force_bn"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["sales_per_cif_000"], gb["quarter_dt"])

    fore_ps = {q: [] for q in range(1, 5)}
    fore_cif = {q: [] for q in range(1, 5)}
    fore_spc = {q: [] for q in range(1, 5)}

    ps  = gb.iloc[-1]["purchase_sales_bn"]
    cif = gb.iloc[-1]["cards_in_force_bn"]
    spc = gb.iloc[-1]["sales_per_cif_000"]
    bank = gb.iloc[0]["bank"]

    anchored = False
    rows = []

    for h in range(1, H + 1):
        t = last + h
        q = t.quarter
        prev_ps = ps

        g_base = np.mean((hist_ps[q] + fore_ps[q])[-K:]) - 1 if hist_ps[q] + fore_ps[q] else 0
        d_cif  = np.mean((hist_cif[q] + fore_cif[q])[-K:]) - 1 if hist_cif[q] + fore_cif[q] else 0
        d_spc  = np.mean((hist_spc[q] + fore_spc[q])[-K:]) - 1 if hist_spc[q] + fore_spc[q] else 0

        g_ps = g_base + (alpha + beta_cif*d_cif + beta_spc*d_spc)
        ps *= (1 + g_ps)

        if not anchored and bank in ONE_TIME_REANCHOR:
            ps *= ONE_TIME_REANCHOR[bank]
            anchored = True

        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        rows.append({
            "quarter_dt": t.to_timestamp(how="end"),
            "quarter_label": period_to_qyy(t),
            "bank": bank,
            "purchase_sales_bn": ps,
            "cards_in_force_bn": cif,
            "sales_per_cif_000": spc,
            "scenario": "Method C",
        })

        fore_ps[q].append(ps / prev_ps)
        fore_cif[q].append(1 + d_cif)
        fore_spc[q].append(1 + d_spc)

    return pd.DataFrame(rows)

proj = pd.concat(
    [
        project_method_C(panel[panel["bank"] == b])
        for b in banks_pick
        if panel[panel["bank"] == b].shape[0] >= 3
    ],
    ignore_index=True
)

# =========================
# CHART DATA
# =========================
hist = panel.assign(
    quarter_label=panel["quarter"],
    scenario="Actual"
)

plot_df = pd.concat([hist, proj], ignore_index=True)

# =========================
# DISPLAY TABLE (NO raw quarter!)
# =========================
display_df = plot_df[
    [
        "quarter_label",
        "bank",
        "purchase_sales_bn",
        "cards_in_force_bn",
        "sales_per_cif_000",
        "scenario",
    ]
]

st.subheader("Projected & Actual Values")
st.dataframe(display_df)

# =========================
# CHART
# =========================
chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "quarter_label:N",
            sort=alt.SortField("quarter_dt", order="ascending"),
            title="Quarter"
        ),
        y=alt.Y("purchase_sales_bn:Q", title="Purchase Sales (Bn)"),
        color=alt.Color(
            "bank:N",
            scale=alt.Scale(
                domain=list(BANK_COLORS.keys()),
                range=list(BANK_COLORS.values())
            )
        ),
        strokeDash=alt.condition(
            alt.datum.scenario == "Actual",
            alt.value([0]),
            alt.value([6,4])
        ),
        tooltip=[
            "bank",
            "purchase_sales_bn",
            "cards_in_force_bn",
            "sales_per_cif_000"
        ]
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)
