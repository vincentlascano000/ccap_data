# app_2.py
# CCAP — Method C ONLY
# True rolling same‑quarter QoQ
# Regime‑corrected intercept + large‑bank premium
# Raw QUARTER labels + fixed colors

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

# regime + bank‑specific adjustments (validated empirically)
REGIME_SHIFT_PPT = 6.0          # applies to all banks
LARGE_BANK_EXTRA_PPT = 5.5      # additional for BDO & BPI
LARGE_BANKS = {"BDO", "BPI"}

BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

BANK_COLORS = {
    "UB": "#f28e2b",
    "BDO": "#4169E1",
    "BPI": "#d62728",
    "SECBANK": "#4CAF50",
    "MB": "#3B5B8A",
    "RCBC": "#7ec8e3",
}

# =========================
# HELPERS
# =========================
def parse_quarter_dt(q):
    q = str(q).strip().upper()   # e.g. 1Q23
    quarter = int(q[0])
    year = 2000 + int(q[2:])
    return pd.Period(year=year, quarter=quarter, freq="Q").to_timestamp(how="end")

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
st.title("CCAP — Method C (Regime‑Adjusted)")

K = st.sidebar.slider("Rolling same‑quarter window (K)", 3, 8, 6)

# =========================
# LOAD DATA
# =========================
raw = pd.read_csv(RAW_URL, engine="python")

raw = raw.rename(columns={
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

banks = sorted(
    panel["bank"].unique(),
    key=lambda b: BANK_ORDER_PREF.index(b) if b in BANK_ORDER_PREF else 999
)

banks_pick = st.multiselect("Banks", banks, default=banks)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================
# FIT BASE COEFFICIENTS (UNCHANGED)
# =========================
def fit_uplift(panel_bank):
    g = panel_bank.copy()
    g["qtr"] = g["quarter_dt"].dt.to_period("Q").apply(lambda p: p.quarter)

    g["d_ps"]  = g.groupby("bank")["purchase_sales_bn"].pct_change()
    g["d_cif"] = g.groupby("bank")["cards_in_force_bn"].pct_change()
    g["d_spc"] = g.groupby("bank")["sales_per_cif_000"].pct_change()

    bases = []
    for _, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        base = []
        for _, r in gb.iterrows():
            base.append(np.mean(pools[r["qtr"]]) if pools[r["qtr"]] else np.nan)
            if pd.notna(r["d_ps"]):
                pools[r["qtr"]].append(r["d_ps"])
        bases.append(pd.Series(base, index=gb.index))

    g["g_base"] = pd.concat(bases).sort_index()
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([np.ones(len(fit)), fit["d_cif"], fit["d_spc"]])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])

alpha_raw, beta_cif, beta_spc = fit_uplift(panel)

# ✅ regime re‑centering
alpha_regime = alpha_raw + REGIME_SHIFT_PPT / 100

# =========================
# METHOD C PROJECTION (ADJUSTED)
# =========================
def project_method_C(gb):
    last_per = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last_per.year) * 4 + (TARGET_END.quarter - last_per.quarter)
    if H <= 0:
        return pd.DataFrame()

    hist_ps  = qoq_factors_by_quarter(gb["purchase_sales_bn"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cards_in_force_bn"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["sales_per_cif_000"], gb["quarter_dt"])

    fore_ps  = {q: [] for q in range(1,5)}
    fore_cif = {q: [] for q in range(1,5)}
    fore_spc = {q: [] for q in range(1,5)}

    ps = gb.iloc[-1]["purchase_sales_bn"]
    cif = gb.iloc[-1]["cards_in_force_bn"]
    spc = gb.iloc[-1]["sales_per_cif_000"]

    bank = gb.iloc[0]["bank"]

    # ✅ bank‑specific intercept
    alpha_bank = alpha_regime + (
        LARGE_BANK_EXTRA_PPT / 100 if bank in LARGE_BANKS else 0
    )

    rows = []

    for h in range(1, H + 1):
        t = last_per + h
        q = t.quarter
        prev_ps = ps

        g_base = np.mean((hist_ps[q] + fore_ps[q])[-K:]) - 1 if (hist_ps[q] + fore_ps[q]) else 0
        d_cif = np.mean((hist_cif[q] + fore_cif[q])[-K:]) - 1 if (hist_cif[q] + fore_cif[q]) else 0
        d_spc = np.mean((hist_spc[q] + fore_spc[q])[-K:]) - 1 if (hist_spc[q] + fore_spc[q]) else 0

        uplift = alpha_bank + beta_cif * d_cif + beta_spc * d_spc
        g_ps = g_base + uplift

        ps *= (1 + g_ps)
        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        fore_ps[q].append(ps / prev_ps)
        fore_cif[q].append(1 + d_cif)
        fore_spc[q].append(1 + d_spc)

        rows.append({
            "bank": bank,
            "quarter": str(t),
            "quarter_dt": t.to_timestamp(how="end"),
            "purchase_sales_bn": ps,
            "cards_in_force_bn": cif,
            "sales_per_cif_000": spc,
            "scenario": "Method C",
        })

    return pd.DataFrame(rows)

proj_frames = []
for b in banks_pick:
    gb = panel[panel["bank"] == b]
    if gb.shape[0] < 3:
        continue
    pc = project_method_C(gb)
    if not pc.empty:
        proj_frames.append(pc)

proj = pd.concat(proj_frames, ignore_index=True) if proj_frames else pd.DataFrame()

# =========================
# CHART — PS ONLY (RAW QUARTER LABELS)
# =========================
hist = panel.assign(scenario="Actual")

plot_df = pd.concat([hist, proj], ignore_index=True)

chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "quarter:N",
            sort=alt.SortField("quarter_dt", "ascending"),
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
            "quarter",
            alt.Tooltip("purchase_sales_bn:Q", title="Purchase Sales (Bn)", format=",.2f"),
            alt.Tooltip("cards_in_force_bn:Q", title="Cards in Force (Bn)", format=",.2f"),
            alt.Tooltip("sales_per_cif_000:Q", title="Sales / CIF ('000)", format=",.2f"),
            "scenario"
        ]
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)
