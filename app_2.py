# app_2.py
# CCAP — Method C ONLY
# Option A:
# +6 ppt structural baseline
# Scenario lever applies ±6 ppt economy‑wide
# ONE‑TIME +5.5% level spike for BDO/BPI at 2026Q1

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="CCAP — Method C", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")

BASELINE_SHIFT_PPT = 6.0          # permanent structural uplift
ONE_TIME_LIFT = 1.055             # +5.5% level jump
LARGE_BANKS = {"BDO", "BPI"}

BANK_COLORS = {
    "UB": "#f28e2b",
    "BDO": "#4169E1",
    "BPI": "#d62728",
    "SECBANK": "#4CAF50",
    "MB": "#3B5B8A",
    "RCBC": "#7ec8e3",
}

# =========================================================
# QUARTER HELPERS
# =========================================================
def parse_quarter_dt(q):
    s = str(q).strip().upper()
    quarter = int(s[0])
    year = 2000 + int(s[2:])
    return pd.Period(
        year=year,
        quarter=quarter,
        freq="Q"
    ).to_timestamp(how="end")

def fmt_q(p):
    return f"{p.quarter}Q{str(p.year)[-2:]}"

# =========================================================
# UI
# =========================================================
st.title("CCAP — Method C (Option A)")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Pessimistic", "Realistic", "Optimistic"],
    index=1  # Realistic default
)

SCENARIO_PPT = {
    "Pessimistic": -6.0,
    "Realistic": 0.0,
    "Optimistic": 6.0,
}
scenario_shift = SCENARIO_PPT[scenario] / 100

# =========================================================
# LOAD DATA
# =========================================================
raw = pd.read_csv(RAW_URL).rename(columns={
    "QUARTER": "quarter",
    "BANK": "bank",
    "Purchase Sales (in Bn)": "ps",
    "Cards in Force (in Bn)": "cif",
    "Sales / CIF ('000)": "spc",
})

raw = raw[["quarter", "bank", "ps", "cif", "spc"]]
raw["quarter_dt"] = raw["quarter"].apply(parse_quarter_dt)

for c in ["ps", "cif", "spc"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

panel = (
    raw.dropna()
       .sort_values(["bank", "quarter_dt"])
       .reset_index(drop=True)
)

banks = panel["bank"].unique().tolist()
banks_pick = st.multiselect("Banks", banks, default=banks)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================================================
# FIT COEFFICIENTS (NO ROLLING WINDOW)
# =========================================================
def fit_uplift(df):
    g = df.copy()
    g["q"] = g["quarter_dt"].dt.to_period("Q").dt.quarter

    g["d_ps"] = g.groupby("bank")["ps"].pct_change()
    g["d_cif"] = g.groupby("bank")["cif"].pct_change()
    g["d_spc"] = g.groupby("bank")["spc"].pct_change()

    # Same‑quarter seasonal baseline (full history)
    bases = []
    for _, gb in g.groupby("bank"):
        pools = gb.groupby("q")["d_ps"].mean()
        bases.append(gb["q"].map(pools))

    g["g_base"] = pd.concat(bases).sort_index()
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([
        np.ones(len(fit)),
        fit["d_cif"],
        fit["d_spc"]
    ])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

alpha_raw, beta_cif, beta_spc = fit_uplift(panel)

# ✅ Effective intercept:
alpha = (
    alpha_raw
    + BASELINE_SHIFT_PPT / 100
    + scenario_shift
)

# =========================================================
# METHOD C PROJECTION — OPTION A
# =========================================================
def project_method_c(gb):
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last.year) * 4 + (TARGET_END.quarter - last.quarter)

    ps = gb.iloc[-1]["ps"]
    cif = gb.iloc[-1]["cif"]
    spc = gb.iloc[-1]["spc"]
    bank = gb.iloc[0]["bank"]

    lifted = False
    rows = []

    # Fixed seasonal baseline by quarter
    seasonal = gb.assign(
        q=gb["quarter_dt"].dt.quarter,
        d_ps=gb["ps"].pct_change(),
        d_cif=gb["cif"].pct_change(),
        d_spc=gb["spc"].pct_change(),
    ).groupby("q")[["d_ps", "d_cif", "d_spc"]].mean()

    for h in range(1, H + 1):
        t = last + h
        q = t.quarter

        g_base = seasonal.loc[q, "d_ps"] if q in seasonal.index else 0
        d_cif = seasonal.loc[q, "d_cif"] if q in seasonal.index else 0
        d_spc = seasonal.loc[q, "d_spc"] if q in seasonal.index else 0

        g_ps = g_base + (alpha + beta_cif * d_cif + beta_spc * d_spc)
        g_ps = np.clip(g_ps, -0.3, 0.3)

        ps *= (1 + g_ps)

        # ✅ ONE‑TIME BDO/BPI spike
        if (
            not lifted
            and bank in LARGE_BANKS
            and t.year == 2026
            and t.quarter == 1
        ):
            ps *= ONE_TIME_LIFT
            lifted = True

        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        rows.append({
            "quarter_dt": t.to_timestamp(how="end"),
            "quarter_label": fmt_q(t),
            "bank": bank,
            "ps": ps,
            "scenario": scenario,
        })

    return pd.DataFrame(rows)

proj = pd.concat(
    [
        project_method_c(panel[panel["bank"] == b])
        for b in banks_pick
        if panel[panel["bank"] == b].shape[0] >= 3
    ],
    ignore_index=True,
)

# =========================================================
# DISPLAY
# =========================================================
hist = panel.assign(
    quarter_label=panel["quarter"],
    scenario="Actual"
)[["quarter_label", "bank", "ps", "scenario"]]

plot_df = pd.concat([hist, proj], ignore_index=True)

chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("quarter_label:N", sort=alt.SortField("quarter_dt")),
        y=alt.Y("ps:Q", title="Purchase Sales (Bn)"),
        color=alt.Color(
            "bank:N",
            scale=alt.Scale(
                domain=list(BANK_COLORS.keys()),
                range=list(BANK_COLORS.values()),
            ),
        ),
        strokeDash=alt.condition(
            alt.datum.scenario == "Actual",
            alt.value([0]),
            alt.value([6, 4]),
        ),
    )
)

st.altair_chart(chart, use_container_width=True)
