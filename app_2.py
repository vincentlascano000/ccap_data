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
st.title("CCAP — Method C (Per‑Bank Adjustment)")

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
# PER‑BANK ADJUSTMENT LEVERS (±10 PPT EACH)
# =========================================================
st.sidebar.header("Per‑Bank Adjustment (ppt)")

bank_adjustments = {}
for b in banks_pick:
    bank_adjustments[b] = st.sidebar.slider(
        f"{b}",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.05,
        help=f"Growth adjustment applied only to {b}"
    ) / 100  # convert ppt to proportion

# =========================================================
# FIT COEFFICIENTS (METHOD C)
# =========================================================
def fit_uplift(df):
    g = df.copy()
    g["q"] = g["quarter_dt"].dt.to_period("Q").dt.quarter

    g["d_ps"] = g.groupby("bank")["ps"].pct_change()
    g["d_cif"] = g.groupby("bank")["cif"].pct_change()
    g["d_spc"] = g.groupby("bank")["spc"].pct_change()

    bases = []
    for _, gb in g.groupby("bank"):
        seasonal_mean = gb.groupby("q")["d_ps"].mean()
        bases.append(gb["q"].map(seasonal_mean))

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

# =========================================================
# METHOD C PROJECTION (PER‑BANK ADJUSTMENT)
# =========================================================
def project_method_c(gb):
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last.year) * 4 + (TARGET_END.quarter - last.quarter)

    ps = gb.iloc[-1]["ps"]
    cif = gb.iloc[-1]["cif"]
    spc = gb.iloc[-1]["spc"]
    bank = gb.iloc[0]["bank"]

    # ✅ Bank-specific intercept adjustment
    bank_adj = bank_adjustments.get(bank, 0.0)
    alpha_bank = alpha_raw + bank_adj

    rows = []

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

        g_ps = g_base + (alpha_bank + beta_cif * d_cif + beta_spc * d_spc)
        g_ps = np.clip(g_ps, -0.3, 0.3)

        ps *= (1 + g_ps)
        cif *= (1 + d_cif)
        spc *= (1 + d_spc)

        rows.append({
            "quarter_dt": t.to_timestamp(how="end"),
            "quarter_label": fmt_q(t),
            "bank": bank,
            "ps": ps,
            "scenario": "Method C",
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

# =========================================================
# MODEL INTERCEPTS & FORMULA (STAKEHOLDER VIEW)
# =========================================================
adj_rows = "\n".join(
    f"| {b} | `{bank_adjustments[b]*100:+.2f} ppt` |"
    for b in banks_pick
)

st.markdown(f"""
### Growth Formula Used (Method C)

**Quarter‑on‑quarter Purchase Sales growth formula**

$$
\\Delta PS
=
G_{{baseline}}
+
(\\alpha + \\text{{bank adjustment}})
+
\\beta_{{CIF}}\\,\\Delta CIF
+
\\beta_{{SPC}}\\,\\Delta(Sales/CIF)
$$

---

### Estimated Model Parameters (from the data)

| Component | Value |
|---------|-------|
| Intercept (α, raw) | `{alpha_raw:.4f}` |
| β (Cards in Force) | `{beta_cif:.4f}` |
| β (Sales / CIF) | `{beta_spc:.4f}` |

---

### Per‑Bank Adjustments (applied to intercept)

| Bank | Adjustment |
|------|-----------|
{adj_rows}

---

### Interpretation (plain English)

• **Same‑Quarter Baseline** — average historical growth for that quarter  
• **Intercept (α)** — baseline growth not explained by drivers  
• **Bank Adjustment** — manual per‑bank growth tuning (independent)  
• **Drivers (β terms)** — growth from card base and spend intensity
""")
