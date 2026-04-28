# app_2.py
# CCAP — Method C ONLY
# Option A: Permanent +6 ppt baseline, ONE-TIME +5.5% level spike for BDO/BPI at 2026Q1
# FIXED quarter parsing and growth logic

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

BASELINE_SHIFT_PPT = 6.0
ONE_TIME_LIFT = 1.055   # +5.5% level jump
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
# QUARTER HELPERS (FIXED)
# =========================================================
def parse_quarter_dt(q):
    """
    Accepts strings like '1Q23', '4Q25'
    Returns quarter-end timestamp
    """
    s = str(q).strip().upper()
    quarter = int(s[0])
    year = 2000 + int(s[2:])
    return (
        pd.Period(year=year, quarter=quarter, freq="Q")
        .to_timestamp(how="end")
    )

def fmt_q(p):
    return f"{p.quarter}Q{str(p.year)[-2:]}"

# =========================================================
# QoQ FACTORS (GUARDED)
# =========================================================
def qoq_factors_by_quarter(series, qdt):
    per = qdt.dt.to_period("Q")
    s = pd.Series(series.values, index=per).sort_index()
    g = (s / s.shift(1)).dropna()

    out = {1: [], 2: [], 3: [], 4: []}
    for p, v in g.items():
        if np.isfinite(v) and 0 < v < 5:
            out[p.quarter].append(float(v))
    return out

# =========================================================
# UI
# =========================================================
st.title("CCAP — Method C (Option A)")
K = st.sidebar.slider("Rolling same‑quarter window (K)", 3, 8, 6)

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

panel = raw.dropna().sort_values(["bank", "quarter_dt"]).reset_index(drop=True)

banks = panel["bank"].unique().tolist()
banks_pick = st.multiselect("Banks", banks, default=banks)
panel = panel[panel["bank"].isin(banks_pick)]

# =========================================================
# FIT COEFFICIENTS
# =========================================================
def fit_uplift(df):
    g = df.copy()
    g["q"] = g["quarter_dt"].dt.to_period("Q").dt.quarter

    g["d_ps"] = g.groupby("bank")["ps"].pct_change()
    g["d_cif"] = g.groupby("bank")["cif"].pct_change()
    g["d_spc"] = g.groupby("bank")["spc"].pct_change()

    bases = []
    for _, gb in g.groupby("bank"):
        pools = {1: [], 2: [], 3: [], 4: []}
        base = []
        for _, r in gb.iterrows():
            base.append(np.mean(pools[r["q"]]) if pools[r["q"]] else np.nan)
            if pd.notna(r["d_ps"]) and abs(r["d_ps"]) < 1:
                pools[r["q"]].append(r["d_ps"])
        bases.append(pd.Series(base, index=gb.index))

    g["g_base"] = pd.concat(bases).sort_index()
    g["r_ps"] = g["d_ps"] - g["g_base"]

    fit = g.dropna(subset=["r_ps", "d_cif", "d_spc"])
    X = np.column_stack([np.ones(len(fit)), fit["d_cif"], fit["d_spc"]])
    y = fit["r_ps"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

alpha_raw, beta_cif, beta_spc = fit_uplift(panel)
alpha = alpha_raw + BASELINE_SHIFT_PPT / 100

# =========================================================
# METHOD C PROJECTION — OPTION A (SAFE)
# =========================================================
def project_method_c(gb):
    last = gb["quarter_dt"].max().to_period("Q")
    H = (TARGET_END.year - last.year) * 4 + (TARGET_END.quarter - last.quarter)

    hist_ps = qoq_factors_by_quarter(gb["ps"], gb["quarter_dt"])
    hist_cif = qoq_factors_by_quarter(gb["cif"], gb["quarter_dt"])
    hist_spc = qoq_factors_by_quarter(gb["spc"], gb["quarter_dt"])

    fore_ps = {q: [] for q in range(1, 5)}
    fore_cif = {q: [] for q in range(1, 5)}
    fore_spc = {q: [] for q in range(1, 5)}

    ps = gb.iloc[-1]["ps"]
    cif = gb.iloc[-1]["cif"]
    spc = gb.iloc[-1]["spc"]
    bank = gb.iloc[0]["bank"]

    lifted = False
    rows = []

    for h in range(1, H + 1):
        t = last + h
        q = t.quarter

        g_base = np.mean((hist_ps[q] + fore_ps[q])[-K:]) - 1 if hist_ps[q] + fore_ps[q] else 0
        d_cif = np.mean((hist_cif[q] + fore_cif[q])[-K:]) - 1 if hist_cif[q] + fore_cif[q] else 0
        d_spc = np.mean((hist_spc[q] + fore_spc[q])[-K:]) - 1 if hist_spc[q] + fore_spc[q] else 0

        g_ps = g_base + (alpha + beta_cif * d_cif + beta_spc * d_spc)
        g_ps = np.clip(g_ps, -0.3, 0.3)  # hard safety clamp

        prev_ps = ps
        ps *= (1 + g_ps)

        # ✅ ONE-TIME spike at 2026 Q1
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

        fore_ps[q].append(ps / prev_ps)
        fore_cif[q].append(1 + d_cif)
        fore_spc[q].append(1 + d_spc)

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
    quarter_label=panel["quarter"], scenario="Actual"
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
st.write("DEBUG: chart rendered above this line")
# =========================================================
# 📌 MODEL INTERCEPTS & FORMULA (STAKEHOLDER VIEW)
# =========================================================
st.markdown(f"""
### Growth Formula Used (Method C)

**Quarter‑on‑quarter Purchase Sales growth formula**

$$
\\Delta PS
=
G_{{baseline}}
+
(\\alpha + {BASELINE_SHIFT_PPT:.1f})
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
| Macro baseline uplift | `+{BASELINE_SHIFT_PPT:.1f} ppt` |
| **Effective intercept** | **`{alpha:.4f}`** |
| β (Cards in Force) | `{beta_cif:.4f}` |
| β (Sales / CIF) | `{beta_spc:.4f}` |

---

### Interpretation (plain English)

• **Same‑Quarter Baseline**  
Average historical growth for the same calendar quarter

• **Intercept (α + 6 ppt)**  
Represents current economy‑wide growth conditions, applied to all banks

• **Drivers (β terms)**  
Explain growth from card base expansion and spend intensity

• **One‑time BDO/BPI adjustment**  
A single +5.5% level uplift in **2026 Q1 only**, not ongoing growth
""")
