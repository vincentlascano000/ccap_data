# app.py
# CCAP — Per-bank historical + projected charts (3-year horizon)
# Compare UB vs. other banks. Fixed Altair layering/facet usage (no .spec()).
# Uses CCAP_DATA.csv (uppercase) from GitHub.

import re
from typing import Tuple, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Try statsmodels; fall back to NumPy OLS if not installed.
try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

# -----------------------------
# 0) SOURCE & LABELS
# -----------------------------
RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "balances_bn": "Balances (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
    "sales_per_cif_000": "Sales / CIF ('000)",
}

# Show UB first in the bank facets
BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "quarter": ["quarter", "qtr", "period", "quarter_str", "q", "date"],
    "bank":    ["bank", "issuer", "bank_name", "issuer_name", "issuer bank"],

    "purchase_sales_bn": [
        "purchase_sales_bn", "purchase_sales", "sales",
        "purchase volume (bn)", "purchase sales (in bn)",
        "purchase_sales_bil", "purchase_sales_billion"
    ],
    "balances_bn": [
        "balances_bn", "balances", "balance (bn)",
        "balances (in bn)", "total_balance_bn", "balances_bil", "balances_billion"
    ],
    "cards_in_force_bn": [
        "cards_in_force_bn", "cards_in_force", "cif (bn)", "cif", "cif_in_bn", "cif_bn", "cards"
    ],
    "sales_per_cif_000": [
        "sales_per_cif_000", "sales_per_cif", "sales per cif (000)", "sales/cif (000)", "sales_cif"
    ],
}

# -----------------------------
# 1) HELPERS
# -----------------------------
def _canon(s: str) -> str:
    s = s.strip().lower()
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

    m = re.match(r"^([1-4])Q(\d{2,4})$", s)  # 1Q23
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
    if pd.api.types.is_numeric_dtype(s):  # already numeric
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

def fit_growth_model(fit_df: pd.DataFrame, driver_cols: List[str]):
    """
    Fit Δlog(Purchase Sales) ~ drivers. Use statsmodels if present; else NumPy OLS.
    Returns (coef_dict, summary_text).
    """
    y = "dlog_purchase_sales_bn"
    if HAS_SM:
        rhs = " + ".join(driver_cols) if driver_cols else "1"
        model = smf.ols(f"{y} ~ {rhs}", data=fit_df).fit()
        return model.params.to_dict(), model.summary().as_text()
    else:
        X_cols = [np.ones(len(fit_df))]
        for c in driver_cols:
            X_cols.append(fit_df[c].to_numpy())
        X = np.column_stack(X_cols)
        yv = fit_df[y].to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        coef = {"Intercept": float(beta[0])}
        for i, c in enumerate(driver_cols, start=1):
            coef[c] = float(beta[i])
        return coef, "NumPy least-squares used (statsmodels not installed)."

# -----------------------------
# 2) APP UI
# -----------------------------
st.set_page_config(page_title="CCAP — Per-bank Projections", layout="wide")
st.title("CCAP — Per-bank Historical & Projected Charts (3-year horizon)")
st.caption(f"Source: {RAW_URL}")

# Scenario & modeling knobs
st.sidebar.header("Scenario settings")
window_q  = st.sidebar.slider("Window for 'Realistic' mean (quarters)", 3, 8, 4, 1)
k_vol     = st.sidebar.slider("Optimistic/Pessimistic spread (× volatility)", 0.2, 1.0, 0.6, 0.1)
horizon_q = st.sidebar.slider("Projection horizon (quarters)", 8, 16, 12, 1)

st.sidebar.header("Model settings")
coef_source = st.sidebar.radio("Coefficient source", ["Pooled (recommended)", "Per-bank"], index=0)
use_cif     = st.sidebar.checkbox("Include CIF as a driver (optional)", value=False)

# -----------------------------
# 3) LOAD & NORMALIZE
# -----------------------------
try:
    raw = load_raw_csv(RAW_URL)
except Exception as e:
    st.error(f"Failed to read CSV from GitHub: {e}")
    st.stop()

df = map_columns(raw)
df = drop_duplicate_names_keep_first(df)

# Identify quarter & bank
qcol = next((c for c in df.columns if _canon(c) in {_canon(x) for x in COLUMN_SYNONYMS["quarter"]}), None)
bcol = next((c for c in df.columns if _canon(c) in {_canon(x) for x in COLUMN_SYNONYMS["bank"]}), None)
if qcol is None or bcol is None:
    st.error("Could not find 'quarter' and/or 'bank' columns. Check headers.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Build quarter_dt/quarter_str
qs, qd = [], []
for v in df[qcol].astype(str):
    qlabel, qdate = parse_quarter_token(v)
    qs.append(qlabel); qd.append(qdate)
df["quarter_str"] = qs
df["quarter_dt"]  = qd

# Numeric metrics (present only)
possible_metrics = ["purchase_sales_bn","balances_bn","cards_in_force_bn","sales_per_cif_000"]
present_metrics  = [m for m in possible_metrics if m in df.columns]
for key in present_metrics:
    df[key] = to_numeric(df[key])

panel = df[["quarter_dt", "quarter_str", bcol] + present_metrics].copy()
panel = panel.rename(columns={bcol: "bank"})
panel = panel.dropna(subset=["quarter_dt", "bank"]).sort_values(["bank","quarter_dt"]).reset_index(drop=True)

if "purchase_sales_bn" not in panel.columns:
    st.error("Missing Purchase Sales column after normalization. Cannot build projections.")
    st.stop()

# -----------------------------
# 4) Δlog per bank & driver set
# -----------------------------
bp = panel.sort_values(["bank","quarter_dt"]).copy()
perf_cols = [c for c in ["purchase_sales_bn","balances_bn","sales_per_cif_000","cards_in_force_bn"] if c in bp.columns]
for col in perf_cols:
    bp[f"log__{col}"]  = np.log(bp[col].replace({0: np.nan}))
    bp[f"dlog__{col}"] = bp.groupby("bank")[f"log__{col}"].diff()

driver_map = {
    "dlog_balances_bn":       "balances_bn",
    "dlog_sales_per_cif_000": "sales_per_cif_000",
    "dlog_cards_in_force_bn": "cards_in_force_bn",
}
driver_cols_base = [d for d, base in driver_map.items() if base in perf_cols and f"dlog__{base}" in bp.columns]

driver_cols = [d for d in driver_cols_base if d in ["dlog_balances_bn","dlog_sales_per_cif_000"]]
if use_cif and "dlog_cards_in_force_bn" in driver_cols_base:
    driver_cols.append("dlog_cards_in_force_bn")

if not any(d in driver_cols for d in ["dlog_balances_bn", "dlog_sales_per_cif_000"]):
    st.error("Need at least one driver among Balances or Sales/CIF to model Purchase Sales.")
    st.write("Present columns:", list(panel.columns))
    st.stop()

# Pooled (stacked per-bank) fit_df
fit_cols = ["bank","quarter_dt","dlog__purchase_sales_bn"] + [f"dlog__{driver_map[d]}" for d in driver_cols]
pooled_df = bp[fit_cols].dropna().copy()
pooled_df = pooled_df.rename(columns={"dlog__purchase_sales_bn":"dlog_purchase_sales_bn",
                                      **{f"dlog__{driver_map[d]}": d for d in driver_cols}})
if pooled_df.shape[0] < 6:
    st.error("Not enough data points to fit the pooled growth model.")
    st.stop()

pooled_coef, pooled_summary = fit_growth_model(pooled_df, driver_cols)

with st.expander("Pooled growth model (Δlog Sales ~ drivers)"):
    st.text(pooled_summary)

# Build bank-specific coefficients
bank_coefs: Dict[str, Dict[str, float]] = {}
if coef_source.startswith("Pooled"):
    for b in bp["bank"].dropna().unique():
        bank_coefs[b] = pooled_coef.copy()
else:
    for b, gbank in bp.groupby("bank"):
        cols = ["dlog__purchase_sales_bn"] + [f"dlog__{driver_map[d]}" for d in driver_cols]
        gfit = gbank[cols + ["quarter_dt"]].dropna().copy()
        gfit = gfit.rename(columns={"dlog__purchase_sales_bn":"dlog_purchase_sales_bn",
                                    **{f"dlog__{driver_map[d]}": d for d in driver_cols}})
        if gfit.shape[0] < 6:
            bank_coefs[b] = pooled_coef.copy()
        else:
            bcoef, _ = fit_growth_model(gfit, driver_cols)
            bank_coefs[b] = bcoef

# Pooled μ,σ for fallback
pooled_mu = {d: float(pooled_df[d].mean()) for d in driver_cols}
pooled_sd = {d: float(pooled_df[d].std(ddof=1)) for d in driver_cols}

def bank_mu_sd(gg: pd.DataFrame, d: str) -> tuple:
    """(mu, sd) for Δlog driver 'd' for this bank over last N quarters; fallback to pooled if sparse."""
    dcol = f"dlog__{driver_map[d]}"
    recent = gg[gg["quarter_dt"] > (gg["quarter_dt"].max() - pd.offsets.QuarterEnd(window_q))][dcol].dropna()
    if recent.shape[0] < 3:
        return pooled_mu.get(d, 0.0), pooled_sd.get(d, 0.0)
    return float(recent.mean()), float(recent.std(ddof=1))

# -----------------------------
# 5) Per-bank projections
# -----------------------------
banks_all = sorted(bp["bank"].dropna().unique().tolist(), key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
default_banks = [b for b in BANK_ORDER_PREF if b in banks_all] or banks_all
banks_pick = st.multiselect("Banks to display", options=banks_all, default=default_banks)

metric_options = [FRIENDLY[m] for m in perf_cols]
default_metrics = [FRIENDLY["purchase_sales_bn"]] + [m for m in metric_options if m != FRIENDLY["purchase_sales_bn"]]
metric_pick = st.multiselect("Metrics to display", options=metric_options, default=default_metrics)

bank_horizon = horizon_q

def project_one_bank(bname: str) -> pd.DataFrame:
    gbank = bp[bp["bank"] == bname].copy()
    if gbank.empty:
        return pd.DataFrame()

    # last levels for this bank
    last_row = gbank.dropna(subset=["quarter_dt"]).sort_values("quarter_dt").iloc[-1]
    last_levels = {m: last_row[m] for m in perf_cols}

    mu_sd_bank = {d: bank_mu_sd(gbank, d) for d in driver_cols}

    def scen_val(d, sign):
        mu, sd = mu_sd_bank.get(d, (0.0,0.0))
        if sign > 0:  return mu + k_vol*sd
        if sign < 0:  return mu - k_vol*sd
        return mu

    scenario_growths = {
        "Realistic":  {d: scen_val(d,  0) for d in driver_cols},
        "Optimistic": {d: scen_val(d, +1) for d in driver_cols},
        "Pessimistic":{d: scen_val(d, -1) for d in driver_cols},
    }

    future_dates = [gbank["quarter_dt"].max() + pd.offsets.QuarterEnd(i) for i in range(1, bank_horizon+1)]

    rows = []
    bcoef = bank_coefs.get(bname, pooled_coef)
    for sc, params in scenario_growths.items():
        lvl = last_levels.copy()
        for dt in future_dates:
            # evolve drivers
            for d, base in driver_map.items():
                if base in lvl and d in params:
                    lvl[base] *= np.exp(params[d])

            # predict Δlog(sales)
            dlog_sales = bcoef.get("Intercept", 0.0)
            for d in driver_cols:
                dlog_sales += bcoef.get(d, 0.0) * params[d]
            if "purchase_sales_bn" in lvl:
                lvl["purchase_sales_bn"] *= np.exp(dlog_sales)

            rows.append({"bank": bname, "scenario": sc, "quarter_dt": dt, **lvl})
    return pd.DataFrame(rows)

plot_frames = []
for b in banks_pick:
    hist_b = (bp[bp["bank"] == b][["quarter_dt"] + perf_cols]
                .dropna(subset=["quarter_dt"])
                .assign(bank=b, scenario="Actual"))
    proj_b = project_one_bank(b)
    if not proj_b.empty:
        proj_b = proj_b.assign(quarter=lambda d: d["quarter_dt"].dt.to_period("Q").astype(str))

    for m in perf_cols:
        if FRIENDLY[m] not in metric_pick:
            continue
        if m in hist_b.columns:
            plot_frames.append(hist_b.assign(metric=FRIENDLY[m], value=hist_b[m])[["bank","scenario","quarter_dt","metric","value"]])
        if not proj_b.empty and m in proj_b.columns:
            plot_frames.append(proj_b.assign(metric=FRIENDLY[m], value=proj_b[m])[["bank","scenario","quarter_dt","metric","value"]])

if not plot_frames:
    st.info("Select at least one bank and one metric to display.")
    st.stop()

pb_plot = pd.concat(plot_frames, ignore_index=True)
sort_banks = [b for b in BANK_ORDER_PREF if b in banks_pick] + [b for b in sorted(set(banks_pick)-set(BANK_ORDER_PREF))]


# -----------------------------
# 6) CHARTS — Fixed layering + facet (set height before facet)
# -----------------------------
color_scale = alt.Scale(
    domain=["Actual","Realistic","Optimistic","Pessimistic"],
    range=["#000000","#1f78b4","#33a02c","#e31a1c"]
)

for metric_nm in sorted(pb_plot["metric"].unique()):
    sub = pb_plot[pb_plot["metric"] == metric_nm].copy()
    st.subheader(f"{metric_nm} — Per bank")

    # Build the two layers and set height on these layers (NOT after facet)
    line_actual = (
        alt.Chart(sub)
          .transform_filter(alt.datum.scenario == "Actual")
          .mark_line(point=True, strokeWidth=2, color="#000000")
          .encode(
              x=alt.X("quarter_dt:T", title="Quarter", axis=alt.Axis(format="%Y Q%q")),
              y=alt.Y("value:Q", title=metric_nm)
          )
          .properties(height=300)  # ✅ set height here
    )

    line_scen = (
        alt.Chart(sub)
          .transform_filter(alt.datum.scenario != "Actual")
          .mark_line(point=False, strokeDash=[6,4], strokeWidth=2)
          .encode(
              x="quarter_dt:T",
              y="value:Q",
              color=alt.Color("scenario:N", title="Scenario", scale=color_scale)
          )
          .properties(height=300)  # ✅ and/or here (keeps both layers consistent)
    )

    # Layer first, THEN facet. Do not call .properties(height=...) on the FacetChart.
    layered = alt.layer(line_actual, line_scen)
    faceted = (
        layered
          .facet(column=alt.Column("bank:N", sort=sort_banks, title="Bank"))
          .resolve_scale(y="independent")
    )

    st.altair_chart(faceted, use_container_width=True)

# -----------------------------
# 7) NOTES
# -----------------------------
with st.expander("Methodology (brief for stakeholders)"):
    st.markdown(
        """
**Equation (per quarter, per bank):**  
\\[
\\Delta \\log(\\text{Purchase Sales})\\_t
= \\alpha
+ \\beta_1\\,\\Delta \\log(\\text{Balances})\\_t
+ \\beta_2\\,\\Delta \\log(\\text{Sales/CIF})\\_t
\\;[+\\;\\beta_3\\,\\Delta \\log(\\text{CIF})\\_t]\\; + \\varepsilon\\_t
\\]

- We model **growth** using quarterly **log differences (Δlog)**, which avoids trend bias and keeps variables on comparable (%‑like) scales.  
- Drivers reflect earlier **correlation analysis**: **Balances** & **Sales/CIF** strongly co‑move with **Sales**; **CIF** is optional.  
- **Pooled coefficients** (default): estimated on the **stacked per‑bank panel** for stability.  
  **Per‑bank** coefficients: mini‑model per bank (more tailored, can be noisier).  
- **Scenarios:** Realistic = mean of last *N* Δlog per driver (bank‑specific, pooled fallback if sparse);  
  Optimistic/Pessimistic = Realistic ± *k ×* std.
        """
    )
