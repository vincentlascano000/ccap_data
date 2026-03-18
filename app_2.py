# app.py
# CCAP — Per-bank projections using straight average period-to-period % change (no logs).
# Factors in Balances & Cards (CIF) (and optionally Sales/CIF) via simple elasticities (cov/var on % changes).
# Renders three separate consolidated overlay charts (Realistic, Optimistic, Pessimistic).
# Includes a projections table (rounded) for the next N quarters.

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
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "balances_bn": "Balances (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
    "sales_per_cif_000": "Sales / CIF ('000)",
}

# Bank order preference (so UB appears first where relevant)
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

# =========================================
# 1) HELPERS (canon, mapping, quarter, numeric, load)
# =========================================
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

# =========================================
# 2) APP UI
# =========================================
st.set_page_config(page_title="CCAP — Per-bank Projections (Simple % Change)", layout="wide")
st.title("CCAP — Per-bank Historical & Projected (Simple % Change + Drivers)")

# Scenario & projection knobs
st.sidebar.header("Projection settings")
window_q  = st.sidebar.slider("Window for % change stats (quarters)", 3, 12, 8, 1)  # used for means/std & elasticities
k_vol     = st.sidebar.slider("Optimistic/Pessimistic spread (× volatility)", 0.0, 1.0, 0.5, 0.1)
horizon_q = st.sidebar.slider("Projection horizon (quarters)", 8, 16, 12, 1)
table_h_q = st.sidebar.slider("Estimates table: next N projected quarters", 4, 12, 8, 1)

# Optional additional driver
use_spc = st.sidebar.checkbox("Include Sales/CIF as additional driver", value=False)

# =========================================
# 3) LOAD & NORMALIZE
# =========================================
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

# Bank selection
banks_all = sorted(panel["bank"].dropna().unique().tolist(), key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
default_banks = [b for b in BANK_ORDER_PREF if b in banks_all] or banks_all
banks_pick = st.multiselect("Banks to include", options=banks_all, default=default_banks)

# Metric selection (single consolidated chart per scenario)
metric_options = [FRIENDLY[m] for m in present_metrics]
default_metric = FRIENDLY["purchase_sales_bn"] if "purchase_sales_bn" in present_metrics else metric_options[0]
metric_sel = st.selectbox("Metric to chart (overlay for all selected banks)", options=metric_options, index=metric_options.index(default_metric))
metric_code = {v:k for k,v in FRIENDLY.items()}[metric_sel]

# =========================================
# 4) % CHANGE per bank + window stats
# =========================================
bp = panel.sort_values(["bank","quarter_dt"]).copy()
perf_cols = [c for c in ["purchase_sales_bn","balances_bn","cards_in_force_bn","sales_per_cif_000"] if c in bp.columns]

# Build % change series (simple period-to-period)
for col in perf_cols:
    bp[f"pct__{col}"] = bp.groupby("bank")[col].pct_change()

# Driver set (Balances & Cards required; Sales/CIF optional)
drivers = [m for m in ["balances_bn", "cards_in_force_bn"] if m in perf_cols]
if use_spc and "sales_per_cif_000" in perf_cols:
    drivers.append("sales_per_cif_000")

# =========================================
# 5) Simple driver-based projection per bank (no logs, % change only)
#    - Project drivers by mean ± k*std of THEIR % changes over the window.
#    - Compute Sales % change: mean_sales + sum(elasticity * driver_pct_change).
#    - Elasticities e computed as cov/var slopes on % changes over the window.
# =========================================
def elastic_slope(y: pd.Series, x: pd.Series) -> float:
    """Simple slope = cov(y,x) / var(x) using % changes; robust to zero variance."""
    df = pd.concat([y, x], axis=1).dropna()
    if df.shape[0] < 3:
        return 0.0
    vx = np.var(df.iloc[:,1], ddof=1)
    if vx <= 0:
        return 0.0
    return float(np.cov(df.iloc[:,0], df.iloc[:,1], ddof=1)[0,1] / vx)

def last_n_window_mask(gbank: pd.DataFrame, n_quarters: int) -> pd.Series:
    """Boolean mask for last N quarters of this bank's history."""
    if gbank.empty:
        return gbank.assign(_=False)["_"]
    last_dt = gbank["quarter_dt"].max()
    return gbank["quarter_dt"] > (last_dt - pd.offsets.QuarterEnd(n_quarters))

def project_one_bank_pct(bname: str) -> pd.DataFrame:
    gbank = bp[bp["bank"] == bname].copy()
    if gbank.empty:
        return pd.DataFrame()

    # Last levels for this bank (for all present perf_cols)
    last_row = gbank.dropna(subset=["quarter_dt"]).sort_values("quarter_dt").iloc[-1]
    last_levels = {m: float(last_row[m]) for m in perf_cols}

    # Windowed stats
    msk = last_n_window_mask(gbank, window_q)
    gwin = gbank[msk].copy()

    # Mean and std of % changes
    mu = {}; sd = {}
    for m in perf_cols:
        pser = gwin[f"pct__{m}"].dropna()
        if pser.shape[0] >= 3:
            mu[m] = float(pser.mean())
            sd[m] = float(pser.std(ddof=1))
        else:
            mu[m] = 0.0
            sd[m] = 0.0

    # Elasticities of Sales to each driver (on % changes)
    e = {}
    y = gwin["pct__purchase_sales_bn"].dropna()
    for d in drivers:
        if f"pct__{d}" in gwin.columns:
            x = gwin[f"pct__{d}"].dropna()
            e[d] = elastic_slope(y, x)
        else:
            e[d] = 0.0

    def scen_pct(m: str, sign: int) -> float:
        # sign: 0=Realistic, +1=Optimistic, -1=Pessimistic
        return mu.get(m, 0.0) + (k_vol * sd.get(m, 0.0) * sign)

    future_dates = [gbank["quarter_dt"].max() + pd.offsets.QuarterEnd(i) for i in range(1, horizon_q+1)]
    rows = []
    for sc_name, sign in [("Realistic", 0), ("Optimistic", +1), ("Pessimistic", -1)]:
        levels = last_levels.copy()
        for dt in future_dates:
            # 1) Evolve drivers by their scenario % change
            pct_driver = {}
            for d in drivers:
                if d in levels:
                    gd = scen_pct(d, sign)      # driver pct change
                    levels[d] *= (1.0 + gd)
                    pct_driver[d] = gd

            # 2) Sales % change = own mean + sum(elasticity * driver_pct_change)
            gS = scen_pct("purchase_sales_bn", sign)
            adj = sum(e[d] * pct_driver.get(d, 0.0) for d in drivers)
            pct_sales = gS + adj
            if "purchase_sales_bn" in levels:
                levels["purchase_sales_bn"] *= (1.0 + pct_sales)

            # 3) Non-driver metrics present but not in drivers → evolve by their own % change mean
            for m in perf_cols:
                if m not in drivers and m != "purchase_sales_bn":
                    levels[m] *= (1.0 + scen_pct(m, sign))

            rows.append({"bank": bname, "scenario": sc_name, "quarter_dt": dt, **levels})
    return pd.DataFrame(rows)

# =========================================
# 6) Build overlay datasets (Actual + projections) for chosen metric
# =========================================
plot_frames_real = []
plot_frames_opt  = []
plot_frames_pes  = []
table_frames     = []

for b in banks_pick:
    # Actual per bank
    hist_b = (bp[bp["bank"] == b][["quarter_dt"] + perf_cols]
                .dropna(subset=["quarter_dt"])
                .assign(bank=b, scenario="Actual"))

    # Projections per bank
    proj_b = project_one_bank_pct(b)
    if not proj_b.empty:
        proj_b = proj_b.assign(quarter=lambda d: d["quarter_dt"].dt.to_period("Q").astype(str))

    # Select metric only
    m = metric_code
    if m in hist_b.columns:
        plot_frames_real.append(
            hist_b.assign(value=hist_b[m])[["bank","scenario","quarter_dt","value"]]
        )
    if not proj_b.empty and m in proj_b.columns:
        plot_frames_real.append(
            proj_b[proj_b["scenario"]=="Realistic"].assign(value=proj_b[proj_b["scenario"]=="Realistic"][m])[
                ["bank","scenario","quarter_dt","value"]
            ]
        )
        plot_frames_opt.append(
            proj_b[proj_b["scenario"]=="Optimistic"].assign(value=proj_b[proj_b["scenario"]=="Optimistic"][m])[
                ["bank","scenario","quarter_dt","value"]
            ]
        )
        plot_frames_pes.append(
            proj_b[proj_b["scenario"]=="Pessimistic"].assign(value=proj_b[proj_b["scenario"]=="Pessimistic"][m])[
                ["bank","scenario","quarter_dt","value"]
            ]
        )
        # Table for all scenarios (rounded)
        table_frames.append(
            proj_b.assign(value=np.round(proj_b[m], 1))[["quarter_dt","quarter","bank","scenario","value"]]
        )

def build_chart(overlay_df: pd.DataFrame, title: str):
    if overlay_df.empty:
        st.info(f"No data to plot for {title}.")
        return
    color_scale = alt.Scale(scheme='tableau10')  # robust default palette
    # Actual (solid)
    line_actual = (
        alt.Chart(overlay_df)
          .transform_filter(alt.datum.scenario == "Actual")
          .mark_line(point=True, strokeWidth=2)
          .encode(
              x=alt.X("quarter_dt:T", title="Quarter", axis=alt.Axis(format="%Y Q%q")),
              y=alt.Y("value:Q", title=FRIENDLY[metric_code]),
              color=alt.Color("bank:N", title="Bank", scale=color_scale),
              tooltip=[
                  alt.Tooltip("bank:N", title="Bank"),
                  alt.Tooltip("quarter_dt:T", title="Quarter", format="%Y Q%q"),
                  alt.Tooltip("value:Q", title=FRIENDLY[metric_code], format=",.2f"),
                  alt.Tooltip("scenario:N", title="Scenario")
              ]
          )
          .properties(height=360)
    )
    # Projections (dashed)
    line_proj = (
        alt.Chart(overlay_df)
          .transform_filter(alt.datum.scenario != "Actual")
          .mark_line(point=False, strokeWidth=2, strokeDash=[6,4])
          .encode(
              x="quarter_dt:T",
              y="value:Q",
              color=alt.Color("bank:N", title="Bank", scale=color_scale),
              tooltip=[
                  alt.Tooltip("bank:N", title="Bank"),
                  alt.Tooltip("quarter_dt:T", title="Quarter", format="%Y Q%q"),
                  alt.Tooltip("value:Q", title=FRIENDLY[metric_code], format=",.2f"),
                  alt.Tooltip("scenario:N", title="Scenario")
              ]
          )
          .properties(height=360)
    )
    st.subheader(title)
    st.altair_chart(alt.layer(line_actual, line_proj), use_container_width=True)

# Concatenate frames per scenario
overlay_real = pd.concat(plot_frames_real, ignore_index=True) if plot_frames_real else pd.DataFrame()
overlay_opt  = pd.concat(plot_frames_opt,  ignore_index=True) if plot_frames_opt  else pd.DataFrame()
overlay_pes  = pd.concat(plot_frames_pes,  ignore_index=True) if plot_frames_pes  else pd.DataFrame()

# =========================================
# 7) THREE SEPARATE CONSOLIDATED CHARTS
# =========================================
build_chart(overlay_real, f"{FRIENDLY[metric_code]} — Realistic")
build_chart(overlay_opt,  f"{FRIENDLY[metric_code]} — Optimistic")
build_chart(overlay_pes,  f"{FRIENDLY[metric_code]} — Pessimistic")

# =========================================
# 8) NUMBERS TABLE (rounded projected volumes)
# =========================================
if table_frames:
    proj_table = pd.concat(table_frames, ignore_index=True)
    proj_table = proj_table.sort_values(["quarter_dt","bank","scenario"])
    # next N quarters
    upcoming_quarters = sorted(proj_table["quarter_dt"].unique())[:table_h_q]
    tbl = (proj_table[proj_table["quarter_dt"].isin(upcoming_quarters)]
           .rename(columns={"value": f"{FRIENDLY[metric_code]} (~)"})
           .loc[:, ["quarter","bank","scenario", f"{FRIENDLY[metric_code]} (~)"]])
    st.subheader("Projected volumes (rough estimates)")
    st.dataframe(tbl, use_container_width=True)
else:
    st.info("No projections available to tabulate (check drivers/selection).")
