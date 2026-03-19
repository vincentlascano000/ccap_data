# app.py
# CCAP — CIF × Sales/CIF projections with "latest same-quarter growth" carry-forward
# - Purchase Sales (Bn) = scale × CIF × Sales/CIF
# - For each quarter-of-year (Q1..Q4), future growth uses the LAST observed same-quarter % growth
#   e.g., Q4-2026 factor = Q4-2025's factor; Q4-2027 uses the latest (which equals Q4-2026's)
# - Extends per bank to 2028 Q4 (inclusive)
# - Outputs NUMERICAL TABLES ONLY (no charts)

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
RAW_URL = "https://raw.githubusercontent.com/vincentlascano000/ccap_data/main/CCAP_DATA.csv"
TARGET_END = pd.Period("2028Q4", freq="Q")  # inclusive
BANK_ORDER_PREF = ["UB", "BDO", "BPI", "SECBANK", "MB", "RCBC"]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "quarter": ["quarter", "qtr", "period", "quarter_str", "q", "date"],
    "bank":    ["bank", "issuer", "bank_name", "issuer_name", "issuer bank"],

    "purchase_sales_bn": [
        "purchase_sales_bn","purchase_sales","sales",
        "purchase volume (bn)","purchase sales (in bn)","purchase_sales_bil","purchase_sales_billion"
    ],
    "cards_in_force_bn": [
        "cards_in_force_bn","cards_in_force","cif (bn)","cif","cif_in_bn","cif_bn","cards","c.i.f."
    ],
    "sales_per_cif_000": [
        "sales_per_cif_000","sales_per_cif","sales per cif (000)","sales/cif (000)","sales_cif"
    ],
}

FRIENDLY = {
    "purchase_sales_bn": "Purchase Sales (Bn)",
    "cards_in_force_bn": "Cards in Force (Bn)",
    "sales_per_cif_000": "Sales / CIF ('000)",
}

# =========================
# HELPERS
# =========================
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[()\[\]%]", "", s)
    s = re.sub(r"[\\/]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort auto-map headers to canonical names."""
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
    """Parse '1Q23', 'Q1 2023', '2023 Q1', or a date → (label, quarter-end Timestamp)."""
    if pd.isna(value):
        return None, pd.NaT
    s = str(value).strip().upper().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"^([1-4])Q(\d{2,4})$", s)  # 1Q23 / 4Q2025
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

    # Fallback: datetime
    try:
        dt = pd.to_datetime(s, errors="raise")
        per = pd.Period(dt, freq="Q")
        return str(per), per.to_timestamp(how="end")
    except Exception:
        return None, pd.NaT

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

# =========================
# UI
# =========================
st.set_page_config(page_title="CCAP — CIF × Sales/CIF (Latest Same-Quarter Growth)", layout="wide")
st.title("CCAP — Projections via CIF × Sales/CIF (Latest same-quarter growth) — Tables Only")

st.caption(f"Forecast end: **{str(TARGET_END)}** — For each quarter-of-year, future growth uses the latest observed same-quarter % change.")

st.sidebar.header("Output settings")
round_dec = st.sidebar.selectbox("Round decimals", options=[0,1,2], index=1)

# =========================
# LOAD & NORMALIZE
# =========================
try:
    raw = load_raw_csv(RAW_URL)
except Exception as e:
    st.error(f"Failed to read CSV from GitHub: {e}")
    st.stop()

df = map_columns(raw)

# Quarter parsing
if "quarter_dt" not in df.columns:
    qsrc = "quarter" if "quarter" in df.columns else None
    if qsrc is None:
        st.error("Missing 'quarter' or 'quarter_dt' column.")
        st.write("Detected columns:", list(df.columns))
        st.stop()
    df["quarter_dt"] = df[qsrc].apply(parse_quarter_token).apply(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)

# Bank
if "bank" not in df.columns:
    st.error("Missing 'bank' column after mapping.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Column mapping for CIF & Sales/CIF
st.sidebar.header("Column mapping (if needed)")
def ensure_col(name_key: str, label: str) -> str:
    if name_key in df.columns:
        return name_key
    cands = candidate_cols(df, COLUMN_SYNONYMS[name_key]) or list(df.columns)
    return st.sidebar.selectbox(f"Select column for {label}", options=cands, index=0)

cif_col = ensure_col("cards_in_force_bn", FRIENDLY["cards_in_force_bn"])
spc_col = ensure_col("sales_per_cif_000", FRIENDLY["sales_per_cif_000"])

# Optional: purchase sales (for scale)
if "purchase_sales_bn" not in df.columns:
    cands = candidate_cols(df, COLUMN_SYNONYMS["purchase_sales_bn"])
    if cands:
        sel = st.sidebar.selectbox("Select column for Purchase Sales (optional — splice scaling)", options=["<none>"] + cands, index=0)
        if sel != "<none>":
            df["purchase_sales_bn"] = df[sel]

# Coerce numeric
df[cif_col] = to_numeric(df[cif_col])
df[spc_col] = to_numeric(df[spc_col])
if "purchase_sales_bn" in df.columns:
    df["purchase_sales_bn"] = to_numeric(df["purchase_sales_bn"])

panel = (df[["bank","quarter_dt", cif_col, spc_col] + (["purchase_sales_bn"] if "purchase_sales_bn" in df.columns else [])]
         .dropna(subset=["bank","quarter_dt"])
         .sort_values(["bank","quarter_dt"])
         .reset_index(drop=True))

# Bank selection
banks_all = sorted(panel["bank"].dropna().unique().tolist(),
                   key=lambda x: (BANK_ORDER_PREF.index(x) if x in BANK_ORDER_PREF else 999, x))
banks_pick = st.multiselect("Banks to include", options=banks_all, default=banks_all)
if not banks_pick:
    st.info("Select at least one bank.")
    st.stop()

# =========================
# CORE: "Latest same-quarter growth" projection
# =========================
def last_same_quarter_factor(series: pd.Series, qtrs: pd.Series) -> Dict[int, float]:
    """
    Return latest observed factor (X_t / X_{t-1}) for each quarter-of-year k in {1..4}.
    If none available for a k, fall back to series' overall last factor or 1.0.
    """
    f = (series / series.shift(1)).astype(float)
    latest = {}
    overall_last = f.dropna().iloc[-1] if f.dropna().size > 0 else 1.0
    for k in (1,2,3,4):
        mask = (qtrs == k) & (f.notna())
        if mask.any():
            # last non-nan for that same quarter-of-year
            latest[k] = float(f[mask].iloc[-1])
        else:
            latest[k] = float(overall_last) if np.isfinite(overall_last) and overall_last > 0 else 1.0
        # safety
        if latest[k] <= 0 or not np.isfinite(latest[k]):
            latest[k] = 1.0
    return latest

def project_bank_latest_same_qtr(bank_df: pd.DataFrame,
                                 target_end: pd.Period) -> pd.DataFrame:
    """
    Project CIF and Sales/CIF by carrying forward the LATEST observed same-quarter factor.
    For each forecast quarter:
      X_{t+1} = X_t * factor_q  (factor_q = latest observed for that quarter-of-year q)
      (factor remains constant across future years unless new observed data arrives)
    Purchase Sales = scale × CIF × Sales/CIF, where scale aligns the splice (last actual).
    """
    g = bank_df.sort_values("quarter_dt").copy()
    g["qtr"] = g["quarter_dt"].dt.quarter

    last_dt = g["quarter_dt"].dropna().max()
    last_per = last_dt.to_period("Q")
    H = (target_end.year - last_per.year) * 4 + (target_end.quarter - last_per.quarter)
    H = int(max(0, H))
    if H == 0:
        return pd.DataFrame(columns=["quarter","bank","projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn"])

    # Last levels
    cif_level = float(g.iloc[-1][cif_col])
    spc_level = float(g.iloc[-1][spc_col])

    # Scale for Purchase Sales at splice (units reconciliation)
    if "purchase_sales_bn" in g.columns and pd.notna(g.iloc[-1].get("purchase_sales_bn", np.nan)) and cif_level and spc_level:
        denom = cif_level * spc_level
        scale = float(g.iloc[-1]["purchase_sales_bn"]) / denom if denom not in (None, 0, np.nan) else 1.0
    else:
        scale = 1.0

    # Latest observed same-quarter factors (constant into the future)
    latest_cif_factor = last_same_quarter_factor(g[cif_col], g["qtr"])
    latest_spc_factor = last_same_quarter_factor(g[spc_col], g["qtr"])

    rows = []
    for h in range(1, H + 1):
        next_per = last_per + h
        q = next_per.quarter

        f_cif = latest_cif_factor.get(q, 1.0)
        f_spc = latest_spc_factor.get(q, 1.0)

        # evolve levels
        cif_level *= f_cif
        spc_level *= f_spc

        ps_level = scale * cif_level * spc_level

        rows.append({
            "quarter": str(next_per),
            "projected_cif_bn": cif_level,
            "projected_sales_per_cif_000": spc_level,
            "projected_purchase_sales_bn": ps_level,
        })

    out = pd.DataFrame(rows)
    out["bank"] = g["bank"].iloc[0]
    return out[["quarter","bank","projected_cif_bn","projected_sales_per_cif_000","projected_purchase_sales_bn"]]

# =========================
# RUN PROJECTIONS
# =========================
proj_frames = []
for b in banks_pick:
    gbank = panel[panel["bank"] == b][["bank","quarter_dt", cif_col, spc_col] + (["purchase_sales_bn"] if "purchase_sales_bn" in panel.columns else [])]
    if gbank.shape[0] < 2:
        continue
    proj_b = project_bank_latest_same_qtr(gbank, TARGET_END)
    if not proj_b.empty:
        proj_frames.append(proj_b)

if not proj_frames:
    st.info("Not enough history or already at/after 2028 Q4 for selected banks — no projections to show.")
    st.stop()

projections = pd.concat(proj_frames, ignore_index=True)

# Round & order
projections["projected_cif_bn"] = projections["projected_cif_bn"].round(int(round_dec))
projections["projected_sales_per_cif_000"] = projections["projected_sales_per_cif_000"].round(int(round_dec))
projections["projected_purchase_sales_bn"] = projections["projected_purchase_sales_bn"].round(int(round_dec))

bank_order_map = {b: i for i, b in enumerate(BANK_ORDER_PREF)}
projections["_b_sort"] = projections["bank"].map(lambda x: bank_order_map.get(x, 999))
projections["_q_sort"] = projections["quarter"].apply(lambda q: (int(q[:4]) * 4) + int(q[-1]))
projections = projections.sort_values(["_q_sort","_b_sort","bank"]).drop(columns=["_q_sort","_b_sort"])

st.subheader("Projected Numbers — CIF × Sales/CIF → Purchase Sales (Latest same-quarter growth) to 2028 Q4")
st.dataframe(projections, use_container_width=True)
