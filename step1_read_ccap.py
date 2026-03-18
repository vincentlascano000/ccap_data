# app.py
import re
from pathlib import Path
from typing import Tuple, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Configuration
# ----------------------------
DEFAULT_FILE = "CCAP_DATA.CSV"

SECTION_METRIC_MAP = {
    'purchase sales (in bn)': ('purchase_sales_bn', 'float'),
    'purchase sales industry share (%)': ('purchase_sales_industry_share_pct', 'pct'),
    'balances (in bn)': ('balances_bn', 'float'),
    'balances industry share (%)': ('balances_industry_share_pct', 'pct'),
    'cards in force (in bn)': ('cards_in_force_bn', 'float'),
    'cif industry share (%)': ('cif_industry_share_pct', 'pct'),
    "sales / cif ('000)": ('sales_per_cif_000', 'float'),
    "balances / cif ('000)": ('balances_per_cif_000', 'float'),
}

BANK_CODE_MAP = {
    'UNIONBANK': 'UB',
    'BDO': 'BDO',
    'BPI': 'BPI',
    'METROBANK': 'MB',
    'RCBC': 'RCBC',
    'SECURITY BANK': 'SECB',
    'SECURITY_BANK': 'SECB',
}

# Display names for the UI (y-axis labels)
FRIENDLY_LABELS = {
    'purchase_sales_bn': "Purchase Sales (Bn)",
    'purchase_sales_industry_share_pct': "Purchase Sales Industry Share (decimal)",
    'balances_bn': "Balances (Bn)",
    'balances_industry_share_pct': "Balances Industry Share (decimal)",
    'cards_in_force_bn': "Cards in Force (Bn)",
    'cif_industry_share_pct': "CIF Industry Share (decimal)",
    'sales_per_cif_000': "Sales / CIF ('000)",
    'balances_per_cif_000': "Balances / CIF ('000)",
}


# ----------------------------
# Helpers
# ----------------------------
def parse_quarter_token(tok: str) -> Tuple[str, pd.Timestamp]:
    """
    Accept tokens like 1Q23, 2Q25, 2025Q3 and return (quarter_str, quarter_dt).
    """
    tok = str(tok).strip()
    m = re.match(r"^([1-4])Q(\d{2,4})$", tok, flags=re.IGNORECASE)
    if not m:
        m2 = re.match(r"^(\d{4})Q([1-4])$", tok, flags=re.IGNORECASE)
        if not m2:
            return None, pd.NaT
        year = int(m2.group(1))
        q = int(m2.group(2))
    else:
        q = int(m.group(1))
        yy = m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)

    per = pd.Period(freq='Q', year=year, quarter=q)
    return f"{year}Q{q}", per.to_timestamp(how='end')


def looks_like_panel(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    has_time = any(c in cols for c in ['quarter_dt', 'quarter_str', 'quarter', 'date'])
    has_bank = 'bank' in cols
    # Consider it panel if at least one known metric exists
    known_metrics = set(FRIENDLY_LABELS.keys())
    has_metric = len(known_metrics & cols) > 0
    return has_time and has_bank and has_metric


def load_raw_or_panel_csv(path: Path) -> pd.DataFrame:
    """
    If CSV is already a tidy panel, normalize and return.
    Otherwise, parse a sectioned file into a panel.
    Keeps percent metrics as decimals (e.g., 25% -> 0.25).
    """
    # Fast path: try panel
    try:
        df_try = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        raise

    if looks_like_panel(df_try):
        panel = df_try.copy()
        # Ensure quarter columns exist
        if 'quarter_dt' not in panel.columns:
            if 'quarter_str' in panel.columns:
                qdt = []
                for s in panel['quarter_str'].astype(str):
                    _, dt = parse_quarter_token(s)
                    qdt.append(dt)
                panel['quarter_dt'] = qdt
            elif 'quarter' in panel.columns:
                qdt, qstr = [], []
                for s in panel['quarter'].astype(str):
                    qs, dt = parse_quarter_token(s)
                    qstr.append(qs)
                    qdt.append(dt)
                panel['quarter_dt'] = qdt
                panel['quarter_str'] = qstr
        if 'quarter_str' not in panel.columns and 'quarter_dt' in panel.columns:
            panel['quarter_str'] = panel['quarter_dt'].dt.to_period('Q').astype(str)

        # Normalize bank codes
        panel['bank'] = panel['bank'].astype(str).str.upper().str.strip()
        panel['bank'] = panel['bank'].map(lambda b: BANK_CODE_MAP.get(b, b))
        return panel

    # Sectioned layout path: parse robustly
    raw_text = path.read_text(encoding='utf-8', errors='ignore')
    lines = [ln.strip() for ln in raw_text.replace('\r\n', '\n').replace('\r', '\n').split('\n') if ln.strip() != ""]
    sections = []
    idx = 0
    while idx < len(lines):
        header = lines[idx]
        if ',' in header:
            parts = [p.strip() for p in header.split(',')]
            title = parts[0]
            quarters = parts[1:]
            q_valid = sum(bool(re.match(r"^[1-4]Q\d{2,4}$", q)) for q in quarters)
            if q_valid >= max(3, len(quarters)//2):
                idx += 1
                bank_rows = []
                while idx < len(lines):
                    ln = lines[idx]
                    # skip comma dividers
                    if re.search(r"^,{2,}$", ln.replace(' ', '')):
                        idx += 1
                        continue
                    # new section?
                    if ',' in ln:
                        tparts = [p.strip() for p in ln.split(',')]
                        if tparts and re.match(r"^[A-Za-z].*", tparts[0]) is not None:
                            q_valid2 = sum(bool(re.match(r"^[1-4]Q\d{2,4}$", q)) for q in tparts[1:])
                            if q_valid2 >= max(3, len(tparts[1:])//2):
                                break
                    bank_rows.append(ln)
                    idx += 1
                sections.append((title, quarters, bank_rows))
                continue
        idx += 1

    # Build long records
    records = []
    for title, quarters, bank_rows in sections:
        key = title.strip().lower().lstrip('\ufeff')
        if key not in SECTION_METRIC_MAP:
            continue
        metric_name, value_type = SECTION_METRIC_MAP[key]

        q_labels, q_dates = [], []
        for qtok in quarters:
            ql, qd = parse_quarter_token(qtok)
            if ql is None:
                ql, qd = qtok.strip(), pd.NaT
            q_labels.append(ql)
            q_dates.append(qd)

        for row in bank_rows:
            parts = [p.strip() for p in row.split(',')]
            if len(parts) < 2:
                continue
            bank_raw = parts[0].upper().strip().replace(" ", "_")
            vals = parts[1:]
            if len(vals) < len(q_labels):
                vals += [''] * (len(q_labels) - len(vals))
            for ql, qd, v in zip(q_labels, q_dates, vals):
                v_clean = str(v).strip()
                if v_clean.endswith('%'):
                    try:
                        num = float(v_clean.replace('%', '').replace(' ', '')) / 100.0
                    except Exception:
                        num = np.nan
                else:
                    try:
                        num = float(v_clean.replace(',', ''))
                    except Exception:
                        num = np.nan
                if value_type == 'pct' and pd.notna(num) and num > 1:
                    num = num / 100.0
                records.append({
                    'quarter_str': ql,
                    'quarter_dt': qd,
                    'bank': bank_raw,
                    metric_name: num
                })

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        st.stop()

    panel_wide = panel.pivot_table(index=['quarter_dt', 'quarter_str', 'bank'], aggfunc='first').reset_index()
    panel_wide['bank'] = panel_wide['bank'].map(lambda b: BANK_CODE_MAP.get(b, b))
    # Friendly column order
    metric_cols = [c for c in FRIENDLY_LABELS.keys() if c in panel_wide.columns]
    cols = ['quarter_dt', 'quarter_str', 'bank'] + metric_cols
    panel_wide = panel_wide[cols].sort_values(['bank', 'quarter_dt']).reset_index(drop=True)
    return panel_wide


def build_metric_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
    label = FRIENDLY_LABELS.get(metric, metric)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X('quarter_dt:T', title='Quarter', axis=alt.Axis(format='%Y Q%q')),
            y=alt.Y(f'{metric}:Q', title=label),
            color=alt.Color('bank:N', title='Bank'),
            tooltip=[
                alt.Tooltip('quarter_str:N', title='Quarter'),
                alt.Tooltip('bank:N', title='Bank'),
                alt.Tooltip(f'{metric}:Q', title=label, format=',.2f')
            ]
        )
        .properties(height=320)
        .interactive()
    )
    return chart


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CCAP Metrics Viewer", layout="wide")
st.title("CCAP Metrics — Quarterly Trends")

file_path = st.text_input("Input CSV filename", value=DEFAULT_FILE, help="Put CCAP_DATA.CSV in the same folder and keep this default, or type a different path.")

path = Path(file_path)
if not path.exists():
    st.warning(f"File not found: {path.resolve()}")
    st.stop()

# Load and normalize
panel = load_raw_or_panel_csv(path)

# Sidebar controls
st.sidebar.header("Filters")
banks = sorted(panel['bank'].dropna().unique().tolist())
bank_sel = st.sidebar.multiselect("Bank(s)", options=banks, default=banks)
metrics_available = [c for c in FRIENDLY_LABELS.keys() if c in panel.columns]
metrics_sel = st.sidebar.multiselect(
    "Metric(s) to plot",
    options=[(m, FRIENDLY_LABELS[m]) for m in metrics_available],
    default=[(m, FRIENDLY_LABELS[m]) for m in metrics_available],
    format_func=lambda x: x[1]
)

# Filter
df_plot = panel[panel['bank'].isin(bank_sel)].copy()

# Layout: one chart per selected metric
if not metrics_sel:
    st.info("Select at least one metric on the left.")
else:
    # create columns grid (2 per row)
    cols_in_row = 2
    rows = (len(metrics_sel) + cols_in_row - 1) // cols_in_row
    idx = 0
    for r in range(rows):
        columns = st.columns(cols_in_row)
        for c in range(cols_in_row):
            if idx >= len(metrics_sel):
                break
            metric_key = metrics_sel[idx][0]  # ('metric_code', 'Friendly Name')
            with columns[c]:
                st.subheader(FRIENDLY_LABELS.get(metric_key, metric_key))
                chart = build_metric_chart(df_plot, metric_key)
                st.altair_chart(chart, use_container_width=True)
            idx += 1

# Data preview expander for trust-but-verify
with st.expander("Preview data (first 30 rows)"):
    st.dataframe(panel.head(30))
