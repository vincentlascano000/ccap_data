# app.py
import re
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Defaults for your GitHub repo
# ----------------------------
DEFAULT_GH_USER = "vincentlascano000"
DEFAULT_GH_REPO = "ccap_data"
DEFAULT_BRANCH  = "main"     # change if you use 'master' or another branch
DEFAULT_PATH_IN_REPO = "CCAP_DATA.CSV"  # adjust to exact path inside repo

# ----------------------------
# Known sections & metrics
# ----------------------------
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
    'SECURITY_BANK': 'SECB',  # in case underscores are used
}

# Human-friendly labels for Y axis
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
# Helper functions
# ----------------------------
def build_github_raw_url(user: str, repo: str, branch: str, path_in_repo: str) -> str:
    """Create a GitHub raw URL from repo components."""
    # sanitize possible leading slashes
    path_in_repo = path_in_repo.lstrip("/")
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path_in_repo}"

def normalize_github_url(url: str) -> str:
    """
    Accept either a 'blob' URL or a 'raw' URL.
    Convert blob to raw if necessary.
    """
    url = url.strip()
    m = re.match(r"^https://github\\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", url)
    if m:
        user, repo, branch, path = m.groups()
        return build_github_raw_url(user, repo, branch, path)
    return url  # assume already raw

def parse_quarter_token(tok: str) -> Tuple[str, pd.Timestamp]:
    """Parse '1Q23', '2Q25', or '2025Q3' into ('YYYYQn', quarter_end_timestamp)."""
    tok = str(tok).strip()
    m = re.match(r"^([1-4])Q(\d{2,4})$", tok, flags=re.IGNORECASE)
    if not m:
        m2 = re.match(r"^(\d{4})Q([1-4])$", tok, flags=re.IGNORECASE)
        if not m2:
            return None, pd.NaT
        year = int(m2.group(1)); q = int(m2.group(2))
    else:
        q = int(m.group(1)); yy = m.group(2)
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
    per = pd.Period(freq='Q', year=year, quarter=q)
    return f"{year}Q{q}", per.to_timestamp(how='end')

def looks_like_panel(df: pd.DataFrame) -> bool:
    """Determine if dataframe already looks like a tidy panel (bank + quarter + metrics)."""
    cols = {c.lower() for c in df.columns}
    has_time = any(c in cols for c in ['quarter_dt', 'quarter_str', 'quarter', 'date'])
    has_bank = 'bank' in cols
    known_metrics = set(FRIENDLY_LABELS.keys())
    has_metric = len(known_metrics & cols) > 0
    return has_time and has_bank and has_metric

@st.cache_data(ttl=600)
def read_csv_from_source(source: str, prefer_local: bool = False) -> pd.DataFrame:
    """
    Read CSV from a URL (raw GitHub or direct) or local path.
    Cached for 10 minutes.
    """
    if prefer_local:
        return pd.read_csv(Path(source))
    return pd.read_csv(source)

def load_raw_or_panel(path_or_url: str) -> pd.DataFrame:
    """
    If CSV is already a tidy panel, normalize & return.
    Otherwise, parse sectioned layout into a panel.
    Keeps percent metrics as decimals (e.g., 25% -> 0.25).
    """
    is_url = path_or_url.lower().startswith("http")
    src = normalize_github_url(path_or_url) if is_url else path_or_url

    try:
        df_try = read_csv_from_source(src, prefer_local=not is_url)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if looks_like_panel(df_try):
        panel = df_try.copy()
        # Ensure quarter_dt/quarter_str
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
                    qstr.append(qs); qdt.append(dt)
                panel['quarter_dt'] = qdt; panel['quarter_str'] = qstr
        if 'quarter_str' not in panel.columns and 'quarter_dt' in panel.columns:
            panel['quarter_str'] = panel['quarter_dt'].dt.to_period('Q').astype(str)

        # Normalize bank codes
        panel['bank'] = panel['bank'].astype(str).str.upper().str.strip()
        panel['bank'] = panel['bank'].map(lambda b: BANK_CODE_MAP.get(b, b))
        return panel

    # Otherwise, parse sectioned layout
    # Re-read raw text to preserve original structure for parsing
    if is_url:
        import urllib.request
        with urllib.request.urlopen(src) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
    else:
        raw_text = Path(src).read_text(encoding="utf-8", errors="ignore")

    lines = [ln.strip() for ln in raw_text.replace('\r\n', '\n').replace('\r', '\n').split('\n') if ln.strip() != ""]
    sections = []
    idx = 0
    while idx < len(lines):
        header = lines[idx]
        if ',' in header:
            parts = [p.strip() for p in header.split(',')]
            title = parts[0]
            quarters = parts[1:]
            q_valid = sum(bool(re.match(r"^[1-4]Q\\d{2,4}$", q)) for q in quarters)
            if q_valid >= max(3, len(quarters)//2):
                idx += 1
                bank_rows = []
                while idx < len(lines):
                    ln = lines[idx]
                    if re.search(r"^,{2,}$", ln.replace(' ', '')):
                        idx += 1; continue
                    if ',' in ln:
                        tparts = [p.strip() for p in ln.split(',')]
                        if tparts and re.match(r"^[A-Za-z].*", tparts[0]) is not None:
                            q_valid2 = sum(bool(re.match(r"^[1-4]Q\\d{2,4}$", q)) for q in tparts[1:])
                            if q_valid2 >= max(3, len(tparts[1:])//2):
                                break
                    bank_rows.append(ln); idx += 1
                sections.append((title, quarters, bank_rows)); continue
        idx += 1

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
            q_labels.append(ql); q_dates.append(qd)

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
        st.error("Parsed no rows—check your file structure and path.")
        st.stop()

    panel_wide = panel.pivot_table(index=['quarter_dt','quarter_str','bank'], aggfunc='first').reset_index()
    panel_wide['bank'] = panel_wide['bank'].map(lambda b: BANK_CODE_MAP.get(b, b))
    metric_cols = [c for c in FRIENDLY_LABELS if c in panel_wide.columns]
    cols = ['quarter_dt','quarter_str','bank'] + metric_cols
    panel_wide = panel_wide[cols].sort_values(['bank','quarter_dt']).reset_index(drop=True)
    return panel_wide

def build_metric_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
    label = FRIENDLY_LABELS.get(metric, metric)
    return (
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

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CCAP Metrics — GitHub Loader", layout="wide")
st.title("CCAP Metrics — Quarterly Trends")

with st.sidebar:
    st.header("Choose Data Source")
    mode = st.radio("Source", ["GitHub (username/repo)", "GitHub (paste URL)", "Local file / Upload"], index=0)

    if mode == "GitHub (username/repo)":
        gh_user = st.text_input("GitHub Username", value=DEFAULT_GH_USER)
        gh_repo = st.text_input("Repository", value=DEFAULT_GH_REPO)
        gh_branch = st.text_input("Branch", value=DEFAULT_BRANCH)
        gh_path = st.text_input("Path inside repo", value=DEFAULT_PATH_IN_REPO,
                                help="e.g., CCAP_DATA.CSV or data/CCAP_DATA.CSV")
        src = build_github_raw_url(gh_user, gh_repo, gh_branch, gh_path)

    elif mode == "GitHub (paste URL)":
        url = st.text_input("GitHub URL (blob or raw)", value="")
        src = normalize_github_url(url) if url else ""

    else:  # Local file / Upload
        local = st.text_input("Local CSV path (optional)", value="")
        uploaded = st.file_uploader("...or upload CSV", type=["csv"])
        if uploaded is not None:
            # write temp file so downstream code can read via path
            tmp_path = Path("uploaded_temp.csv")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            src = str(tmp_path)
        else:
            src = local

# Guard clause
if not src:
    st.info("Provide a valid GitHub URL/coordinates or select a local file.")
    st.stop()

# Load & normalize
panel = load_raw_or_panel(src)

# Sidebar filters
st.sidebar.header("Filters")
banks = sorted(panel['bank'].dropna().unique().tolist())
bank_sel = st.sidebar.multiselect("Bank(s)", options=banks, default=banks)

metrics_available = [c for c in FRIENDLY_LABELS if c in panel.columns]
metrics_sel = st.sidebar.multiselect(
    "Metric(s) to plot",
    options=metrics_available,
    default=metrics_available,
    format_func=lambda m: FRIENDLY_LABELS[m]
)

# Filter and plot
df_plot = panel[panel['bank'].isin(bank_sel)].copy()

if not metrics_sel:
    st.info("Select at least one metric to plot.")
else:
    cols_in_row = 2
    rows = (len(metrics_sel) + cols_in_row - 1) // cols_in_row
    idx = 0
    for _ in range(rows):
        columns = st.columns(cols_in_row)
        for c in range(cols_in_row):
            if idx >= len(metrics_sel):
                break
            metric = metrics_sel[idx]
            with columns[c]:
                st.subheader(FRIENDLY_LABELS.get(metric, metric))
                st.altair_chart(build_metric_chart(df_plot, metric), use_container_width=True)
            idx += 1

# Data preview
with st.expander("Preview data (first 30 rows)"):
    st.dataframe(panel.head(30))
