import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

try:
    from dashboard.data_loader import load_csv_safely, calculate_advanced_metrics
except ModuleNotFoundError:
    from data_loader import load_csv_safely, calculate_advanced_metrics


SIGNAL_PATH = "data/signal_log.csv"
TRADE_PATH = "data/trade_journal.csv"
PAPER_ORDER_PATH = "data/paper_orders.json"
CANDLE_PATH = "data/candle_snapshot.json"
ACTIVE_ORDER_KEY = "bot:execution:active_orders"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Bot Monitor",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .main { background-color: #080c12; color: #c2cad6; }

    h1 { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important;
         font-weight: 800 !important; letter-spacing: -0.02em !important;
         color: #e8edf5 !important; margin-bottom: 0 !important; }

    h2 { font-family: 'Syne', sans-serif !important; font-size: 1.05rem !important;
         font-weight: 700 !important; color: #c2cad6 !important;
         text-transform: uppercase; letter-spacing: 0.08em !important;
         border-bottom: 1px solid #1c2535; padding-bottom: 6px; margin-bottom: 12px !important; }

    h3 { font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important;
         font-weight: 600 !important; color: #8a94a6 !important;
         text-transform: uppercase; letter-spacing: 0.1em !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #0d1420;
        border: 1px solid #1c2535;
        border-top: 2px solid #1c2535;
        padding: 14px 16px 12px;
        border-radius: 6px;
        transition: border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover { border-color: #2c3f5a; }

    div[data-testid="stMetricValue"] > div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #e8edf5 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        color: #4a5568 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 0.7rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Positive / Negative metric accents */
    .metric-pos div[data-testid="stMetricValue"] > div { color: #00e676 !important; }
    .metric-neg div[data-testid="stMetricValue"] > div { color: #ff4f6e !important; }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a5568 !important;
        background: transparent !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00e676 !important;
        background: #0d1420 !important;
        border-bottom: 2px solid #00e676 !important;
    }

    /* Dataframe */
    .stDataFrame { border: 1px solid #1c2535 !important; border-radius: 6px !important; }
    .stDataFrame thead th {
        background: #0d1420 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        color: #4a5568 !important;
        letter-spacing: 0.08em;
    }

    /* Status badges */
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 3px;
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
    }
    .badge-ok { background: #0a2a1a; color: #00e676; border: 1px solid #00e676; }
    .badge-warn { background: #2a1a00; color: #ffb03a; border: 1px solid #ffb03a; }
    .badge-err { background: #2a0a0f; color: #ff4f6e; border: 1px solid #ff4f6e; }

    /* Status bar */
    .status-bar {
        display: flex; align-items: center; gap: 20px; padding: 8px 14px;
        background: #0d1420; border: 1px solid #1c2535; border-radius: 6px;
        margin-bottom: 18px; flex-wrap: wrap;
    }
    .status-item { display: flex; align-items: center; gap: 6px;
                   font-size: 0.68rem; color: #4a5568; }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; }
    .dot-green { background: #00e676; box-shadow: 0 0 6px #00e676; }
    .dot-yellow { background: #ffb03a; }
    .dot-red { background: #ff4f6e; }

    /* Divider */
    hr { border-color: #1c2535 !important; }

    /* Info/warning banners */
    .stAlert { border-radius: 6px !important; font-size: 0.75rem !important;
               font-family: 'JetBrains Mono', monospace !important; }

    /* Plotly chart container */
    .stPlotlyChart { border: 1px solid #1c2535; border-radius: 6px; overflow: hidden; }

    /* Button */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important; text-transform: uppercase;
        letter-spacing: 0.08em; background: #0d1420 !important;
        border: 1px solid #2c3f5a !important; color: #8a94a6 !important;
        border-radius: 4px !important; padding: 6px 18px !important;
    }
    .stButton > button:hover {
        border-color: #00e676 !important; color: #00e676 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Plotly theme ──────────────────────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#4a5568", size=11),
    margin=dict(t=30, b=30, l=10, r=10),
    xaxis=dict(gridcolor="#1c2535", linecolor="#1c2535", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1c2535", linecolor="#1c2535", tickfont=dict(size=10)),
    colorway=["#00e676", "#ff4f6e", "#3d9cff", "#ffb03a", "#b06bff"],
)

GREEN = "#00e676"
RED = "#ff4f6e"
BLUE = "#3d9cff"
AMBER = "#ffb03a"


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1)
def get_csv(file_path: str):
    return load_csv_safely(file_path)


@st.cache_data(ttl=2)
def get_active_orders():
    rows = []
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url, decode_responses=True,
                                       socket_connect_timeout=1, socket_timeout=1)
        raw = client.get(ACTIVE_ORDER_KEY)
        client.close()
        if raw:
            data = json.loads(raw)
            rows.extend(
                dict(order) | {"order_id": oid, "source": "order_manager"}
                for oid, order in data.items()
            )
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    try:
        if os.path.exists(PAPER_ORDER_PATH):
            with open(PAPER_ORDER_PATH, "r", encoding="utf-8") as f:
                paper_rows = json.load(f)
            if isinstance(paper_rows, list):
                rows.extend(row for row in paper_rows if isinstance(row, dict))
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    return pd.DataFrame(rows), None


@st.cache_data(ttl=2)
def get_candle_snapshot():
    if not os.path.exists(CANDLE_PATH):
        return {}, None
    try:
        with open(CANDLE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}, None
    except Exception as exc:
        return {}, str(exc)


def normalize_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.dropna(subset=["timestamp"])
    for col in ["entry", "sl", "target", "quant_score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["symbol", "side", "status", "rejection_reason"]:
        if col in work.columns:
            work[col] = work[col].fillna("")
    return work


def dedupe_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    work = df.copy()
    work["_bucket"] = work["timestamp"].dt.floor("5min")
    subset = [c for c in ["_bucket", "symbol", "side", "status", "rejection_reason"] if c in work.columns]
    work = work.sort_values("timestamp").drop_duplicates(subset=subset, keep="last")
    return work.drop(columns=["_bucket"])


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    date_col = "date" if "date" in work.columns else "timestamp" if "timestamp" in work.columns else None
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
    for col in ["entry_price", "exit_price", "qty", "pnl_inr", "pnl_after_costs", "confidence", "quant_score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


# ── Load data ─────────────────────────────────────────────────────────────────
df_signals_raw, signal_err = get_csv(SIGNAL_PATH)
df_trades_raw, trade_err = get_csv(TRADE_PATH)
df_signals_raw = normalize_signals(df_signals_raw)
df_signals = dedupe_signals(df_signals_raw)
df_trades = normalize_trades(df_trades_raw)
df_active_orders, redis_err = get_active_orders()
candle_snapshot, candle_err = get_candle_snapshot()

today = datetime.now().date()
signals_today = (
    df_signals[df_signals["timestamp"].dt.date == today]
    if not df_signals.empty and "timestamp" in df_signals.columns
    else pd.DataFrame()
)
signals_view = signals_today if not signals_today.empty else df_signals
trades_today = (
    df_trades[df_trades["date"].dt.date == today]
    if not df_trades.empty and "date" in df_trades.columns
    else pd.DataFrame()
)
trades_view = trades_today if not trades_today.empty else df_trades

# ── Computed KPIs ─────────────────────────────────────────────────────────────
pnl_col = "pnl_after_costs" if "pnl_after_costs" in trades_view.columns else "pnl_inr"
net_pnl = trades_view[pnl_col].sum() if not trades_view.empty and pnl_col in trades_view.columns else 0.0
win_rate = (
    trades_view[pnl_col].gt(0).sum() / len(trades_view) * 100
    if not trades_view.empty and pnl_col in trades_view.columns
    else None
)
adv = calculate_advanced_metrics(trades_view, pnl_col)

signal_count = len(signals_view)
trade_count = len(signals_view[signals_view["status"] == "TRADE"]) if not signals_view.empty and "status" in signals_view.columns else 0
no_trade_count = len(signals_view[signals_view["status"] == "NO_TRADE"]) if not signals_view.empty and "status" in signals_view.columns else 0
closed_trades = len(trades_view)

active_risk = 0.0
if not df_active_orders.empty:
    for _, row in df_active_orders.iterrows():
        try:
            active_risk += abs(float(row.get("entry", 0)) - float(row.get("sl", 0))) * int(float(row.get("qty", 1)))
        except (TypeError, ValueError):
            pass


# ── Header ────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.title("NSE Paper Trading Monitor")
with hdr_right:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("⟳  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Status bar
sig_dot = "dot-green" if os.path.exists(SIGNAL_PATH) else "dot-red"
trade_dot = "dot-green" if os.path.exists(TRADE_PATH) else "dot-red"
redis_dot = "dot-green" if not redis_err else "dot-yellow"
st.markdown(
    f"""
    <div class="status-bar">
      <div class="status-item">
        <div class="status-dot {sig_dot}"></div>
        <span>Signal CSV</span>
      </div>
      <div class="status-item">
        <div class="status-dot {trade_dot}"></div>
        <span>Trade Journal</span>
      </div>
      <div class="status-item">
        <div class="status-dot {redis_dot}"></div>
        <span>Redis</span>
      </div>
      <div class="status-item" style="margin-left:auto; color:#2c3f5a">
        Updated {datetime.now().strftime('%H:%M:%S')}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if signal_err:
    st.warning(f"Signal log: {signal_err}")
if trade_err:
    st.warning(f"Trade journal: {trade_err}")
if redis_err:
    st.warning(f"Redis: {redis_err}")
if candle_err:
    st.warning(f"Candle snapshot: {candle_err}")

# ── Top KPI strip ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

pnl_color = GREEN if net_pnl >= 0 else RED
k1.markdown(f"""
<div data-testid="stMetric" style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid {pnl_color};padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Net P&L</div>
  <div style="font-size:1.35rem;font-weight:600;color:{pnl_color};
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    ₹{net_pnl:,.2f}
  </div>
</div>""", unsafe_allow_html=True)

wr_color = GREEN if (win_rate or 0) >= 50 else RED
k2.markdown(f"""
<div style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid {wr_color};padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Win Rate</div>
  <div style="font-size:1.35rem;font-weight:600;color:{wr_color};
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    {f"{win_rate:.1f}%" if win_rate is not None else "—"}
  </div>
</div>""", unsafe_allow_html=True)

pf = adv["profit_factor"]
pf_color = GREEN if pf != float("inf") and pf >= 1.5 else AMBER if pf >= 1.0 else RED
pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
k3.markdown(f"""
<div style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid {pf_color};padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Profit Factor</div>
  <div style="font-size:1.35rem;font-weight:600;color:{pf_color};
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    {pf_str}
  </div>
</div>""", unsafe_allow_html=True)

exp = adv["expectancy"]
exp_color = GREEN if exp > 0 else RED
k4.markdown(f"""
<div style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid {exp_color};padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Expectancy</div>
  <div style="font-size:1.35rem;font-weight:600;color:{exp_color};
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    ₹{exp:,.2f}
  </div>
</div>""", unsafe_allow_html=True)

k5.markdown(f"""
<div style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid #1c2535;padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Max Drawdown</div>
  <div style="font-size:1.35rem;font-weight:600;color:{RED};
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    ₹{adv["max_drawdown"]:,.2f}
  </div>
</div>""", unsafe_allow_html=True)

k6.markdown(f"""
<div style="background:#0d1420;border:1px solid #1c2535;
     border-top:2px solid #1c2535;padding:14px 16px 12px;border-radius:6px;">
  <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
              font-family:'JetBrains Mono',monospace;">Active Orders</div>
  <div style="font-size:1.35rem;font-weight:600;color:#e8edf5;
              font-family:'JetBrains Mono',monospace;margin-top:4px;">
    {len(df_active_orders)} <span style="font-size:.8rem;color:#4a5568">/ {closed_trades} closed</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Performance", "Signals", "Orders", "Risk", "Health", "Alpha Insights"])


# ─── TAB 0: PERFORMANCE ───────────────────────────────────────────────────────
with tabs[0]:
    if trades_view.empty:
        st.info("No closed trade data. Equity curve and analytics will appear once trades are journaled.")
    else:
        date_col = "date" if "date" in trades_view.columns else "timestamp"
        pnl_curve_col = "pnl_after_costs" if "pnl_after_costs" in trades_view.columns else "pnl_inr"
        curve = trades_view.sort_values(date_col).copy()
        curve["cum_pnl"] = curve[pnl_curve_col].cumsum()
        curve["peak"] = curve["cum_pnl"].cummax()
        curve["drawdown"] = curve["peak"] - curve["cum_pnl"]
        curve["trade_num"] = range(1, len(curve) + 1)

        # Equity + Drawdown dual chart
        fig = go.Figure()

        # Drawdown fill (behind equity)
        fig.add_trace(go.Scatter(
            x=curve[date_col], y=-curve["drawdown"],
            fill="tozeroy", name="Drawdown",
            line=dict(color=RED, width=0),
            fillcolor="rgba(255,79,110,0.12)",
            hovertemplate="Drawdown: ₹%{y:,.2f}<extra></extra>",
        ))

        # Equity line
        line_color = GREEN if curve["cum_pnl"].iloc[-1] >= 0 else RED
        fig.add_trace(go.Scatter(
            x=curve[date_col], y=curve["cum_pnl"],
            mode="lines", name="Equity",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'0,230,118' if line_color == GREEN else '255,79,110'},0.05)",
            hovertemplate="Equity: ₹%{y:,.2f}<extra></extra>",
        ))

        # Breakeven line
        fig.add_hline(y=0, line_dash="dot", line_color="#2c3f5a", line_width=1)

        fig.update_layout(
            **CHART_BASE,
            height=320,
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            yaxis_title="INR",
            hovermode="x unified",
        )
        st.subheader("Equity Curve & Drawdown")
        st.plotly_chart(fig, use_container_width=True)

        # P&L Distribution + Symbol Performance side-by-side
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("P&L Distribution")
            fig2 = go.Figure()
            wins = curve[curve[pnl_curve_col] > 0][pnl_curve_col]
            losses = curve[curve[pnl_curve_col] < 0][pnl_curve_col]
            fig2.add_trace(go.Histogram(
                x=wins, name="Winners", marker_color=GREEN,
                opacity=0.75, nbinsx=20,
                hovertemplate="₹%{x:,.0f}<extra>Winner</extra>",
            ))
            fig2.add_trace(go.Histogram(
                x=losses, name="Losers", marker_color=RED,
                opacity=0.75, nbinsx=20,
                hovertemplate="₹%{x:,.0f}<extra>Loser</extra>",
            ))
            fig2.update_layout(**CHART_BASE, height=280, barmode="overlay",
                               showlegend=True,
                               legend=dict(orientation="h", y=1.08, x=0,
                                           bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Performance by Symbol")
            if "symbol" in trades_view.columns:
                sym_perf = trades_view.groupby("symbol").agg(
                    Trades=("symbol", "count"),
                    Net_PnL=(pnl_col, "sum"),
                    Wins=(pnl_col, lambda x: (x > 0).sum()),
                ).reset_index()
                sym_perf["Win%"] = (sym_perf["Wins"] / sym_perf["Trades"] * 100).round(1)
                sym_perf = sym_perf.sort_values("Net_PnL", ascending=False)

                fig3 = go.Figure(go.Bar(
                    x=sym_perf["symbol"],
                    y=sym_perf["Net_PnL"],
                    marker_color=[GREEN if v >= 0 else RED for v in sym_perf["Net_PnL"]],
                    text=sym_perf["Net_PnL"].apply(lambda v: f"₹{v:,.0f}"),
                    textposition="outside",
                    hovertemplate="%{x}: ₹%{y:,.2f}<extra></extra>",
                ))
                fig3.update_layout(**CHART_BASE, height=280,
                                   yaxis_title="Net P&L (INR)",
                                   showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

        # Unified Activity Feed
        st.subheader("Unified Signal & Activity Feed")
        if signals_view.empty:
            st.info("No signal data available for the selected period.")
        else:
            # Prepare display dataframe
            feed_df = signals_view.sort_values("timestamp", ascending=False).copy()
            
            # Select and rename columns for clarity
            cols = {
                "timestamp": "Time",
                "symbol": "Symbol",
                "side": "Side",
                "status": "Decision",
                "quant_score": "AI Score",
                "rejection_reason": "Reason / Detail"
            }
            
            # Only keep columns that exist
            existing_cols = [c for c in cols.keys() if c in feed_df.columns]
            display_df = feed_df[existing_cols].rename(columns={c: cols[c] for c in existing_cols})

            def style_decision(val):
                color = '#00e676' if val == 'TRADE' else '#ff4f6e'
                return f'color: {color}; font-weight: bold;'

            st.dataframe(
                display_df.head(100).style.applymap(style_decision, subset=['Decision'] if 'Decision' in display_df.columns else []),
                hide_index=True,
                use_container_width=True
            )


# ─── TAB 1: SIGNALS ──────────────────────────────────────────────────────────
with tabs[1]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Signal Volume by Symbol")
        if signals_view.empty:
            st.info("No signal data available.")
        elif "symbol" not in signals_view.columns:
            st.info("Symbol column missing.")
        else:
            sym_counts = (signals_view.groupby("symbol")
                          .size().reset_index(name="count")
                          .sort_values("count", ascending=True).tail(15))
            fig = go.Figure(go.Bar(
                y=sym_counts["symbol"], x=sym_counts["count"],
                orientation="h",
                marker_color=BLUE,
                text=sym_counts["count"], textposition="outside",
            ))
            fig.update_layout(**CHART_BASE, height=340, showlegend=False,
                               xaxis_title="Signal Count")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top Rejection Reasons")
        rejections = (signals_view[signals_view["status"] == "NO_TRADE"]
                      if not signals_view.empty and "status" in signals_view.columns
                      else pd.DataFrame())
        if rejections.empty:
            st.info("No rejections in current window.")
        else:
            reason_counts = (
                rejections.assign(reason=rejections["rejection_reason"].replace("", "Unknown"))
                .groupby("reason").size().reset_index(name="count")
                .sort_values("count", ascending=True).tail(10)
            )
            fig = go.Figure(go.Bar(
                y=reason_counts["reason"], x=reason_counts["count"],
                orientation="h", marker_color=AMBER,
                text=reason_counts["count"], textposition="outside",
            ))
            fig.update_layout(**CHART_BASE, height=340, showlegend=False,
                               xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # Signal quality heatmap (quant_score by hour)
    if not signals_view.empty and "quant_score" in signals_view.columns and "timestamp" in signals_view.columns:
        st.subheader("Quant Score Heat Map (Hour × Symbol)")
        heat_df = signals_view.copy()
        heat_df["hour"] = heat_df["timestamp"].dt.hour
        if "symbol" in heat_df.columns:
            pivot = (heat_df.groupby(["symbol", "hour"])["quant_score"]
                     .mean().reset_index()
                     .pivot(index="symbol", columns="hour", values="quant_score"))
            fig = px.imshow(
                pivot, color_continuous_scale=[[0, "#0d1420"], [0.4, "#1c3a5a"],
                                               [0.7, "#00aaff"], [1, GREEN]],
                aspect="auto",
            )
            fig.update_layout(**CHART_BASE, height=280,
                               coloraxis_colorbar=dict(tickfont=dict(size=9)))
            fig.update_traces(hovertemplate="Symbol: %{y}<br>Hour: %{x}:00<br>Avg Score: %{z:.2f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

    # TRADE signal details
    st.subheader("TRADE Candidates")
    trade_signals = (signals_view[signals_view["status"] == "TRADE"]
                     if not signals_view.empty and "status" in signals_view.columns
                     else pd.DataFrame())
    if trade_signals.empty:
        st.info("No TRADE signals in the current window.")
    else:
        dc = [c for c in ["timestamp", "symbol", "side", "entry", "sl", "target", "quant_score"]
              if c in trade_signals.columns]
        st.dataframe(
            trade_signals.sort_values("timestamp", ascending=False)[dc].head(100),
            hide_index=True, use_container_width=True,
        )


# ─── TAB 2: ORDERS ────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Order Lifecycle")
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Open Paper Orders", len(df_active_orders))
    oc2.metric("Closed Paper Orders", len(trades_view))
    oc3.metric(
        "Closed Winners",
        len(trades_view[trades_view[pnl_col] > 0])
        if not trades_view.empty and pnl_col in trades_view.columns
        else 0,
    )
    oc4.metric(
        "Closed Losers",
        len(trades_view[trades_view[pnl_col] < 0])
        if not trades_view.empty and pnl_col in trades_view.columns
        else 0,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Open Paper Orders")
        if df_active_orders.empty:
            st.info("No open paper orders right now. Closed orders are shown below.")
        else:
            cols = [c for c in ["order_id", "symbol", "side", "entry", "sl", "target",
                                  "qty", "qty_open", "status", "strategy", "source"] if c in df_active_orders.columns]
            st.dataframe(df_active_orders[cols], hide_index=True, use_container_width=True)

    with c2:
        st.subheader("Outcomes Distribution")
        if trades_view.empty:
            st.info("No closed trades yet.")
        elif "outcome" not in trades_view.columns:
            st.info("No outcome column in trade journal.")
        else:
            outcome_counts = (trades_view.groupby("outcome")
                               .size().reset_index(name="count")
                               .sort_values("count", ascending=False))
            outcome_colors = {o: (GREEN if "win" in o.lower() or "target" in o.lower()
                                  else RED if "loss" in o.lower() or "sl" in o.lower()
                                  else AMBER)
                              for o in outcome_counts["outcome"]}
            fig = go.Figure(go.Bar(
                x=outcome_counts["outcome"],
                y=outcome_counts["count"],
                marker_color=[outcome_colors.get(o, BLUE) for o in outcome_counts["outcome"]],
                text=outcome_counts["count"], textposition="outside",
            ))
            fig.update_layout(**CHART_BASE, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.subheader("Closed Paper Orders")
    if trades_view.empty:
        st.info("No closed paper orders yet.")
    else:
        _date_col = "date" if "date" in trades_view.columns else "timestamp"
        closed = trades_view.sort_values(_date_col, ascending=False).copy()
        closed["status"] = closed["outcome"] if "outcome" in closed.columns else "CLOSED"
        display_cols = [c for c in ["date", "symbol", "side", "strategy", "entry_price", "exit_price",
                                     "qty", "pnl_inr", "pnl_after_costs", "outcome", "confidence"]
                        if c in closed.columns]
        st.dataframe(
            closed[display_cols],
            hide_index=True, use_container_width=True,
        )


# ─── TAB 3: RISK ──────────────────────────────────────────────────────────────
with tabs[3]:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Active Risk (INR)", f"₹{active_risk:,.2f}")
    r2.metric("Open Orders", len(df_active_orders))
    r3.metric(
        "Closed Losses",
        len(trades_view[trades_view[pnl_col] < 0])
        if not trades_view.empty and pnl_col in trades_view.columns
        else 0,
    )
    r4.metric(
        "Filter Rate",
        f"{(no_trade_count / signal_count * 100):.1f}%"
        if signal_count else "—",
        help="% of signals filtered as NO_TRADE",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if not trades_view.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("R:R Analysis")
            avg_win = trades_view[trades_view[pnl_col] > 0][pnl_col].mean()
            avg_loss = abs(trades_view[trades_view[pnl_col] < 0][pnl_col].mean())
            rr = (avg_win / avg_loss) if avg_loss and pd.notna(avg_loss) and pd.notna(avg_win) else 0.0

            fig = go.Figure(go.Bar(
                x=["Avg Winner", "Avg Loser"],
                y=[avg_win if pd.notna(avg_win) else 0, avg_loss if pd.notna(avg_loss) else 0],
                marker_color=[GREEN, RED],
                text=[f"₹{avg_win:,.0f}" if pd.notna(avg_win) else "—",
                      f"₹{avg_loss:,.0f}" if pd.notna(avg_loss) else "—"],
                textposition="outside",
            ))
            fig.update_layout(**CHART_BASE, height=260, showlegend=False,
                               yaxis_title="INR")
            fig.add_annotation(
                text=f"Achieved R:R  1 : {rr:.2f}",
                xref="paper", yref="paper", x=0.5, y=1.06,
                showarrow=False,
                font=dict(color=GREEN if rr >= 1.5 else AMBER if rr >= 1 else RED,
                          size=11, family="JetBrains Mono"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Daily P&L")
            if "date" in trades_view.columns:
                daily = (trades_view.groupby(trades_view["date"].dt.date)[pnl_col]
                         .sum().reset_index())
                daily.columns = ["date", "pnl"]
                fig2 = go.Figure(go.Bar(
                    x=daily["date"], y=daily["pnl"],
                    marker_color=[GREEN if v >= 0 else RED for v in daily["pnl"]],
                    text=daily["pnl"].apply(lambda v: f"₹{v:,.0f}"),
                    textposition="outside",
                ))
                fig2.add_hline(y=0, line_dash="dot", line_color="#2c3f5a", line_width=1)
                fig2.update_layout(**CHART_BASE, height=260, showlegend=False,
                                   yaxis_title="INR")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Date column unavailable for daily breakdown.")
    else:
        st.info("Risk analytics require closed trade rows in the journal.")


# ─── TAB 4: HEALTH ────────────────────────────────────────────────────────────
with tabs[4]:
    raw_count = len(df_signals_raw)
    health_data = [
        {"Check": "Signal CSV", "Status": "OK" if os.path.exists(SIGNAL_PATH) else "Missing", "Detail": SIGNAL_PATH},
        {"Check": "Trade Journal", "Status": "OK" if os.path.exists(TRADE_PATH) else "Missing", "Detail": TRADE_PATH},
        {"Check": "Redis", "Status": "OK" if not redis_err else "Warning", "Detail": redis_err or f"{len(df_active_orders)} orders"},
        {"Check": "Candle Snapshot", "Status": "OK" if candle_snapshot else "Missing", "Detail": CANDLE_PATH},
        {"Check": "Duplicate Signals", "Status": "Info", "Detail": f"{max(raw_count - len(df_signals), 0)} collapsed (5-min buckets)"},
        {"Check": "Ticks DB", "Status": "OK" if os.path.exists("data/ticks.db") else "Missing", "Detail": "data/ticks.db"},
    ]

    def badge(status):
        cls = {"OK": "badge-ok", "Warning": "badge-warn", "Missing": "badge-err"}.get(status, "badge-warn")
        return f'<span class="badge {cls}">{status}</span>'

    table_html = """
    <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:.75rem;">
      <thead>
        <tr style="border-bottom:1px solid #1c2535;">
          <th style="text-align:left;padding:8px 12px;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;font-size:.65rem;">Check</th>
          <th style="text-align:left;padding:8px 12px;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;font-size:.65rem;">Status</th>
          <th style="text-align:left;padding:8px 12px;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;font-size:.65rem;">Detail</th>
        </tr>
      </thead>
      <tbody>
    """
    for row in health_data:
        table_html += f"""
        <tr style="border-bottom:1px solid #0d1420;">
          <td style="padding:8px 12px;color:#8a94a6;">{row["Check"]}</td>
          <td style="padding:8px 12px;">{badge(row["Status"])}</td>
          <td style="padding:8px 12px;color:#4a5568;">{row["Detail"] or "—"}</td>
        </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.subheader("Historical Warm-up Files")

    configured_symbols: list = []
    try:
        with open("config/config.yaml", "r") as _f:
            _cfg = yaml.safe_load(_f)
        configured_symbols = (
            _cfg.get("instruments", {}).get("equity", []) +
            _cfg.get("instruments", {}).get("currency", [])
        )
    except Exception:
        configured_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK",
                               "USDINR", "EURINR", "GBPINR", "JPYINR"]

    history_dir = "data/historical"
    hist_rows = []
    for sym in configured_symbols:
        path = os.path.join(history_dir, f"{sym}_6m.parquet")
        hist_rows.append({"Symbol": sym, "File": path, "Present": os.path.exists(path)})
    st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.subheader("Latest 15-Minute Candles")
    symbols_data = candle_snapshot.get("symbols", {}) if candle_snapshot else {}
    rows = []
    for symbol, by_tf in symbols_data.items():
        candles_15m = by_tf.get("15min") or []
        if not candles_15m:
            continue
        last = candles_15m[-1]
        rows.append({
            "symbol": symbol,
            "timestamp": last.get("timestamp"),
            "open": last.get("open"),
            "high": last.get("high"),
            "low": last.get("low"),
            "close": last.get("close"),
            "volume": last.get("volume"),
            "oi": last.get("oi"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("symbol"), hide_index=True, use_container_width=True)
    else:
        st.info("No 15-minute candle snapshot yet. Start the bot and wait for live ticks, or check historical warm-up files.")


# ─── TAB 5: ALPHA INSIGHTS (RSMB vs Existing) ────────────────────────────────
with tabs[5]:
    st.markdown("## 🔬 Dual-Strategy Performance Lab")

    if trades_view.empty or "strategy" not in trades_view.columns:
        st.info(
            "Waiting for strategy-tagged trades to appear. "
            "The RSMB strategy runs on the **15-minute** candle close. "
            "Existing strategies run on the **5-minute** close. "
            "Both write to the same `trade_journal.csv` with a `strategy` column."
        )
    else:
        # Split datasets
        rsmb_trades = trades_view[trades_view["strategy"] == "rsmb"]
        existing_trades = trades_view[trades_view["strategy"] != "rsmb"]

        rsmb_signals = signals_view[signals_view["strategy"] == "rsmb"] if not signals_view.empty and "strategy" in signals_view.columns else pd.DataFrame()
        existing_signals = signals_view[signals_view["strategy"] != "rsmb"] if not signals_view.empty and "strategy" in signals_view.columns else pd.DataFrame()

        # --- KPI Scorecard ---
        st.subheader("Strategy Scorecard")
        k_r, k_e = st.columns(2)

        def _kpi_card(label, value, color="#00e676"):
            return f"""<div style="background:#0d1420;border:1px solid #1c2535;border-top:2px solid {color};
            padding:14px 16px 12px;border-radius:6px;margin-bottom:8px;">
            <div style="font-size:.65rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em;
            font-family:'JetBrains Mono',monospace;">{label}</div>
            <div style="font-size:1.2rem;font-weight:600;color:#e8edf5;
            font-family:'JetBrains Mono',monospace;margin-top:4px;">{value}</div></div>"""

        def _strategy_kpis(df, name, color):
            m = calculate_advanced_metrics(df)
            pnl = df["pnl_after_costs"].sum() if not df.empty and "pnl_after_costs" in df.columns else 0.0
            pnl_color = "#00e676" if pnl >= 0 else "#ff4f6e"
            pf = m.get("profit_factor", 0.0)
            pf_display = "∞" if pf == float("inf") else f"{pf:.2f}"
            cards = [
                _kpi_card("Trades", len(df), color),
                _kpi_card("Net P&L", f"₹{pnl:+.2f}", pnl_color),
                _kpi_card("Win Rate", f"{m.get('win_rate', 0.0):.1f}%", color),
                _kpi_card("Profit Factor", pf_display, color),
                _kpi_card("Expectancy", f"₹{m.get('expectancy', 0.0):.2f}", color),
                _kpi_card("Max Drawdown", f"₹{m.get('max_drawdown', 0.0):.2f}", "#ff4f6e"),
            ]
            return cards

        with k_r:
            st.markdown(f"### 🟢 RSMB Strategy")
            for card in _strategy_kpis(rsmb_trades, "rsmb", "#00e676"):
                st.markdown(card, unsafe_allow_html=True)

        with k_e:
            st.markdown(f"### 🔵 Existing Strategy")
            for card in _strategy_kpis(existing_trades, "existing", "#3d9cff"):
                st.markdown(card, unsafe_allow_html=True)

        st.markdown("---")

        # --- Cumulative Equity Curves (overlaid) ---
        st.subheader("Cumulative Equity Curves")
        pnl_c = "pnl_after_costs" if "pnl_after_costs" in trades_view.columns else "pnl_inr"
        date_c = "date" if "date" in trades_view.columns else "timestamp"

        eq_fig = go.Figure()
        for strat_name, strat_df, color in [
            ("RSMB", rsmb_trades, GREEN),
            ("Existing", existing_trades, BLUE),
        ]:
            if not strat_df.empty:
                curve = strat_df.sort_values(date_c).copy()
                curve["cum_pnl"] = curve[pnl_c].cumsum()
                curve["trade_num"] = range(1, len(curve) + 1)
                eq_fig.add_trace(go.Scatter(
                    x=curve["trade_num"], y=curve["cum_pnl"],
                    mode="lines", name=strat_name,
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba({'0,230,118' if color == GREEN else '61,156,255'},0.05)",
                    hovertemplate=f"{strat_name} Trade #%{{x}}: ₹%{{y:,.2f}}<extra></extra>",
                ))
        eq_fig.update_layout(
            **CHART_BASE,
            xaxis_title="Trade Number",
            yaxis_title="Cumulative P&L (₹)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(color="#c2cad6")),
        )
        st.plotly_chart(eq_fig, use_container_width=True)

        st.markdown("---")

        # --- Signal Accuracy by Strategy ---
        c_acc, c_rej = st.columns(2)
        with c_acc:
            st.subheader("Signal Acceptance Rate")
            if not signals_view.empty and "strategy" in signals_view.columns:
                acc_df = signals_view.groupby(["strategy", "status"]).size().reset_index(name="count")
                # Map strategy names to readable labels
                acc_df["strategy"] = acc_df["strategy"].apply(
                    lambda s: "RSMB" if s == "rsmb" else s
                )
                acc_fig = px.bar(
                    acc_df, x="strategy", y="count", color="status",
                    barmode="group",
                    color_discrete_map={"TRADE": GREEN, "NO_TRADE": RED},
                    labels={"count": "Signals", "strategy": "Strategy"},
                )
                acc_fig.update_layout(**CHART_BASE)
                st.plotly_chart(acc_fig, use_container_width=True)

        with c_rej:
            st.subheader("Top Rejection Reasons")
            if not signals_view.empty and "rejection_reason" in signals_view.columns:
                rej = signals_view[signals_view["status"] == "NO_TRADE"].copy()
                rej["reason_clean"] = rej["rejection_reason"].fillna("Unknown").str.split(":").str[0]
                reason_counts = rej.groupby(["strategy", "reason_clean"]).size().reset_index(name="count")
                reason_counts["strategy"] = reason_counts["strategy"].apply(
                    lambda s: "RSMB" if s == "rsmb" else s
                )
                rej_fig = px.bar(
                    reason_counts.sort_values("count", ascending=False).head(15),
                    x="count", y="reason_clean", color="strategy",
                    orientation="h",
                    color_discrete_map={"RSMB": GREEN, "TrendFollowing": BLUE,
                                        "MeanReversion": AMBER, "Ensemble_AI": "#b06bff"},
                    labels={"count": "Count", "reason_clean": "Reason"},
                )
                rej_fig.update_layout(**CHART_BASE, height=320)
                st.plotly_chart(rej_fig, use_container_width=True)

        st.markdown("---")

        # --- Per-Strategy Trade Log ---
        st.subheader("Trade Log by Strategy")
        tab_rsmb, tab_existing, tab_all = st.tabs(["RSMB Trades", "Existing Trades", "All Trades"])

        def _trade_table(df):
            if df.empty:
                st.info("No trades for this strategy in the selected period.")
                return
            display = df.sort_values(date_c, ascending=False).head(50)
            st.dataframe(
                display.style.applymap(
                    lambda v: f"color: {'#00e676' if v in ('TARGET_HIT', 'T1_HIT') else '#ff4f6e' if v == 'SL_HIT' else '#c2cad6'}",
                    subset=["outcome"] if "outcome" in display.columns else []
                ),
                use_container_width=True,
                hide_index=True,
            )

        with tab_rsmb:
            _trade_table(rsmb_trades)

        with tab_existing:
            _trade_table(existing_trades)

        with tab_all:
            _trade_table(trades_view)
