import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from modules.translations import I18N
from modules.data_loader import load_multi_asset_data, DEFAULT_ASSETS
from modules.feature_engineering import build_features_and_targets
from modules.ml_regime_model import WalkForwardRegimeDetector
from modules.portfolio_optimizer import optimize_portfolio, compute_dynamic_regime_allocation, ledoit_wolf_covariance
from modules.options_hedging import BlackScholesEngine, simulate_options_hedged_portfolio
from modules.backtester import run_strategy_backtest, compute_financial_metrics

st.set_page_config(
    page_title="Quant Regime Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom institutional styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #151B26;
        border: 1px solid #252D3D;
        padding: 14px 18px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #8B949E;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] > div {
        color: #F0F6FC;
        font-size: 1.5rem;
        font-weight: 600;
    }
    /* Section Headings */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #E6EDF3;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .disclaimer-box {
        background-color: rgba(234, 179, 8, 0.08);
        border-left: 3px solid #EAB308;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #D1D5DB;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        background-color: #121721;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #00D2A0 !important;
        border-bottom: 2px solid #00D2A0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Language Selector in Sidebar Top
lang = st.sidebar.radio("🌐 Language / Langue", ["FR", "EN"], horizontal=True)
T = I18N[lang]

st.sidebar.markdown(f"### {T['sidebar_config']}")

# Offline / Demo Mode Toggle
data_mode = st.sidebar.selectbox(
    T["mode_selection"],
    [T["live_mode"], T["demo_mode"]]
)
is_demo = (data_mode == T["demo_mode"])

# Asset Selection
default_tickers = ["SPY", "QQQ", "EZU", "VGK", "GLD", "TLT", "SHY"]
selected_tickers = st.sidebar.multiselect(
    "Assets Basket / Panier d'Actifs",
    options=["SPY", "QQQ", "EZU", "VGK", "GLD", "TLT", "SHY", "AAPL", "MSFT", "NVDA", "BTC-USD"],
    default=default_tickers
)

# Model configuration
model_type = st.sidebar.selectbox("ML Regime Model", ["Random Forest", "Logistic Regression (L2)"])
model_key = "rf" if "Random Forest" in model_type else "logistic"

# Friction settings
cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10, step=5)
slippage_bps = st.sidebar.slider("Slippage (bps)", 0, 30, 5, step=5)
max_weight = st.sidebar.slider("Markowitz Max Asset Cap", 0.15, 0.60, 0.35, step=0.05)

# Date range
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))

# Main Title & Header
st.title(f"🏛️ {T['title']}")
st.caption(T['subtitle'])

# Legal Disclaimer Box
st.markdown(f"""
<div class="disclaimer-box">
    <strong>{T['disclaimer_title']}</strong> {T['disclaimer_body']}
</div>
""", unsafe_allow_html=True)

# Data Fetching
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_data(tickers, start_str, demo_flag):
    return load_multi_asset_data(tickers, start_date=start_str, force_demo=demo_flag)

with st.spinner("Processing institutional quantitative pipelines..."):
    df_prices, used_demo = get_cached_data(selected_tickers, str(start_date), is_demo)
    dataset, features = build_features_and_targets(df_prices)
    
    # Run Walk-Forward Regime Detector
    wf_detector = WalkForwardRegimeDetector(model_type=model_key, horizon=20, embargo=5)
    df_regimes = wf_detector.run_walk_forward(dataset, list(features.columns))

if used_demo:
    st.info("ℹ️ Mode démo actif : séries financières synthétiques calibrées sur processus stochastiques à 3 régimes.")

# Current Status Header
latest_date = df_regimes.index[-1]
curr_regime = int(df_regimes.loc[latest_date, "predicted_regime"])
curr_prob_bull = df_regimes.loc[latest_date, "prob_bull"] * 100
curr_prob_sell = df_regimes.loc[latest_date, "prob_selloff"] * 100

regime_labels = {
    0: (T["regime_0"], "#00D2A0"),
    1: (T["regime_1"], "#F59E0B"),
    2: (T["regime_2"], "#EF4444")
}
reg_name, reg_color = regime_labels[curr_regime]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(T["kpi_regime"], reg_name)
with kpi2:
    st.metric(T["kpi_prob_bull"], f"{curr_prob_bull:.1f}%")
with kpi3:
    st.metric(T["kpi_prob_selloff"], f"{curr_prob_sell:.1f}%")
with kpi4:
    last_vix = df_prices["^VIX"].iloc[-1] if "^VIX" in df_prices.columns else 18.5
    st.metric("VIX Index Proxy", f"{last_vix:.2f}")

# Tabs Organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    T["tab_overview"],
    T["tab_portfolio"],
    T["tab_backtest"],
    T["tab_options"],
    T["tab_interpret"]
])

# ----------------- TAB 1: OVERVIEW & REGIMES -----------------
with tab1:
    st.subheader("Market Regime Classification Over Time (Walk-Forward Out-Of-Sample)")
    
    ref_ticker = "SPY" if "SPY" in df_prices.columns else df_prices.columns[0]
    fig_regime = go.Figure()
    
    # Price line
    fig_regime.add_trace(go.Scatter(
        x=df_prices.loc[df_regimes.index].index,
        y=df_prices.loc[df_regimes.index, ref_ticker],
        mode='lines',
        name=f"{ref_ticker} Price",
        line=dict(color='#94A3B8', width=1.5)
    ))
    
    # Regime Scatter points
    colors = {0: "#00D2A0", 1: "#F59E0B", 2: "#EF4444"}
    names = {0: "Bullish", 1: "Neutral", 2: "High-Risk Selloff"}
    for reg_id in [0, 1, 2]:
        sub = df_regimes[df_regimes["predicted_regime"] == reg_id]
        if not sub.empty:
            fig_regime.add_trace(go.Scatter(
                x=sub.index,
                y=df_prices.loc[sub.index, ref_ticker],
                mode='markers',
                marker=dict(size=4.5, color=colors[reg_id]),
                name=names[reg_id]
            ))
            
    fig_regime.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#151B26",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_regime, use_container_width=True)
    
    # Feature Importances
    if wf_detector.feature_importances_ is not None:
        st.markdown("#### Feature Importance (Factors Driving the Regime Classifier)")
        imp_df = wf_detector.feature_importances_.sort_values(ascending=True)
        fig_imp = px.bar(
            x=imp_df.values,
            y=imp_df.index,
            orientation='h',
            color=imp_df.values,
            color_continuous_scale="Viridis",
            labels={"x": "Normalized Importance", "y": "Feature"}
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B0E14",
            plot_bgcolor="#151B26",
            height=320,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- TAB 2: PORTFOLIO ALLOCATION -----------------
with tab2:
    st.subheader("Markowitz Efficient Frontier & Shrinkage Dynamic Weighting")
    
    clean_rets = df_prices[[c for c in df_prices.columns if c != "^VIX"]].pct_change().dropna()
    base_opt_weights = optimize_portfolio(clean_rets, method="max_sharpe", max_weight=max_weight)
    dynamic_weights = compute_dynamic_regime_allocation(curr_regime, base_opt_weights)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("##### Base Optimal Weights (Max Sharpe)")
        fig_pie1 = px.pie(
            values=base_opt_weights.values,
            names=base_opt_weights.index,
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_pie1.update_layout(paper_bgcolor="#0B0E14", template="plotly_dark", height=320, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_pie1, use_container_width=True)
        
    with col_p2:
        st.markdown(f"##### Dynamic Weights ({reg_name})")
        fig_pie2 = px.pie(
            values=dynamic_weights.values,
            names=dynamic_weights.index,
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Burg
        )
        fig_pie2.update_layout(paper_bgcolor="#0B0E14", template="plotly_dark", height=320, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_pie2, use_container_width=True)

# ----------------- TAB 3: BACKTESTING -----------------
with tab3:
    st.subheader("Performance Comparison vs 60/40 & Buy-and-Hold Benchmarks")
    df_curves = run_strategy_backtest(
        df_prices[[c for c in df_prices.columns if c != "^VIX"]],
        df_regimes["predicted_regime"],
        transaction_cost_bps=cost_bps,
        slippage_bps=slippage_bps
    )
    
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(x=df_curves.index, y=df_curves["Dynamic Regime ML Engine"], name="Dynamic Regime ML Engine", line=dict(color="#00D2A0", width=2.5)))
    fig_backtest.add_trace(go.Scatter(x=df_curves.index, y=df_curves["Benchmark 60/40"], name="Benchmark 60/40", line=dict(color="#3B82F6", width=1.8, dash="dot")))
    fig_backtest.add_trace(go.Scatter(x=df_curves.index, y=df_curves["S&P 500 Buy & Hold"], name="S&P 500 Buy & Hold", line=dict(color="#6B7280", width=1.5)))
    
    fig_backtest.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#151B26",
        height=450,
        yaxis_title="Normalized Portfolio Value (Base 100)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    # Metrics Table
    m_strat = compute_financial_metrics(df_curves["Dynamic Regime ML Engine"])
    m_6040 = compute_financial_metrics(df_curves["Benchmark 60/40"])
    m_spy = compute_financial_metrics(df_curves["S&P 500 Buy & Hold"])
    
    df_metrics = pd.DataFrame([m_strat, m_6040, m_spy], index=["Dynamic Regime ML Engine", "Benchmark 60/40", "S&P 500 Buy & Hold"]).T
    st.dataframe(df_metrics, use_container_width=True)

# ----------------- TAB 4: OPTIONS HEDGING -----------------
with tab4:
    st.subheader("Options Overlay: Systematic Protective Put Simulation")
    ref_spot = df_prices[ref_ticker]
    vix_s = df_prices["^VIX"] if "^VIX" in df_prices.columns else pd.Series(18.0, index=df_prices.index)
    
    df_hedged = simulate_options_hedged_portfolio(
        ref_spot.loc[df_regimes.index],
        vix_s.loc[df_regimes.index],
        df_regimes["predicted_regime"]
    )
    
    fig_hedge = go.Figure()
    fig_hedge.add_trace(go.Scatter(x=df_hedged.index, y=df_hedged["Regime_Options_Hedged"], name="Regime Dynamic Put Overlay", line=dict(color="#F59E0B", width=2.5)))
    fig_hedge.add_trace(go.Scatter(x=df_hedged.index, y=df_hedged["Unhedged"], name="Unhedged Underlying", line=dict(color="#64748B", width=1.5)))
    fig_hedge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#151B26",
        height=420,
        yaxis_title="Base 100",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_hedge, use_container_width=True)
    
    # Real-time Greeks Dashboard for Current Protective Put
    st.markdown("#### Real-Time Black-Scholes Greeks Sensitivity Tracker")
    S_now = float(ref_spot.iloc[-1])
    K_now = S_now * 0.95
    vix_now = float(vix_s.iloc[-1]) / 100.0
    greeks = BlackScholesEngine.compute_greeks(S=S_now, K=K_now, T=30/365.0, r=0.03, sigma=vix_now, opt_type="put")
    
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Delta (Δ)", f"{greeks['Delta']:.3f}", help="Hedge ratio sensitivity per $1 move in underlying")
    with g2:
        st.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}", help="Rate of change of Delta (convexity protection)")
    with g3:
        st.metric("Theta (Θ)", f"${greeks['Theta']:.2f}/day", help="Time decay cost per contract per calendar day")
    with g4:
        st.metric("Vega (ν)", f"{greeks['Vega']:.3f}", help="Sensitivity to +1% volatility spike")

# ----------------- TAB 5: HOW TO INTERPRET & METHODOLOGY -----------------
with tab5:
    if lang == "FR":
        st.markdown("""
        ### Guide d'Interprétation & Méthodologie Institutionnelle
        
        #### 1. Détection de Régimes (Machine Learning Walk-Forward)
        * **Pourquoi pas prédire le cours exact ?** Les séries de prix quotidiennes ont un ratio signal/bruit proche de zéro. Prédire le cours exact conduit inévitablement au surapprentissage (*overfitting*). Prédire l'**état de volatilité et de retournement (régime)** est en revanche économiquement fondé.
        * **Embargo & Purging :** Les rendements futurs étant calculés sur 20 jours, nous purgeons les 25 derniers jours de chaque jeu d'entraînement pour éliminer le biais du futur (*look-ahead bias*).
        
        #### 2. Réserve méthodologique Options (Point clé d'entretien)
        * *yfinance* ne fournit pas l'historique intraday de la chaîne d'options fermée. Nous simulons donc les primes passées via le modèle analytique de **Black-Scholes (1973)** en utilisant le **VIX** (ou la volatilité réalisée) comme proxy de la volatilité implicite.
        
        #### 3. Optimisation Moyenne-Variance (Markowitz & Shrinkage)
        * L'inversion directe de la matrice de covariance empirique est instable lorsque $N$ actifs sont corrélés. Nous utilisons l'estimateur de **Ledoit-Wolf Shrinkage** pour conditionner la matrice.
        * Les contraintes *long-only* (poids $\in [0, 35\%]$) évitent la surconcentration sur 1 ou 2 actifs à fort momentum passé.
        
        #### 4. Friction de marché
        * Contrairement aux backtests académiques naïfs, chaque transition de régime impute 10 bps de frais de courtage et 5 bps de slippage estimé.
        """)
    else:
        st.markdown("""
        ### Institutional Interpretation & Methodology Guide
        
        #### 1. Regime Detection (Walk-Forward Machine Learning)
        * **Why avoid predicting next-day prices?** Raw daily price returns have an ultra-low signal-to-noise ratio. Predicting price levels induces overfitting. Conversely, classifying **volatility regimes & distribution states** reflects structural market dynamics.
        * **Embargo & Purging :** Because forward returns span 20 trading days, we enforce a strict 25-day purging window between train and test splits to strictly prevent look-ahead contamination.
        
        #### 2. Options Methodology Caveat (Key Interview Point)
        * Free historical options tick feeds are unavailable on public APIs. Thus, historical options overlays are dynamically evaluated using the **Black-Scholes analytical formula** with the **VIX index** as an implied volatility proxy.
        
        #### 3. Portfolio Optimization (Markowitz & Shrinkage)
        * Standard sample covariance matrices suffer from noise and extreme weight allocation. We apply **Ledoit-Wolf covariance shrinkage** with a 35% single-asset cap to enforce institutional diversification.
        
        #### 4. Execution Frictions
        * Backtests systematically integrate 10 bps transaction fees and 5 bps execution slippage per regime turnover.
        """)
