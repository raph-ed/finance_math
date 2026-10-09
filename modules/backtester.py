import numpy as np
import pandas as pd

def compute_financial_metrics(series, risk_free_rate=0.03):
    """
    Computes institutional performance and risk metrics:
    CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown.
    """
    total_days = (series.index[-1] - series.index[0]).days
    years = max(total_days / 365.25, 0.2)
    
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0
    
    daily_returns = series.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    excess_ret = cagr - risk_free_rate
    sharpe = excess_ret / (ann_vol + 1e-8)
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = excess_ret / (downside_vol + 1e-8)
    
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    max_drawdown = drawdown.min()
    
    calmar = cagr / (abs(max_drawdown) + 1e-8)
    
    return {
        "Total Return": f"{total_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Annual Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Calmar Ratio": f"{calmar:.2f}",
    }

def run_strategy_backtest(df_prices, regime_series, transaction_cost_bps=10, slippage_bps=5):
    """
    Simulates dynamic strategy vs 60/40 benchmark and S&P 500 Buy & Hold,
    incorporating realistic round-trip transaction costs and slippage.
    """
    dates = regime_series.index.intersection(df_prices.index)
    prices = df_prices.loc[dates]
    returns = prices.pct_change().dropna()
    valid_dates = returns.index
    
    equity_assets = [c for c in prices.columns if c in ["SPY", "QQQ", "EZU", "VGK"]]
    bond_assets = [c for c in prices.columns if c in ["TLT", "SHY", "GLD"]]
    
    if not equity_assets:
        equity_assets = [prices.columns[0]]
    if not bond_assets:
        bond_assets = [prices.columns[-1]]
        
    bench_spy = (1 + returns[equity_assets[0]]).cumprod() * 100.0
    
    # 60/40 benchmark
    ret_60_40 = 0.6 * returns[equity_assets].mean(axis=1) + 0.4 * returns[bond_assets].mean(axis=1)
    bench_60_40 = (1 + ret_60_40).cumprod() * 100.0
    
    # Dynamic strategy
    strat_wealth = [100.0]
    total_cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0
    current_regime = None
    
    for i in range(1, len(valid_dates)):
        dt = valid_dates[i]
        regime = regime_series.loc[dt]
        
        # Apply rebalance friction when regime changes
        friction = total_cost_pct if (current_regime is not None and regime != current_regime) else 0.0
        current_regime = regime
        
        if regime == 0:  # Bull: 100% Equities
            step_ret = returns[equity_assets].iloc[i].mean() - friction
        elif regime == 1: # Neutral: 50% Equities, 50% Defensive
            step_ret = 0.5 * returns[equity_assets].iloc[i].mean() + 0.5 * returns[bond_assets].iloc[i].mean() - friction
        else: # High Risk: 10% Equities, 90% Defensive
            step_ret = 0.1 * returns[equity_assets].iloc[i].mean() + 0.9 * returns[bond_assets].iloc[i].mean() - friction
            
        strat_wealth.append(strat_wealth[-1] * (1.0 + step_ret))
        
    df_curves = pd.DataFrame({
        "Dynamic Regime ML Engine": strat_wealth,
        "Benchmark 60/40": bench_60_40.values,
        "S&P 500 Buy & Hold": bench_spy.values
    }, index=valid_dates)
    
    return df_curves
