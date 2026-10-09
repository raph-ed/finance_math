import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(returns):
    """Calculates Ledoit-Wolf shrinkage covariance matrix to avoid ill-conditioned inversion."""
    lw = LedoitWolf()
    cov_shrunk = lw.fit(returns.dropna()).covariance_
    return pd.DataFrame(cov_shrunk * 252, index=returns.columns, columns=returns.columns)

def optimize_portfolio(returns, method="max_sharpe", max_weight=0.35, risk_free_rate=0.03):
    """
    Markowitz mean-variance optimization with long-only constraints and max asset caps.
    """
    clean_rets = returns.dropna()
    mean_rets = clean_rets.mean() * 252
    cov_matrix = ledoit_wolf_covariance(clean_rets)
    num_assets = len(mean_rets)
    
    # Bounds: Long-only (0% to max_weight)
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})
    
    init_weights = np.ones(num_assets) / num_assets
    
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        
    def neg_sharpe(w):
        port_ret = np.sum(mean_rets * w)
        port_vol = portfolio_vol(w)
        return -(port_ret - risk_free_rate) / (port_vol + 1e-8)
        
    if method == "min_volatility":
        res = minimize(portfolio_vol, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        res = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
    weights = np.clip(res.x, 0.0, 1.0)
    weights = weights / np.sum(weights)
    return pd.Series(weights, index=returns.columns)

def compute_dynamic_regime_allocation(regime, base_weights, defensive_assets=["GLD", "TLT", "SHY"]):
    """
    Adjusts asset allocation dynamically according to predicted regime:
    - Bull (0): 100% active risky basket
    - Neutral (1): 70% active risky, 30% defensive/cash
    - Sell-Off (2): 20% active risky, 80% defensive/cash
    """
    adj_weights = base_weights.copy()
    risky_assets = [a for a in base_weights.index if a not in defensive_assets]
    def_assets = [a for a in base_weights.index if a in defensive_assets]
    
    if regime == 0:  # Bullish
        mult_risky = 1.0
        mult_def = 0.5
    elif regime == 1: # Neutral
        mult_risky = 0.7
        mult_def = 1.2
    else:             # High Risk Sell-Off
        mult_risky = 0.15
        mult_def = 2.0
        
    for a in risky_assets:
        adj_weights[a] *= mult_risky
    for a in def_assets:
        adj_weights[a] *= mult_def
        
    adj_weights = adj_weights / np.sum(adj_weights)
    return adj_weights
