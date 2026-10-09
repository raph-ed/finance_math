import numpy as np
import pandas as pd
from scipy.stats import norm

class BlackScholesEngine:
    """
    Standard Black-Scholes analytical engine for synthetic options hedging backtests
    and real-time Greeks sensitivity monitoring (Delta, Gamma, Theta, Vega).
    """
    @staticmethod
    def d1_d2(S, K, T, r, sigma):
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-5)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def put_price(cls, S, K, T, r, sigma):
        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return np.maximum(put, 0.0)

    @classmethod
    def call_price(cls, S, K, T, r, sigma):
        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return np.maximum(call, 0.0)

    @classmethod
    def compute_greeks(cls, S, K, T, r, sigma, opt_type="put"):
        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        T_sqrt = np.sqrt(np.maximum(T, 1e-5))
        pdf_d1 = norm.pdf(d1)
        
        gamma = pdf_d1 / (S * sigma * T_sqrt)
        vega = S * T_sqrt * pdf_d1 * 0.01  # 1% vol change
        
        if opt_type == "put":
            delta = norm.cdf(d1) - 1.0
            theta = (- (S * pdf_d1 * sigma) / (2 * T_sqrt) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        else:
            delta = norm.cdf(d1)
            theta = (- (S * pdf_d1 * sigma) / (2 * T_sqrt) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
            
        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega
        }

def simulate_options_hedged_portfolio(spot_prices, vix_series, regime_series, hedge_otm_pct=0.05, put_duration_days=30):
    """
    Simulates rolling protective put overlay during High Risk / Sell-Off regimes.
    Uses Black-Scholes with VIX as implied volatility proxy.
    Accounts for option premium decay (Theta) and payoff protection.
    """
    n = len(spot_prices)
    dates = spot_prices.index
    portfolio_val_unhedged = np.zeros(n)
    portfolio_val_hedged = np.zeros(n)
    
    portfolio_val_unhedged[0] = 100.0
    portfolio_val_hedged[0] = 100.0
    
    hedge_active = False
    put_strike = 0.0
    put_expiry_idx = 0
    put_entry_price = 0.0
    r = 0.03
    
    hedge_cost_tracker = []
    
    for i in range(1, n):
        ret_underlying = spot_prices.iloc[i] / spot_prices.iloc[i-1] - 1.0
        portfolio_val_unhedged[i] = portfolio_val_unhedged[i-1] * (1.0 + ret_underlying)
        
        # Check current regime
        current_date = dates[i]
        regime = regime_series.loc[current_date] if current_date in regime_series.index else 1
        vol_proxy = (vix_series.iloc[i] if current_date in vix_series.index else 20.0) / 100.0
        S_curr = spot_prices.iloc[i]
        
        # Protective put logic: Activate if regime is Sell-off (2) and not already protected
        if regime == 2 and not hedge_active:
            hedge_active = True
            put_strike = S_curr * (1.0 - hedge_otm_pct)
            put_expiry_idx = i + put_duration_days
            T = put_duration_days / 365.0
            put_entry_price = BlackScholesEngine.put_price(S_curr, put_strike, T, r, vol_proxy)
            cost_pct = put_entry_price / S_curr
            # Pay option premium
            portfolio_val_hedged[i] = portfolio_val_hedged[i-1] * (1.0 + ret_underlying - cost_pct)
            hedge_cost_tracker.append({"date": current_date, "type": "BUY_PUT", "cost_pct": cost_pct})
        elif hedge_active:
            days_left = max(1, put_expiry_idx - i)
            T_rem = days_left / 365.0
            curr_put_val = BlackScholesEngine.put_price(S_curr, put_strike, T_rem, r, vol_proxy)
            prev_T_rem = (days_left + 1) / 365.0
            prev_put_val = BlackScholesEngine.put_price(spot_prices.iloc[i-1], put_strike, prev_T_rem, r, vol_proxy)
            put_delta_pnl = (curr_put_val - prev_put_val) / spot_prices.iloc[i-1]
            
            portfolio_val_hedged[i] = portfolio_val_hedged[i-1] * (1.0 + ret_underlying + put_delta_pnl)
            
            if i >= put_expiry_idx or regime == 0:
                hedge_active = False # Expire or exit hedge when bull resumes
        else:
            portfolio_val_hedged[i] = portfolio_val_hedged[i-1] * (1.0 + ret_underlying)
            
    df_result = pd.DataFrame({
        "Unhedged": portfolio_val_unhedged,
        "Regime_Options_Hedged": portfolio_val_hedged
    }, index=dates)
    return df_result
