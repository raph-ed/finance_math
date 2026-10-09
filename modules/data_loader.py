import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DEFAULT_ASSETS = {
    "US_EQUITIES": ["SPY", "QQQ"],
    "EU_EQUITIES": ["EZU", "VGK"],
    "DEFENSIVE": ["GLD", "TLT", "SHY"]
}

def generate_synthetic_market_data(start_date="2018-01-01", end_date=None):
    """Generates realistic synthetic multi-asset data aligned with current October 2026 pricing."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)
    
    np.random.seed(42)
    regimes = np.zeros(n)
    curr_state = 0
    transition = np.array([
        [0.97, 0.02, 0.01],
        [0.05, 0.90, 0.05],
        [0.03, 0.07, 0.90]
    ])
    for i in range(1, n):
        curr_state = np.random.choice([0, 1, 2], p=transition[curr_state])
        regimes[i] = curr_state
        
    tickers = ["SPY", "QQQ", "EZU", "VGK", "GLD", "TLT", "SHY"]
    drift_map = {0: 0.0006, 1: 0.0001, 2: -0.0012}
    vol_map = {0: 0.008, 1: 0.012, 2: 0.024}
    
    df_prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        base_price = 100.0 if ticker != "SPY" else 250.0
        ret = np.zeros(n)
        for i in range(n):
            st = regimes[i]
            beta = 1.2 if ticker == "QQQ" else (1.0 if ticker in ["SPY", "EZU", "VGK"] else (-0.3 if ticker in ["GLD", "TLT"] else 0.05))
            ret[i] = drift_map[st] * beta + np.random.normal(0, vol_map[st] * (abs(beta) + 0.2))
        df_prices[ticker] = base_price * np.exp(np.cumsum(ret))
        
    vix = 14 + regimes * 8 + np.random.normal(0, 1.5, n)
    vix = np.clip(vix, 10, 65)
    df_prices["^VIX"] = vix
    return df_prices

def load_multi_asset_data(tickers, start_date="2018-01-01", end_date=None, force_demo=False):
    """
    Télécharge les cours ajustés jusqu'au jour le plus récent (inclus intraday / clôture veille).
    Si end_date n'est pas spécifié, télécharge tout le flux disponible jusqu'à aujourd'hui.
    """
    if force_demo:
        return generate_synthetic_market_data(start_date, end_date), True

    all_symbols = list(set(tickers + ["^VIX"]))
    try:
        # Sans argument 'end', yfinance télécharge jusqu'à la dernière bougie disponible en direct
        kwargs = {"start": start_date, "progress": False, "auto_adjust": True}
        if end_date:
            # yfinance exclut la date de fin : on ajoute 1 jour si fourni par l'UI
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)
            kwargs["end"] = end_dt.strftime("%Y-%m-%d")

        data = yf.download(all_symbols, **kwargs)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                prices = data["Close"]
            else:
                prices = data.xs("Close", axis=1, level=0)
        else:
            prices = data
            
        prices = prices.dropna(how="all")
        if prices.empty or len(prices) < 30:
            raise ValueError("Données insuffisantes téléchargées.")
            
        prices = prices.ffill().bfill()
        return prices, False
    except Exception as e:
        print(f"yfinance indisponible ({e}), passage au générateur de marché démo.")
        return generate_synthetic_market_data(start_date, end_date), True
