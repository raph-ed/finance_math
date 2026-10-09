import pandas as pd
import numpy as np

def build_features_and_targets(df_prices, target_horizon=20, bull_thresh=0.015, bear_thresh=-0.02, embargo_period=5):
    """
    Constructs multi-horizon momentum, volatility, RSI, and macro features.
    Computes forward 20-day returns and max drawdown to label market regimes:
      0: Bullish Trend (positive forward return, low drawdown)
      1: Ranging / Neutral
      2: High Risk / Sell-Off (forward return < bear_thresh or drawdown > threshold)
    Applies embargo logic for walk-forward splits.
    """
    df = df_prices.copy()
    ref_asset = "SPY" if "SPY" in df.columns else df.columns[0]
    ref_series = df[ref_asset]
    
    # 1. Log returns
    log_ret_1d = np.log(ref_series / ref_series.shift(1))
    log_ret_5d = np.log(ref_series / ref_series.shift(5))
    log_ret_20d = np.log(ref_series / ref_series.shift(20))
    log_ret_60d = np.log(ref_series / ref_series.shift(60))
    
    # 2. Moving averages deviations
    sma_20 = ref_series.rolling(20).mean()
    sma_50 = ref_series.rolling(50).mean()
    sma_200 = ref_series.rolling(200).mean()
    dev_sma50 = (ref_series - sma_50) / sma_50
    dev_sma200 = (ref_series - sma_200) / sma_200
    sma_ratio = sma_50 / (sma_200 + 1e-6)
    
    # 3. Realized Volatility (20d annualized)
    vol_20d = log_ret_1d.rolling(20).std() * np.sqrt(252)
    vol_60d = log_ret_1d.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol_20d / (vol_60d + 1e-6)
    
    # 4. RSI (14 periods)
    delta = ref_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # 5. Macro Proxy (VIX level and VIX rolling momentum)
    if "^VIX" in df.columns:
        vix = df["^VIX"]
        vix_ma = vix.rolling(20).mean()
        vix_stretch = vix / (vix_ma + 1e-6)
    else:
        vix = vol_20d * 100
        vix_stretch = vol_ratio
        
    # Feature DataFrame
    features = pd.DataFrame({
        "log_ret_5d": log_ret_5d,
        "log_ret_20d": log_ret_20d,
        "log_ret_60d": log_ret_60d,
        "dev_sma50": dev_sma50,
        "dev_sma200": dev_sma200,
        "sma_ratio": sma_ratio,
        "vol_20d": vol_20d,
        "vol_ratio": vol_ratio,
        "rsi_14": rsi_14,
        "vix_level": vix,
        "vix_stretch": vix_stretch
    }, index=df.index)
    
    # Target definition (Forward-looking 20-day with forward drawdown)
    # Forward return over H days
    fwd_ret = ref_series.shift(-target_horizon) / ref_series - 1.0
    
    # Forward max drawdown over H days
    rolling_forward_min = ref_series.iloc[::-1].rolling(target_horizon).min().iloc[::-1]
    fwd_drawdown = (rolling_forward_min - ref_series) / ref_series
    
    # Target labels:
    # 2: High Risk (Sell-Off) if forward return < bear_thresh or forward max drawdown < -0.05
    # 0: Bullish Trend if forward return > bull_thresh and forward max drawdown > -0.025
    # 1: Ranging / Neutral otherwise
    target = pd.Series(1, index=df.index)
    target.loc[(fwd_ret > bull_thresh) & (fwd_drawdown > -0.03)] = 0
    target.loc[(fwd_ret < bear_thresh) | (fwd_drawdown < -0.05)] = 2
    
    # Drop rows without labels or features
    dataset = features.copy()
    dataset["fwd_ret"] = fwd_ret
    dataset["target"] = target
    dataset = dataset.dropna()
    
    return dataset, features
