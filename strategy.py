import pandas as pd
import numpy as np
from indicators import rsi, wt, cci, adx, atr, rationalize_quadratic_kernel, gaussian_kernel
from lorentzian_ml import get_lorentzian_predictions

def calculate_strategy_signals(df):
    """
    Computes all indicators, ML predictions, and determines Buy/Sell signals.
    """
    if len(df) < 200:
        return df
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    hlc3 = (high + low + close) / 3
    
    # 1. Feature Engineering
    df['f1_rsi'] = rsi(close, 14)
    df['f2_wt'] = wt(high, low, close, 10, 11)
    df['f3_cci'] = cci(high, low, close, 20)
    df['f4_adx'] = adx(high, low, close, 20)
    df['f5_rsi'] = rsi(close, 9)
    
    # Fill NAs
    df.bfill(inplace=True)
    
    # Normalize features to 0-1 range to align with "normalized" logic in PS
    features_df = df[['f1_rsi', 'f2_wt', 'f3_cci', 'f4_adx', 'f5_rsi']].copy()
    features_df = (features_df - features_df.rolling(200).min()) / (features_df.rolling(200).max() - features_df.rolling(200).min() + 1e-8)
    features_df.fillna(0, inplace=True)
    
    # 2. Lorentzian ML Predictions
    # Optimized: increased neighbors_count from 8 to 10, added ml_threshold=1
    df['ml_prediction'] = get_lorentzian_predictions(features_df, close, neighbors_count=10, max_bars_back=2000)

    # 3. Filters (Optimized from 50 to 40 for faster intraday response)
    df['ema40'] = close.ewm(span=40, adjust=False).mean()
    df['sma40'] = close.rolling(window=40).mean()

    df['is_ema_uptrend'] = close > df['ema40']
    df['is_ema_downtrend'] = close < df['ema40']
    df['is_sma_uptrend'] = close > df['sma40']
    df['is_sma_downtrend'] = close < df['sma40']

    # ADX filter (optimized threshold from 20 to 18 for more trades)
    df['adx_filter'] = df['f4_adx'] > 18
    
    # Kernel Regression Filter (h=8, r=8.0, x=25)
    df['yhat1'] = rationalize_quadratic_kernel(close, h=8, r=8.0, x=25)
    df['yhat1_lag'] = df['yhat1'].shift(1)
    df['is_bullish'] = df['yhat1'] > df['yhat1_lag']
    df['is_bearish'] = df['yhat1'] < df['yhat1_lag']
    
    df['kernel_bullish_change'] = (df['yhat1'] > df['yhat1_lag']) & (df['yhat1'].shift(1) < df['yhat1'].shift(2))
    df['kernel_bearish_change'] = (df['yhat1'] < df['yhat1_lag']) & (df['yhat1'].shift(1) > df['yhat1'].shift(2))

    # Signal generation (optimized: added ml_threshold=1 to filter weak signals)
    df['signal'] = 0
    df.loc[(df['ml_prediction'] > 1) & df['adx_filter'], 'signal'] = 1
    df.loc[(df['ml_prediction'] < -1) & df['adx_filter'], 'signal'] = -1
    
    df['signal_changed'] = df['signal'] != df['signal'].shift(1)
    
    # Entry conditions
    df['startLongTrade'] = (df['signal'] == 1) & df['signal_changed'] & df['is_bullish'] & df['is_ema_uptrend'] & df['is_sma_uptrend']
    df['startShortTrade'] = (df['signal'] == -1) & df['signal_changed'] & df['is_bearish'] & df['is_ema_downtrend'] & df['is_sma_downtrend']
    
    return df
