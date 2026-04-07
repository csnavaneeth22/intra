import pandas as pd
import numpy as np

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def wt(high, low, close, n1=10, n2=11):
    ap = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = abs(ap - esa).ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    tci = ci.ewm(span=n2, adjust=False).mean()
    return tci

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    # pd.Series.mad is deprecated, using alternative:
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def tr(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high, low, close, period=14):
    return tr(high, low, close).rolling(period).mean()

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    # +DI and -DI
    tr14 = tr(high, low, close).rolling(period).sum()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr14)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr14)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def rationalize_quadratic_kernel(src, h, r, x):
    """
    Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel.
    h = Lookback window (bandwidth)
    r = Relative weighting
    x = Regression Level (bar index offset)
    """
    # For a rolling window, computing kernel regression.
    # To keep it efficient in python for backtesting, we compute a rolling estimation.
    # Formula: yhat_i = sum(src_j * K(i,j)) / sum(K(i,j))
    # where K(i,j) = (1 + (i-j)^2 / (2 * r * h^2)) ^ -r
    
    out = np.zeros_like(src, dtype=float)
    # Optimize by using a fixed window size W >> h to limit lookback
    W = int(h * 3)
    n = len(src)
    src_val = src.values
    
    # Precompute weights for a window W
    weights = np.zeros(W)
    for i in range(W):
        # distance squared
        d2 = (i)**2
        weights[i] = (1.0 + d2 / (2.0 * r * (h**2))) ** (-r)
    
    for i in range(W, n):
        window_src = src_val[i-W+1:i+1] # Length W
        # Dot product of reversed weights and window
        yhat = np.sum(window_src * weights[::-1]) / np.sum(weights)
        out[i] = yhat
        
    out[:W] = np.nan
    return pd.Series(out, index=src.index)

def gaussian_kernel(src, h, x):
    """
    Gaussian Kernel
    K(i,j) = exp(-(i-j)^2 / (2 * h^2))
    """
    out = np.zeros_like(src, dtype=float)
    W = int(h * 3)
    n = len(src)
    src_val = src.values
    
    weights = np.zeros(W)
    for i in range(W):
        d2 = (i)**2
        weights[i] = np.exp(-d2 / (2.0 * (h**2)))
        
    for i in range(W, n):
        window_src = src_val[i-W+1:i+1]
        yhat = np.sum(window_src * weights[::-1]) / np.sum(weights)
        out[i] = yhat
        
    out[:W] = np.nan
    return pd.Series(out, index=src.index)
