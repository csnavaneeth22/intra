#!/usr/bin/env python3
"""
Lorentzian Classification Backtester with Live Winrate Gate
============================================================
Exact replication of PineScript ML Extensions library logic.

Usage:
    python lorentzian_backtester.py            # full backtest
    python lorentzian_backtester.py --demo     # demo with synthetic data (no Fyers needed)
"""

import os
import sys
import math
import time
import warnings
import argparse
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURATION
# ========================================================================================
STARTING_CAPITAL    = 500_000     # ₹5,00,000 (configurable)
RISK_PER_TRADE      = 0.01        # 1% of portfolio per trade
NEIGHBORS_COUNT     = 8           # KNN k
MAX_BARS_BACK       = 2000        # Lorentzian lookback
WINRATE_THRESHOLD   = 0.70        # 70% gate
MIN_TRADES_FOR_GATE = 20          # minimum trades before gate fires
MAX_TRADES_PER_DAY  = 20          # across all 50 stocks
MAX_SECTOR_TRADES   = 3           # max simultaneous positions per sector
ATR_PERIOD          = 14
ATR_SL_MULT         = 1.5         # Stop-loss: ATR(14) * 1.5
ATR_TP_MULT         = 1.5 * 1.5  # Take-profit: ATR(14) * 2.25  (1.5 R:R)
FORCE_CLOSE_HOUR    = 15
FORCE_CLOSE_MIN     = 20          # 15:20 IST
BACKTEST_DAYS       = 30
DATA_CACHE_DIR      = "data_parquet"
REPORT_DIR          = "backtest_reports"

# ========================================================================================
# NIFTY 50 UNIVERSE
# ========================================================================================
NIFTY_50 = [
    "ADANIENT",  "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO","BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA",      "COALINDIA",  "DIVISLAB",   "DRREDDY",
    "EICHERMOT", "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",
    "HEROMOTOCO","HINDALCO",   "HINDUNILVR", "ICICIBANK",  "INDUSINDBK",
    "INFY",      "ITC",        "JSWSTEEL",   "KOTAKBANK",  "LT",
    "M&M",       "MARUTI",     "NESTLEIND",  "NTPC",       "ONGC",
    "POWERGRID", "RELIANCE",   "SBILIFE",    "SBIN",       "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",  "TCS",
    "TECHM",     "TITAN",      "ULTRACEMCO", "WIPRO",
    "ETERNAL",   # Formerly Zomato Ltd., renamed to Eternal Ltd. (NSE: ETERNAL) in 2025
]

NIFTY_50_SECTORS: Dict[str, str] = {
    "ADANIENT":   "Energy",         "ADANIPORTS":  "Infrastructure",
    "APOLLOHOSP": "Healthcare",     "ASIANPAINT":  "Consumer",
    "AXISBANK":   "Banking",        "BAJAJ-AUTO":  "Auto",
    "BAJFINANCE": "NBFC",           "BAJAJFINSV":  "NBFC",
    "BHARTIARTL": "Telecom",        "BPCL":        "Energy",
    "BRITANNIA":  "FMCG",           "CIPLA":       "Pharma",
    "COALINDIA":  "Mining",         "DIVISLAB":    "Pharma",
    "DRREDDY":    "Pharma",         "EICHERMOT":   "Auto",
    "GRASIM":     "Cement",         "HCLTECH":     "IT",
    "HDFCBANK":   "Banking",        "HDFCLIFE":    "Insurance",
    "HEROMOTOCO": "Auto",           "HINDALCO":    "Metals",
    "HINDUNILVR": "FMCG",           "ICICIBANK":   "Banking",
    "INDUSINDBK": "Banking",        "INFY":        "IT",
    "ITC":        "FMCG",           "JSWSTEEL":    "Metals",
    "KOTAKBANK":  "Banking",        "LT":          "Infrastructure",
    "M&M":        "Auto",           "MARUTI":      "Auto",
    "NESTLEIND":  "FMCG",           "NTPC":        "Power",
    "ONGC":       "Energy",         "POWERGRID":   "Power",
    "RELIANCE":   "Energy",         "SBILIFE":     "Insurance",
    "SBIN":       "Banking",        "SHRIRAMFIN":  "NBFC",
    "SUNPHARMA":  "Pharma",         "TATACONSUM":  "FMCG",
    "TATAMOTORS": "Auto",           "TATASTEEL":   "Metals",
    "TCS":        "IT",             "TECHM":       "IT",
    "TITAN":      "Consumer",       "ULTRACEMCO":  "Cement",
    "WIPRO":      "IT",             "ETERNAL":     "Consumer",    # Formerly Zomato Ltd. (renamed 2025)
}

# ========================================================================================
# DATA FETCHING
# ========================================================================================

def _fetch_single(symbol: str, from_str: str, to_str: str, fyers) -> Optional[pd.DataFrame]:
    """Fetch 5-minute OHLCV data for one symbol from Fyers API."""
    try:
        from_ts = int(datetime.strptime(from_str, '%Y-%m-%d').timestamp())
        to_ts   = int(datetime.strptime(to_str,   '%Y-%m-%d').timestamp())
        data = {
            "symbol":     f"NSE:{symbol}-EQ",
            "resolution": "5",
            "date_format":"0",
            "range_from": str(from_ts),
            "range_to":   str(to_ts),
            "cont_flag":  "1",
        }
        resp = fyers.history(data=data)
        if resp.get('s') != 'ok' or not resp.get('candles'):
            return None
        df = pd.DataFrame(
            resp['candles'],
            columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'],
        )
        df['Datetime'] = pd.to_datetime(df['ts'], unit='s')
        df.set_index('Datetime', inplace=True)
        df.drop('ts', axis=1, inplace=True)
        # UTC → IST, strip tz
        if df.index.tz is None:
            df.index = (
                df.index
                .tz_localize('UTC')
                .tz_convert('Asia/Kolkata')
                .tz_localize(None)
            )
        # Keep only NSE market hours
        df = df.between_time('09:15', '15:30')
        df.sort_index(inplace=True)
        return df if not df.empty else None
    except Exception as e:
        print(f"  [!] {symbol}: {e}")
        return None


def fetch_all_data(
    days: int = BACKTEST_DAYS,
    cache_dir: str = DATA_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch 5-minute data for all Nifty 50 stocks.
    Uses Fyers API with parquet local cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    to_dt   = datetime.now()
    from_dt = to_dt - timedelta(days=days + 7)   # buffer for weekends / holidays
    from_str = from_dt.strftime('%Y-%m-%d')
    to_str   = to_dt.strftime('%Y-%m-%d')

    data_dict: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []

    # Load from parquet cache if fresh (< 6 hours)
    for sym in NIFTY_50:
        path = os.path.join(cache_dir, f"{sym}.parquet")
        if os.path.exists(path):
            age_h = (time.time() - os.path.getmtime(path)) / 3600
            if age_h < 6:
                try:
                    df = pd.read_parquet(path)
                    data_dict[sym] = df
                    continue
                except Exception:
                    pass
        to_fetch.append(sym)

    if to_fetch:
        print(f"Fetching {len(to_fetch)} stocks from Fyers API ({from_str} → {to_str})…")
        try:
            from fyers_auth import get_fyers_client
            fyers = get_fyers_client()
        except Exception as e:
            print(f"[!] Fyers auth failed: {e}")
            print("    Using cached / synthetic data only.")
            return data_dict

        def _worker(sym):
            return sym, _fetch_single(sym, from_str, to_str, fyers)

        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = {exe.submit(_worker, sym): sym for sym in to_fetch}
            done = 0
            for future in as_completed(futures):
                sym, df = future.result()
                done += 1
                if df is not None:
                    path = os.path.join(cache_dir, f"{sym}.parquet")
                    df.to_parquet(path)
                    data_dict[sym] = df
                    print(f"  [{done:2d}/{len(to_fetch)}] ✓ {sym} ({len(df)} bars)")
                else:
                    print(f"  [{done:2d}/{len(to_fetch)}] ✗ {sym}")

    print(f"✓ Data ready for {len(data_dict)}/{len(NIFTY_50)} stocks\n")
    return data_dict


# ========================================================================================
# INDICATORS
# ========================================================================================

def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder ATR via EWM (alpha = 1/period)."""
    return _tr(high, low, close).ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _wt(high: pd.Series, low: pd.Series, close: pd.Series, n1: int = 10, n2: int = 11) -> pd.Series:
    hlc3 = (high + low + close) / 3.0
    esa  = hlc3.ewm(span=n1, adjust=False).mean()
    d    = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci   = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    return ci.ewm(span=n2, adjust=False).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp  = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_s    = _tr(high, low, close).ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


# ========================================================================================
# FEATURE ENGINEERING  (exact PineScript normalization)
# ========================================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with 5 normalized features per bar:
      f1  n_rsi14  = (RSI(close,14) − 50) / 50        → [−1, +1]
      f2  n_wt     = WaveTrend(hlc3,10,11) / 100       → approx [−1, +1]
      f3  n_cci    = clip(CCI(close,20) / 200, −1, +1)
      f4  n_adx    = ADX(20) / 100                     → [0, 1]
      f5  n_rsi9   = (RSI(close,9) − 50) / 50          → [−1, +1]
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    f1 = (_rsi(close, 14) - 50.0) / 50.0
    f2 = _wt(high, low, close, 10, 11) / 100.0
    f3 = (_cci(high, low, close, 20) / 200.0).clip(-1.0, 1.0)
    f4 = _adx(high, low, close, 20) / 100.0
    f5 = (_rsi(close, 9) - 50.0) / 50.0
    return pd.DataFrame({'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}, index=df.index)


# ========================================================================================
# LORENTZIAN KNN  (exact ANN replication of PineScript)
# ========================================================================================

def _knn_scan(distances: np.ndarray, labels: np.ndarray, k: int) -> int:
    """
    Exact PineScript KNN ANN search.

    Iterates history bars (most-recent-first, every-4th-skipped already applied).
    Maintains a growing list of neighbours gated by a rising last_distance threshold.
    When list exceeds k, pop the oldest entry and reset threshold to distances[cutoff].

    Returns: sum of the k (or fewer) selected labels — prediction score.
    """
    last_d = -1.0
    d_buf: List[float] = []
    p_buf: List[int]   = []
    cutoff = round(k * 3 / 4)   # = 6 for k=8

    for d, lbl in zip(distances, labels):
        if d >= last_d:
            last_d = d
            d_buf.append(d)
            p_buf.append(int(lbl))
            if len(p_buf) > k:
                last_d = d_buf[cutoff]
                del d_buf[0]
                del p_buf[0]

    return sum(p_buf)


def compute_lorentzian_predictions(
    features_arr: np.ndarray,   # shape (n, 5)
    close_arr:    np.ndarray,   # shape (n,)
    k:            int = NEIGHBORS_COUNT,
    max_bars_back:int = MAX_BARS_BACK,
) -> np.ndarray:
    """
    Compute Lorentzian KNN prediction for every bar.

    Training label (per PineScript definition):
        y[j] = +1  if close[j] > close[j-4]   (price rose over last 4 bars)
        y[j] = −1  if close[j] < close[j-4]
        y[j] =  0  if equal

    ANN loop: scan offsets 1 … max_bars_back-1, skip offset when offset % 4 == 0
    (matches Pine's `if i % 4` truthy check — truthy = nonzero → skip when 0).
    """
    n = len(features_arr)

    # Training labels — look BACKWARD 4 bars (not forward)
    y_train = np.zeros(n, dtype=np.int8)
    y_train[4:] = np.sign(close_arr[4:] - close_arr[:-4]).astype(np.int8)

    predictions = np.zeros(n, dtype=np.float32)

    warmup = max(50, k * 2 + 10)

    for i in range(warmup, n):
        start = max(0, i - max_bars_back)

        # Valid offsets: 1 … (i-start), dropping multiples of 4
        max_off  = i - start
        offsets  = np.arange(1, max_off + 1)
        offsets  = offsets[offsets % 4 != 0]          # skip every 4th (Pine logic)

        if len(offsets) == 0:
            continue

        hist_idx = i - offsets                         # absolute indices in the array
        diff     = np.abs(features_arr[hist_idx] - features_arr[i])  # (N, 5)
        dists    = np.sum(np.log1p(diff), axis=1)      # (N,) Lorentzian distance

        predictions[i] = _knn_scan(dists, y_train[hist_idx], k)

    return predictions


# ========================================================================================
# FILTERS
# ========================================================================================

def compute_volatility_filter(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Replicates ml.filter_volatility(minLength=1, maxLength=10).
    Pass when ATR(1) < ATR(10) * 1.2
    """
    tr    = _tr(high, low, close)
    atr1  = tr.ewm(alpha=1.0,     adjust=False).mean()   # alpha=1 → just TR
    atr10 = tr.ewm(alpha=1.0/10,  adjust=False).mean()
    return (atr1 < atr10 * 1.2).fillna(False)


def compute_regime_filter(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    threshold: float = -0.1,
) -> pd.Series:
    """
    Replicates ml.regime_filter(ohlc4, threshold=−0.1).
    Uses Ehlers' Kaufman-Adaptive slope on ohlc4.

    Recursive state:
        value1[i] = 0.2*(src[i]-src[i-1]) + 0.8*value1[i-1]
        value2[i] = 0.1*TR[i] + 0.9*value2[i-1]
        omega     = |value1 / value2|
        alpha     = (-omega² + sqrt(omega⁴ + 16·omega²)) / 8
        klmh[i]   = alpha*src[i] + (1-alpha)*klmh[i-1]
        regime    = klmh[i] − klmh[i-2]
    Pass when regime > threshold.
    """
    src    = (open_ + high + low + close) / 4.0    # ohlc4
    tr_arr = _tr(high, low, close).values
    src_v  = src.values
    n      = len(src_v)

    value1 = np.zeros(n)
    value2 = np.zeros(n)
    klmh   = np.zeros(n)
    klmh[0] = src_v[0]

    for i in range(1, n):
        value1[i] = 0.2 * (src_v[i] - src_v[i - 1]) + 0.8 * value1[i - 1]
        v2_raw    = 0.1 * tr_arr[i] + 0.9 * value2[i - 1]
        value2[i] = v2_raw
        denom     = v2_raw if v2_raw != 0.0 else 1e-10
        omega     = abs(value1[i] / denom)
        # Solving: alpha = (-omega² + sqrt(omega⁴ + 16·omega²)) / 8
        o2        = omega * omega
        alpha     = (-o2 + math.sqrt(o2 * o2 + 16.0 * o2)) / 8.0
        alpha     = max(0.0, min(1.0, alpha))
        klmh[i]   = alpha * src_v[i] + (1.0 - alpha) * klmh[i - 1]

    klmh_s      = pd.Series(klmh, index=src.index)
    regime_slope = klmh_s - klmh_s.shift(2)
    return (regime_slope > threshold).fillna(False)


# Precompute rational-quadratic kernel weights once
_RQ_H, _RQ_R, _RQ_X = 8, 8.0, 25
_RQ_WEIGHTS = np.array(
    [(1.0 + (i * i) / (2.0 * _RQ_R * _RQ_H * _RQ_H)) ** (-_RQ_R) for i in range(_RQ_X)]
)
_RQ_WEIGHTS_SUM = _RQ_WEIGHTS.sum()


def compute_kernel_estimates(close: pd.Series) -> pd.Series:
    """
    Rational Quadratic Kernel regression (Nadaraya-Watson), h=8, r=8, x=25.
    yhat1[i] = Σ w[j] * close[i-j]  /  Σ w[j]   for j=0..24
    """
    n       = len(close)
    arr     = close.values
    yhat    = np.full(n, np.nan)
    x       = _RQ_X
    weights = _RQ_WEIGHTS
    wsum    = _RQ_WEIGHTS_SUM

    for i in range(x - 1, n):
        # window[0]=close[i], window[1]=close[i-1], ...
        window  = arr[i - x + 1: i + 1][::-1]
        yhat[i] = np.dot(weights, window) / wsum

    return pd.Series(yhat, index=close.index)


# ========================================================================================
# SIGNAL COMPUTATION
# ========================================================================================

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline for one stock: features → KNN → filters → signals.
    Returns enriched DataFrame with columns:
        prediction, signal, startLongTrade, startShortTrade,
        is_bullish, is_bearish, atr14, yhat1
    """
    df = df.copy()

    if len(df) < 100:
        for col in ('prediction', 'signal', 'atr14', 'yhat1'):
            df[col] = np.nan
        df['startLongTrade']  = False
        df['startShortTrade'] = False
        df['is_bullish']      = False
        df['is_bearish']      = False
        return df

    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    open_  = df['Open']

    # 1. Features
    feats     = compute_features(df).fillna(0.0)
    feats_arr = feats.values  # (n, 5)

    # 2. Lorentzian KNN predictions
    pred_arr      = compute_lorentzian_predictions(feats_arr, close.values)
    df['prediction'] = pred_arr

    # 3. Filters
    df['filter_volatility'] = compute_volatility_filter(high, low, close)
    df['filter_regime']     = compute_regime_filter(open_, high, low, close)

    yhat1          = compute_kernel_estimates(close)
    yhat1_prev     = yhat1.shift(1)
    df['yhat1']    = yhat1
    df['is_bullish'] = (yhat1 > yhat1_prev).fillna(False)
    df['is_bearish'] = (yhat1 < yhat1_prev).fillna(False)

    # ADX / EMA / SMA filters: disabled → always pass
    df['all_filters'] = df['filter_volatility'] & df['filter_regime']

    # 4. Raw signal from prediction
    raw_signal = pd.Series(0, index=df.index)
    raw_signal[df['prediction'] > 0] =  1
    raw_signal[df['prediction'] < 0] = -1

    # Hold previous signal when prediction is 0 (sticky)
    df['signal'] = raw_signal.replace(0, np.nan).ffill().fillna(0).astype(int)

    df['signal_changed'] = df['signal'] != df['signal'].shift(1)

    # 5. Entry conditions
    df['startLongTrade'] = (
        (df['signal'] == 1)
        & df['signal_changed']
        & df['is_bullish']
        & df['all_filters']
    )
    df['startShortTrade'] = (
        (df['signal'] == -1)
        & df['signal_changed']
        & df['is_bearish']
        & df['all_filters']
    )

    # 6. ATR for position sizing and SL/TP
    df['atr14'] = _wilder_atr(high, low, close, ATR_PERIOD)

    return df


# ========================================================================================
# LIVE WINRATE TRACKER  (per-stock, replicates PineScript ml.backtest())
# ========================================================================================

class LiveWinrateTracker:
    """
    Tracks cumulative win/loss based on 4-bar-ahead closes.
    Matches PineScript ml.backtest() bar-by-bar tracking exactly.
    Returns 0.0 until at least min_trades completed trades.
    """

    def __init__(self, min_trades: int = MIN_TRADES_FOR_GATE):
        self.wins:       int = 0
        self.losses:     int = 0
        self.min_trades: int = min_trades
        # (bar_index, entry_close, direction)
        self.pending: List[Tuple[int, float, int]] = []

    def log_entry(self, bar_index: int, close: float, direction: int) -> None:
        self.pending.append((bar_index, close, direction))

    def update(self, bar_index: int, current_close: float) -> None:
        """Resolve trades that are now ≥ 4 bars old."""
        remaining = []
        for (entry_bar, entry_close, direction) in self.pending:
            if bar_index - entry_bar >= 4:
                if direction == 1 and current_close > entry_close:
                    self.wins += 1
                elif direction == -1 and current_close < entry_close:
                    self.wins += 1
                else:
                    self.losses += 1
            else:
                remaining.append((entry_bar, entry_close, direction))
        self.pending = remaining

    def winrate(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 1.0 if self.min_trades == 0 else 0.0
        return (self.wins / total) if total >= self.min_trades else 0.0


# ========================================================================================
# BROKERAGE & TAX CALCULATION  (Fyers intraday equity, all 6 components)
# ========================================================================================

def calculate_costs(
    entry_price: float,
    exit_price:  float,
    shares:      int,
    direction:   int,   # +1 long, −1 short
) -> float:
    """
    Round-trip intraday equity transaction cost.

    For Long  (direction=+1): entry = BUY, exit = SELL
    For Short (direction=−1): entry = SELL, exit = BUY

    Components:
      Brokerage  : min(turnover*0.03%, ₹20) per leg
      STT        : 0.025% on SELL leg turnover
      ETC        : 0.00297% on total turnover (both legs)
      SEBI       : 0.0001% on total turnover
      GST        : 18% on (brokerage + ETC + SEBI)
      Stamp      : 0.003% on BUY leg turnover
    """
    entry_to = entry_price * shares
    exit_to  = exit_price  * shares
    total_to = entry_to + exit_to

    if direction == 1:          # long: buy entry, sell exit
        buy_to  = entry_to
        sell_to = exit_to
    else:                       # short: sell entry, buy exit
        sell_to = entry_to
        buy_to  = exit_to

    brokerage = (min(entry_to * 0.0003, 20.0) +
                 min(exit_to  * 0.0003, 20.0))
    stt       = sell_to  * 0.00025
    etc       = total_to * 0.0000297
    sebi      = total_to * 0.000001
    gst       = (brokerage + etc + sebi) * 0.18
    stamp     = buy_to   * 0.00003

    return brokerage + stt + etc + sebi + gst + stamp


# ========================================================================================
# TRADE DATA CLASS
# ========================================================================================

@dataclass
class Trade:
    ticker:           str
    direction:        int          # +1 long, −1 short
    entry_time:       pd.Timestamp
    entry_price:      float
    shares:           int
    sl:               float
    tp:               float
    bars_held:        int = 0
    exit_time:        Optional[pd.Timestamp] = None
    exit_price:       Optional[float] = None
    gross_pnl:        float = 0.0
    costs:            float = 0.0
    net_pnl:          float = 0.0
    win:              bool  = False
    exit_reason:      str   = ""
    winrate_at_entry: float = 0.0
    prediction_score: int   = 0


# ========================================================================================
# PORTFOLIO BACKTESTING ENGINE
# ========================================================================================

def _close_trade(
    trade: Trade,
    exit_time:   pd.Timestamp,
    exit_price:  float,
    exit_reason: str,
) -> float:
    """Fill exit fields and return net PnL (updates capital)."""
    trade.exit_time   = exit_time
    trade.exit_price  = exit_price
    trade.exit_reason = exit_reason
    trade.gross_pnl   = (exit_price - trade.entry_price) * trade.shares * trade.direction
    trade.costs       = calculate_costs(trade.entry_price, exit_price, trade.shares, trade.direction)
    trade.net_pnl     = trade.gross_pnl - trade.costs
    trade.win         = trade.net_pnl > 0.0
    return trade.net_pnl


def run_portfolio_backtest(
    data_dict:          Dict[str, pd.DataFrame],
    starting_capital:   float = STARTING_CAPITAL,
    winrate_threshold:  float = WINRATE_THRESHOLD,
    min_trades_for_gate:int   = MIN_TRADES_FOR_GATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Main bar-by-bar portfolio backtest.

    Returns (trade_log_df, daily_summary_df, overall_stats_dict).
    """
    # ── Pass 1: compute signals for all stocks ──────────────────────────────
    print("=== Pass 1: computing signals ===")
    signals: Dict[str, pd.DataFrame] = {}
    for sym, df in data_dict.items():
        print(f"  {sym}…", end='\r')
        try:
            signals[sym] = compute_signals(df)
        except Exception as e:
            print(f"  [!] {sym}: {e}")
    print(f"\n✓ Signals ready for {len(signals)} stocks\n")

    # ── Build sorted unified timeline ───────────────────────────────────────
    all_ts = sorted(
        {ts for df in signals.values() for ts in df.index}
    )

    # ── Per-stock winrate trackers ──────────────────────────────────────────
    trackers: Dict[str, LiveWinrateTracker] = {
        sym: LiveWinrateTracker(min_trades=min_trades_for_gate)
        for sym in signals
    }

    # ── State ───────────────────────────────────────────────────────────────
    capital            = starting_capital
    open_trades:       List[Trade] = []
    completed_trades:  List[Trade] = []
    portfolio_values:  List[Tuple[date, float]] = []

    prev_date:     Optional[date] = None
    trades_today:  int            = 0
    sector_counts: Dict[str, int] = {}

    print("=== Pass 2: bar-by-bar simulation ===")
    total_bars = len(all_ts)
    report_interval = max(1, total_bars // 20)

    for bar_num, ts in enumerate(all_ts):
        if bar_num % report_interval == 0:
            pct = bar_num / total_bars * 100
            print(f"  {pct:5.1f}%  ts={ts}  capital=₹{capital:,.0f}  open={len(open_trades)}", end='\r')

        ts_pd   = pd.Timestamp(ts)
        cur_date = ts_pd.date()
        cur_h, cur_m = ts_pd.hour, ts_pd.minute

        # ── New day reset ──────────────────────────────────────────────────
        if cur_date != prev_date:
            if prev_date is not None:
                # snapshot portfolio value at end of previous day
                open_val = sum(t.shares * t.entry_price for t in open_trades)
                portfolio_values.append((prev_date, capital + open_val))
            prev_date    = cur_date
            trades_today = 0
            sector_counts = {}

        # ── Update winrate trackers ────────────────────────────────────────
        for sym, df in signals.items():
            if ts not in df.index:
                continue
            bar_idx = df.index.get_loc(ts)
            cur_close = df.at[ts, 'Close']
            trackers[sym].update(bar_idx, cur_close)

        # ── Force-close flag ───────────────────────────────────────────────
        is_force_close = (cur_h == FORCE_CLOSE_HOUR and cur_m >= FORCE_CLOSE_MIN)

        # ── Process exits ──────────────────────────────────────────────────
        still_open: List[Trade] = []
        for trade in open_trades:
            sym = trade.ticker
            df  = signals.get(sym)
            if df is None or ts not in df.index:
                still_open.append(trade)
                continue

            row   = df.loc[ts]
            hi    = float(row['High'])
            lo    = float(row['Low'])
            close = float(row['Close'])
            trade.bars_held += 1

            exit_p: Optional[float] = None
            reason: str             = ""

            if trade.direction == 1:       # ── Long exit checks
                if lo <= trade.sl:
                    exit_p, reason = trade.sl, "SL"
                elif hi >= trade.tp:
                    exit_p, reason = trade.tp, "TP"
                elif is_force_close:
                    exit_p, reason = close, "ForceClose"
                elif trade.bars_held >= 4:
                    exit_p, reason = close, "4-Bar Hold"
                elif bool(row.get('startShortTrade', False)):
                    exit_p, reason = close, "Signal Flip"
            else:                          # ── Short exit checks
                if hi >= trade.sl:
                    exit_p, reason = trade.sl, "SL"
                elif lo <= trade.tp:
                    exit_p, reason = trade.tp, "TP"
                elif is_force_close:
                    exit_p, reason = close, "ForceClose"
                elif trade.bars_held >= 4:
                    exit_p, reason = close, "4-Bar Hold"
                elif bool(row.get('startLongTrade', False)):
                    exit_p, reason = close, "Signal Flip"

            if exit_p is not None:
                net_pnl = _close_trade(trade, ts_pd, exit_p, reason)
                capital += net_pnl
                # Release sector slot
                sect = NIFTY_50_SECTORS.get(sym, 'Other')
                sector_counts[sect] = max(0, sector_counts.get(sect, 0) - 1)
                completed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── Collect entry signals at this bar ─────────────────────────────
        # Step 1: log ALL signal firings into winrate trackers (gate-agnostic),
        #         matching PineScript ml.backtest() unconditional tracking.
        # Step 2: apply gate, sector, position limits for actual execution.

        near_close = (cur_h == 15 and cur_m >= 10) or (cur_h > 15)

        candidates = []
        for sym, df in signals.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]

            direction = 0
            if bool(row.get('startLongTrade', False)):
                direction = 1
            elif bool(row.get('startShortTrade', False)):
                direction = -1
            if direction == 0:
                continue

            # Always log signal in winrate tracker (builds historical win/loss record)
            bar_idx = df.index.get_loc(ts)
            trackers[sym].log_entry(bar_idx, float(row['Close']), direction)

            # Do not open new positions near market close
            if near_close:
                continue

            # Winrate gate (only fires once ≥ min_trades_for_gate completed)
            wr = trackers[sym].winrate()
            if wr < winrate_threshold:
                continue

            # No existing position in this stock
            if any(t.ticker == sym for t in open_trades):
                continue

            atr14 = float(row.get('atr14', 0) or 0)
            if atr14 <= 0 or math.isnan(atr14):
                continue

            candidates.append({
                'sym':        sym,
                'direction':  direction,
                'close':      float(row['Close']),
                'atr14':      atr14,
                'prediction': int(row.get('prediction', 0) or 0),
                'winrate':    wr,
            })

        # Rank by |prediction strength|, take top MAX_TRADES_PER_DAY
        candidates.sort(key=lambda x: abs(x['prediction']), reverse=True)

        for cand in candidates:
            if trades_today >= MAX_TRADES_PER_DAY:
                break

            sym  = cand['sym']
            sect = NIFTY_50_SECTORS.get(sym, 'Other')

            if sector_counts.get(sect, 0) >= MAX_SECTOR_TRADES:
                continue

            entry_price   = cand['close']
            atr14         = cand['atr14']
            sl_dist       = atr14 * ATR_SL_MULT
            capital_at_risk = capital * RISK_PER_TRADE
            shares        = int(capital_at_risk / sl_dist)

            if shares <= 0:
                continue

            direction = cand['direction']
            if direction == 1:
                sl = entry_price - sl_dist
                tp = entry_price + atr14 * ATR_TP_MULT
            else:
                sl = entry_price + sl_dist
                tp = entry_price - atr14 * ATR_TP_MULT

            trade = Trade(
                ticker           = sym,
                direction        = direction,
                entry_time       = ts_pd,
                entry_price      = entry_price,
                shares           = shares,
                sl               = sl,
                tp               = tp,
                winrate_at_entry = cand['winrate'],
                prediction_score = cand['prediction'],
            )
            open_trades.append(trade)
            trades_today  += 1
            sector_counts[sect] = sector_counts.get(sect, 0) + 1

    # ── Close any trades still open at end of simulation ───────────────────
    for trade in open_trades:
        sym = trade.ticker
        df  = signals.get(sym)
        last_close = float(df['Close'].iloc[-1]) if df is not None else trade.entry_price
        last_ts    = df.index[-1] if df is not None else trade.entry_time
        net_pnl    = _close_trade(trade, last_ts, last_close, "EndOfBacktest")
        capital   += net_pnl
        completed_trades.append(trade)

    # Final day snapshot
    if prev_date is not None:
        portfolio_values.append((prev_date, capital))

    print(f"\n✓ Simulation complete.  Trades: {len(completed_trades)},  Final capital: ₹{capital:,.2f}\n")

    # ── Build trade-log DataFrame ───────────────────────────────────────────
    rows = []
    for t in completed_trades:
        rows.append({
            'Date':              t.entry_time.date(),
            'Stock':             t.ticker,
            'Direction':         'Long' if t.direction == 1 else 'Short',
            'Entry Time':        t.entry_time,
            'Entry Price':       round(t.entry_price, 2),
            'Exit Time':         t.exit_time,
            'Exit Price':        round(t.exit_price, 2) if t.exit_price else None,
            'Shares':            t.shares,
            'Gross PnL':         round(t.gross_pnl, 2),
            'Total Costs':       round(t.costs, 2),
            'Net PnL':           round(t.net_pnl, 2),
            'Win/Loss':          'Win' if t.win else 'Loss',
            'Winrate at Entry':  round(t.winrate_at_entry, 4),
            'Prediction Score':  t.prediction_score,
            'Exit Reason':       t.exit_reason,
        })
    trade_df = pd.DataFrame(rows)

    # ── Build daily summary DataFrame ──────────────────────────────────────
    pv_df = pd.DataFrame(portfolio_values, columns=['Date', 'Portfolio Value'])

    if not trade_df.empty:
        daily_agg = (
            trade_df.groupby('Date')
            .agg(
                Trades     =('Net PnL',    'count'),
                Wins       =('Win/Loss',   lambda x: (x == 'Win').sum()),
                Losses     =('Win/Loss',   lambda x: (x == 'Loss').sum()),
                Gross_PnL  =('Gross PnL',  'sum'),
                Total_Costs=('Total Costs','sum'),
                Net_PnL    =('Net PnL',    'sum'),
            )
            .reset_index()
        )
        daily_agg['Winrate'] = (
            daily_agg['Wins'] / daily_agg['Trades'].replace(0, np.nan)
        ).round(4)
        daily_df = daily_agg.merge(pv_df, on='Date', how='outer').sort_values('Date')
    else:
        daily_df = pv_df.copy()

    # ── Overall stats ───────────────────────────────────────────────────────
    total_net_pnl   = trade_df['Net PnL'].sum()       if not trade_df.empty else 0.0
    total_costs     = trade_df['Total Costs'].sum()   if not trade_df.empty else 0.0
    total_wins      = (trade_df['Win/Loss'] == 'Win').sum() if not trade_df.empty else 0
    total_trades    = len(trade_df)
    win_rate        = total_wins / total_trades if total_trades > 0 else 0.0

    # Max drawdown (on daily portfolio values)
    pv_series = pd.Series(
        pv_df['Portfolio Value'].values if not pv_df.empty else [starting_capital],
        index=range(len(pv_df)) if not pv_df.empty else [0],
    )
    roll_max = pv_series.cummax()
    drawdowns = (pv_series - roll_max) / roll_max.replace(0, np.nan)
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    # Daily returns for Sharpe
    daily_returns = pv_series.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if daily_returns.std() > 0 else 0.0
    )

    best_day_row  = daily_df.loc[daily_df['Net_PnL'].idxmax()]  if not daily_df.empty and 'Net_PnL' in daily_df.columns else None
    worst_day_row = daily_df.loc[daily_df['Net_PnL'].idxmin()]  if not daily_df.empty and 'Net_PnL' in daily_df.columns else None

    stats = {
        'Starting Capital':           starting_capital,
        'Ending Capital':             capital,
        'Total Net PnL':              round(total_net_pnl, 2),
        'Total Brokerage/Taxes':      round(total_costs, 2),
        'Total Trades':               total_trades,
        'Total Wins':                 total_wins,
        'Total Losses':               total_trades - total_wins,
        'Overall Win Rate':           round(win_rate, 4),
        'Max Drawdown (daily)':       round(max_drawdown, 4),
        'Sharpe Ratio (annualised)':  round(sharpe, 4),
        'Best Day':                   (str(best_day_row['Date'])  if best_day_row  is not None else 'N/A'),
        'Best Day PnL':               (round(float(best_day_row['Net_PnL']), 2)   if best_day_row  is not None else 0.0),
        'Worst Day':                  (str(worst_day_row['Date']) if worst_day_row is not None else 'N/A'),
        'Worst Day PnL':              (round(float(worst_day_row['Net_PnL']), 2)  if worst_day_row is not None else 0.0),
        'Avg Trades per Day':         round(total_trades / max(len(pv_df), 1), 2),
        'Avg Winrate Across Days':    round(daily_df['Winrate'].mean(), 4) if 'Winrate' in daily_df.columns else 0.0,
    }

    return trade_df, daily_df, stats


# ========================================================================================
# REPORT GENERATION
# ========================================================================================

def _fmt_inr(v: float) -> str:
    sign = '-' if v < 0 else ''
    return f"{sign}₹{abs(v):,.2f}"


def generate_report(
    trade_df:  pd.DataFrame,
    daily_df:  pd.DataFrame,
    stats:     dict,
    out_dir:   str = REPORT_DIR,
) -> None:
    """Save CSV files, HTML report and three matplotlib charts."""
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── 1. CSVs ─────────────────────────────────────────────────────────────
    trade_csv = os.path.join(out_dir, f"trades_{ts_tag}.csv")
    daily_csv = os.path.join(out_dir, f"daily_{ts_tag}.csv")
    trade_df.to_csv(trade_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)
    print(f"  Saved: {trade_csv}")
    print(f"  Saved: {daily_csv}")

    # ── 2. Charts ────────────────────────────────────────────────────────────
    fig_path = os.path.join(out_dir, f"charts_{ts_tag}.png")
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    pv_col = 'Portfolio Value'
    if pv_col in daily_df.columns and not daily_df[pv_col].dropna().empty:
        dates_pv = pd.to_datetime(daily_df['Date'])
        pv_vals  = daily_df[pv_col].astype(float)
        ax1.plot(dates_pv, pv_vals, color='steelblue', linewidth=2, label='Portfolio Value')
        ax1.axhline(stats['Starting Capital'], color='grey', linestyle='--', linewidth=1, label='Starting Capital')
        ax1.fill_between(dates_pv, stats['Starting Capital'], pv_vals,
                         where=(pv_vals >= stats['Starting Capital']),
                         alpha=0.2, color='green', label='Profit')
        ax1.fill_between(dates_pv, stats['Starting Capital'], pv_vals,
                         where=(pv_vals < stats['Starting Capital']),
                         alpha=0.2, color='red', label='Loss')
        ax1.set_title('Equity Curve', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.legend(fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No portfolio data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Equity Curve')

    if 'Net_PnL' in daily_df.columns:
        dates_d  = pd.to_datetime(daily_df['Date'])
        net_pnl  = daily_df['Net_PnL'].astype(float)
        colors   = ['green' if v >= 0 else 'red' for v in net_pnl]
        ax2.bar(dates_d, net_pnl, color=colors, width=0.7, alpha=0.85)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Daily Net PnL', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Net PnL (₹)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No daily PnL data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Daily Net PnL')

    if 'Winrate' in daily_df.columns:
        dates_w = pd.to_datetime(daily_df['Date'])
        wr_vals = daily_df['Winrate'].astype(float).fillna(0)
        ax3.plot(dates_w, wr_vals, color='purple', linewidth=2, label='Daily Win Rate')
        ax3.axhline(WINRATE_THRESHOLD, color='orange', linestyle='--', linewidth=1.5,
                    label=f'{WINRATE_THRESHOLD*100:.0f}% Threshold')
        ax3.set_ylim(0, 1.05)
        ax3.set_title('Rolling Win Rate', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Win Rate')
        ax3.legend(fontsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No winrate data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Rolling Win Rate')

    plt.suptitle(
        f"Lorentzian Classification Backtester — {BACKTEST_DAYS}d, "
        f"Nifty 50  |  Net PnL: {_fmt_inr(stats.get('Total Net PnL', 0))}",
        fontsize=14, fontweight='bold',
    )
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # ── 3. HTML report ───────────────────────────────────────────────────────
    html_path = os.path.join(out_dir, f"report_{ts_tag}.html")
    _write_html_report(html_path, trade_df, daily_df, stats, fig_path, ts_tag)
    print(f"  Saved: {html_path}")


def _write_html_report(
    path: str,
    trade_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    stats:    dict,
    fig_path: str,
    ts_tag:   str,
) -> None:
    # Summary table rows
    stat_rows = ''.join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in stats.items()
    )

    # Per-day table
    if not daily_df.empty:
        daily_html = daily_df.to_html(index=False, classes='data-table', border=0, na_rep='—')
    else:
        daily_html = "<p>No daily data.</p>"

    # Per-trade table (last 200 to keep HTML manageable)
    if not trade_df.empty:
        trade_html = (
            trade_df.tail(200)
            .to_html(index=False, classes='data-table', border=0, na_rep='—')
        )
        trade_note = (
            f"<p><em>Showing last 200 of {len(trade_df)} trades. "
            "See CSV for full log.</em></p>"
        )
    else:
        trade_html = "<p>No trades executed.</p>"
        trade_note = ""

    img_src = os.path.basename(fig_path)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Lorentzian Backtest Report — {ts_tag}</title>
<style>
  body   {{ font-family: Arial, sans-serif; margin: 30px; background: #f9f9f9; color: #222; }}
  h1     {{ color: #2c3e50; }}
  h2     {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 6px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.data-table th {{ background: #2c3e50; color: #fff; padding: 8px; text-align: left; }}
  table.data-table td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
  table.data-table tr:nth-child(even) {{ background: #f2f2f2; }}
  .summary-table td, .summary-table th {{ padding: 7px 14px; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
  .win  {{ color: green; font-weight: bold; }}
  .loss {{ color: red;   font-weight: bold; }}
</style>
</head>
<body>
<h1>&#128200; Lorentzian Classification Backtest Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>Overall Summary</h2>
<table class="summary-table data-table">
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{stat_rows}</tbody>
</table>

<h2>Charts</h2>
<img src="{img_src}" alt="Backtest Charts">

<h2>Daily Summary</h2>
{daily_html}

<h2>Trade Log</h2>
{trade_note}
{trade_html}
</body>
</html>"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ========================================================================================
# SYNTHETIC DATA  (demo mode — no Fyers auth needed)
# ========================================================================================

def _make_synthetic_df(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """
    Generate realistic 5-minute OHLCV bars for a single stock.
    Produces exactly n_days trading days of NSE market-hours data
    (09:15 – 15:25, 75 bars/day = n_days * 75 bars total).

    Uses GBM with mild trending to increase the chance that the
    Lorentzian strategy accumulates a reasonable winrate.
    """
    rng  = np.random.default_rng(seed)

    # Build timestamps: n_days business days × 75 5-min bars (09:15–15:25)
    trading_days = pd.bdate_range('2024-11-01', periods=n_days)
    times_list   = []
    for d in trading_days:
        bars = pd.date_range(
            start=f"{d.date()} 09:15",
            end=f"{d.date()} 15:25",
            freq='5min',
        )
        times_list.extend(bars)
    times = pd.DatetimeIndex(times_list)
    n     = len(times)

    # GBM with mild trend: slight drift so price series has detectable direction
    drift  = rng.choice([-1, 1]) * 0.00008   # mild directional drift
    log_ret = rng.normal(drift, 0.0018, n)

    # Inject a few regime changes for realism
    regime_change_bars = rng.choice(n, size=max(1, n // 300), replace=False)
    for rc in regime_change_bars:
        log_ret[rc:] += rng.choice([-1, 1]) * 0.00015

    close  = np.exp(np.cumsum(log_ret)) * rng.uniform(200.0, 3000.0)
    spread = close * rng.uniform(0.001, 0.003, n)
    high   = close + spread * rng.uniform(0.4, 1.0, n)
    low    = close - spread * rng.uniform(0.4, 1.0, n)
    open_  = np.roll(close, 1); open_[0] = close[0]   # open = previous close
    volume = rng.integers(10_000, 800_000, n).astype(float)

    df = pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=times,
    )
    df.index.name = 'Datetime'
    return df


def load_demo_data(n_stocks: int = 10, n_days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Return synthetic data for n_stocks Nifty-50 symbols.
    Used when --demo flag is passed or Fyers auth is unavailable.
    """
    print(f"[DEMO] Generating {n_days}-day synthetic data for {n_stocks} stocks…")
    data: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(NIFTY_50[:n_stocks]):
        data[sym] = _make_synthetic_df(n_days=n_days, seed=i)
        print(f"  {sym}: {len(data[sym])} bars  ({n_days} trading days)")
    print()
    return data


# ========================================================================================
# ENTRY POINT
# ========================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Lorentzian Classification Backtester with Live Winrate Gate'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run with synthetic data (no Fyers auth required)',
    )
    parser.add_argument(
        '--capital', type=float, default=STARTING_CAPITAL,
        help=f'Starting capital in INR (default: {STARTING_CAPITAL:,.0f})',
    )
    parser.add_argument(
        '--days', type=int, default=BACKTEST_DAYS,
        help=f'Backtest period in calendar days (default: {BACKTEST_DAYS})',
    )
    parser.add_argument(
        '--stocks', type=int, default=len(NIFTY_50),
        help='Number of Nifty-50 stocks to include (default: all 50)',
    )
    parser.add_argument(
        '--min-winrate', type=float, default=None,
        help=(
            'Override winrate gate threshold (0.0–1.0). '
            'Use 0.0 to disable the gate entirely for testing. '
            f'Default: {WINRATE_THRESHOLD}'
        ),
    )
    args = parser.parse_args()

    # Apply optional winrate override (module-level, affects gate checks)
    gate_threshold  = args.min_winrate if args.min_winrate is not None else WINRATE_THRESHOLD
    gate_min_trades = MIN_TRADES_FOR_GATE if gate_threshold > 0.0 else 0

    print("=" * 70)
    print("  LORENTZIAN CLASSIFICATION BACKTESTER  |  Live Winrate Gate")
    print("=" * 70)
    print(f"  Capital       : ₹{args.capital:,.0f}")
    print(f"  Days          : {args.days}")
    print(f"  Universe      : {args.stocks} stocks")
    print(f"  Winrate gate  : ≥ {gate_threshold*100:.0f}%  (min {gate_min_trades} trades)")
    print(f"  Max/day       : {MAX_TRADES_PER_DAY} trades  |  Max/sector: {MAX_SECTOR_TRADES}")
    if args.demo:
        print("  Mode          : DEMO (synthetic data)")
    print("=" * 70)
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    if args.demo:
        data_dict = load_demo_data(
            n_stocks=min(args.stocks, len(NIFTY_50)),
            n_days=args.days,
        )
    else:
        full_data = fetch_all_data(days=args.days)
        # Restrict to requested number of stocks
        syms      = list(full_data.keys())[:args.stocks]
        data_dict = {s: full_data[s] for s in syms}

    if not data_dict:
        print("[ERROR] No data available. Try --demo or check Fyers credentials.")
        sys.exit(1)

    # ── Run backtest ───────────────────────────────────────────────────────
    t0 = time.time()
    trade_df, daily_df, stats = run_portfolio_backtest(
        data_dict,
        starting_capital    = args.capital,
        winrate_threshold   = gate_threshold,
        min_trades_for_gate = gate_min_trades,
    )
    elapsed = time.time() - t0

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)
    for k, v in stats.items():
        if isinstance(v, float):
            if 'Capital' in k or 'PnL' in k or 'Taxes' in k:
                print(f"  {k:<35s}: {_fmt_inr(v)}")
            elif 'Rate' in k or 'Winrate' in k or 'Drawdown' in k:
                print(f"  {k:<35s}: {v*100:.2f}%")
            else:
                print(f"  {k:<35s}: {v:.4f}")
        else:
            print(f"  {k:<35s}: {v}")
    print(f"\n  Backtest runtime: {elapsed:.1f}s")
    print("=" * 70)

    # ── Generate report ────────────────────────────────────────────────────
    print("\n=== Generating report ===")
    generate_report(trade_df, daily_df, stats, out_dir=REPORT_DIR)
    print(f"\nAll outputs written to: {REPORT_DIR}/")


if __name__ == '__main__':
    main()
