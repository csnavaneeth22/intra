"""
Generate synthetic stock data for testing parameter optimization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_stock_data(ticker, days=30, start_price=1000):
    """
    Generate realistic intraday 5-minute stock data with trends and volatility
    """
    # 5-minute intervals: 75 bars per day (9:15 AM to 3:30 PM IST = 6.25 hours = 75 bars)
    bars_per_day = 75
    total_bars = days * bars_per_day

    # Start date
    start_date = datetime.now() - timedelta(days=days)

    # Generate timestamps (skip weekends)
    timestamps = []
    current_date = start_date
    while len(timestamps) < total_bars:
        if current_date.weekday() < 5:  # Monday to Friday
            for i in range(bars_per_day):
                time = current_date.replace(hour=9, minute=15) + timedelta(minutes=i*5)
                timestamps.append(time)
                if len(timestamps) >= total_bars:
                    break
        current_date += timedelta(days=1)

    timestamps = timestamps[:total_bars]

    # Generate price with trend and mean reversion
    np.random.seed(hash(ticker) % 2**32)

    # Random walk with drift and volatility
    returns = np.random.normal(0.0001, 0.003, total_bars)

    # Add trend component
    trend = np.sin(np.linspace(0, 4*np.pi, total_bars)) * 0.002
    returns += trend

    # Add intraday patterns (higher volatility at open/close)
    for day in range(days):
        start_idx = day * bars_per_day
        end_idx = start_idx + bars_per_day
        if end_idx > total_bars:
            end_idx = total_bars

        # U-shaped volatility (higher at open and close)
        intraday_pattern = np.concatenate([
            np.linspace(1.5, 0.8, bars_per_day//3),
            np.ones(bars_per_day//3) * 0.8,
            np.linspace(0.8, 1.5, bars_per_day - 2*(bars_per_day//3))
        ])
        if end_idx - start_idx < len(intraday_pattern):
            intraday_pattern = intraday_pattern[:end_idx-start_idx]

        returns[start_idx:end_idx] *= intraday_pattern

    # Calculate prices
    price = start_price * np.exp(np.cumsum(returns))

    # Generate OHLC data
    high = price * (1 + np.abs(np.random.normal(0, 0.002, total_bars)))
    low = price * (1 - np.abs(np.random.normal(0, 0.002, total_bars)))
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    close = price

    # Generate volume (higher volume at open and close)
    base_volume = 100000
    volume = np.random.lognormal(np.log(base_volume), 0.5, total_bars)

    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(int)
    }, index=pd.DatetimeIndex(timestamps, name='Datetime'))

    return df


def generate_nifty100_synthetic_data(days=30):
    """Generate synthetic data for all Nifty 100 stocks"""
    from data_fetcher import NIFTY_100

    data_dict = {}

    # Use different base prices for variety
    base_prices = np.random.uniform(500, 3000, len(NIFTY_100))

    print(f"Generating synthetic data for {len(NIFTY_100)} stocks ({days} days)...")
    for i, ticker in enumerate(NIFTY_100):
        df = generate_synthetic_stock_data(ticker, days=days, start_price=base_prices[i])
        data_dict[ticker] = df
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{len(NIFTY_100)} stocks")

    return data_dict


if __name__ == "__main__":
    # Test
    data_dict = generate_nifty100_synthetic_data(days=30)
    print(f"\nGenerated {len(data_dict)} stocks")
    for ticker, df in list(data_dict.items())[:3]:
        print(f"\n{ticker}: {len(df)} bars, Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
