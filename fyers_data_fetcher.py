"""
Fyers API Historical Data Fetcher
Fetches historical 5-minute data using Fyers API instead of yfinance
"""
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from fyers_auth import get_fyers_client
from data_fetcher import NIFTY_100


def fetch_fyers_historical_data(fyers, symbol, from_date, to_date, resolution="5"):
    """
    Fetch historical data from Fyers API

    Parameters:
    -----------
    fyers : FyersModel instance
    symbol : str - Stock symbol (e.g., 'HDFCBANK')
    from_date : str - Start date in 'YYYY-MM-DD' format
    to_date : str - End date in 'YYYY-MM-DD' format
    resolution : str - Candle resolution ('1', '5', '15', '30', '60', 'D')

    Returns:
    --------
    DataFrame with OHLCV data
    """
    try:
        # Convert dates to epoch timestamps (Fyers expects Unix timestamps)
        from_timestamp = int(datetime.strptime(from_date, '%Y-%m-%d').timestamp())
        to_timestamp = int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())

        # Fyers symbol format: NSE:SYMBOL-EQ
        fyers_symbol = f"NSE:{symbol}-EQ"

        # Prepare data request
        data = {
            "symbol": fyers_symbol,
            "resolution": resolution,  # 5 for 5-minute candles
            "date_format": "0",  # 0 for Unix timestamp, 1 for date string
            "range_from": from_timestamp,
            "range_to": to_timestamp,
            "cont_flag": "1"  # 1 for continuous data
        }

        # Fetch historical data
        response = fyers.history(data=data)

        if response.get('s') != 'ok':
            print(f"Error fetching {symbol}: {response.get('message', 'Unknown error')}")
            return None

        if 'candles' not in response or not response['candles']:
            print(f"No data returned for {symbol}")
            return None

        # Parse the candles data
        # Fyers returns: [timestamp, open, high, low, close, volume]
        candles = response['candles']

        df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert timestamp to datetime
        df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Datetime', inplace=True)
        df.drop('Timestamp', axis=1, inplace=True)

        # Convert to IST (Fyers returns UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

        return df

    except Exception as e:
        print(f"Exception fetching {symbol}: {e}")
        return None


def fetch_5m_data_fyers(tickers=NIFTY_100, days=30, cache_dir="data_fyers"):
    """
    Fetch 5-minute historical data for multiple stocks using Fyers API

    Parameters:
    -----------
    tickers : list - List of stock symbols (e.g., ['HDFCBANK.NS', 'RELIANCE.NS'])
    days : int - Number of days of historical data to fetch
    cache_dir : str - Directory to cache the data

    Returns:
    --------
    dict - Dictionary mapping ticker to DataFrame
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get Fyers client
    try:
        fyers = get_fyers_client()
    except Exception as e:
        print(f"Failed to initialize Fyers client: {e}")
        print("Make sure you have run fyers_auth.py and have a valid access_token.txt")
        return {}

    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    print(f"Fetching {days} days of 5m data from Fyers API ({from_date_str} to {to_date_str})...")
    print(f"Target: {len(tickers)} stocks")

    data_dict = {}
    success_count = 0

    for i, ticker in enumerate(tickers):
        # Remove .NS suffix if present (yfinance format → Fyers format)
        symbol = ticker.replace('.NS', '')

        cache_file = os.path.join(cache_dir, f"{symbol}_5m_fyers.csv")

        # Check cache freshness (delete if older than 5 minutes to ensure fresh data)
        if os.path.exists(cache_file):
            file_age_minutes = (time.time() - os.path.getmtime(cache_file)) / 60
            if file_age_minutes > 5:
                os.remove(cache_file)
                print(f"[{i+1}/{len(tickers)}] Cache expired for {symbol}, refetching...")
            else:
                # Load from cache
                df = pd.read_csv(cache_file, index_col="Datetime", parse_dates=True)
                if not df.empty:
                    data_dict[ticker] = df
                    success_count += 1
                    if (i + 1) % 10 == 0:
                        print(f"[{i+1}/{len(tickers)}] Loaded from cache: {symbol}")
                    continue

        # Fetch from Fyers API
        df = fetch_fyers_historical_data(fyers, symbol, from_date_str, to_date_str, resolution="5")

        if df is not None and not df.empty:
            # Save to cache
            df.to_csv(cache_file)
            data_dict[ticker] = df
            success_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                print(f"[{i+1}/{len(tickers)}] Fetched: {symbol} ({len(df)} bars)")
        else:
            print(f"[{i+1}/{len(tickers)}] Failed: {symbol}")

        # Rate limiting - be respectful to the API
        if (i + 1) % 10 == 0:
            time.sleep(1)  # Small delay every 10 requests

    print(f"\n✓ Successfully fetched {success_count}/{len(tickers)} stocks")
    print(f"✓ Data saved to {cache_dir}/")

    return data_dict


def test_fyers_data_fetch():
    """Test function to fetch data for a few stocks"""
    print("Testing Fyers historical data fetch...")

    # Test with 5 stocks
    test_tickers = ['HDFCBANK.NS', 'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS']

    data_dict = fetch_5m_data_fyers(test_tickers, days=30)

    if data_dict:
        print("\n--- Sample Data ---")
        for ticker, df in list(data_dict.items())[:2]:
            print(f"\n{ticker}:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
            print(f"  First 3 rows:")
            print(df.head(3))
    else:
        print("No data fetched. Check your Fyers authentication.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode with a few stocks
        test_fyers_data_fetch()
    else:
        # Full fetch for all Nifty 100
        print("Fetching historical data for all Nifty 100 stocks...")
        data_dict = fetch_5m_data_fyers(NIFTY_100, days=30)

        if data_dict:
            print(f"\n✓ Data ready for training with {len(data_dict)} stocks!")
        else:
            print("\n✗ Failed to fetch data. Please check your Fyers credentials.")
