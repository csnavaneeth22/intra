"""
Example: Using Fyers Data for Backtesting
Demonstrates how to use Fyers API for fetching training data
"""

from data_fetcher import fetch_5m_data, NIFTY_100
from backtest_engine import run_backtest, analyze_all
import pandas as pd


def example_1_simple_fetch():
    """Example 1: Simple data fetch with Fyers"""
    print("="*80)
    print("Example 1: Fetching data with Fyers API")
    print("="*80)

    # Test with a few stocks
    test_stocks = ['HDFCBANK.NS', 'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS']

    # Fetch using Fyers (will use environment variable or pass source='fyers')
    data_dict = fetch_5m_data(tickers=test_stocks, period='30d', source='fyers')

    print(f"\nFetched data for {len(data_dict)} stocks:")
    for ticker, df in data_dict.items():
        print(f"  {ticker}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")


def example_2_backtest_with_fyers():
    """Example 2: Run backtest with Fyers data"""
    print("\n" + "="*80)
    print("Example 2: Backtesting with Fyers data")
    print("="*80)

    # Fetch data using Fyers
    test_stocks = ['HDFCBANK.NS', 'RELIANCE.NS', 'SBIN.NS']
    data_dict = fetch_5m_data(tickers=test_stocks, period='30d', source='fyers')

    # Run backtest
    all_trades = []
    for ticker, df in data_dict.items():
        print(f"\nBacktesting {ticker}...")
        trades = run_backtest(df, ticker, initial_capital=25000, risk_per_trade=0.02)
        all_trades.extend(trades)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        print(f"\n--- Results ---")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Win Rate: {trades_df['Win'].mean() * 100:.2f}%")
        print(f"Total PnL: ₹{trades_df['PnL'].sum():.2f}")
        print(f"Avg PnL per Trade: ₹{trades_df['PnL'].mean():.2f}")
    else:
        print("\nNo trades generated")


def example_3_compare_sources():
    """Example 3: Compare Fyers vs yfinance data"""
    print("\n" + "="*80)
    print("Example 3: Comparing Fyers vs yfinance")
    print("="*80)

    test_ticker = ['HDFCBANK.NS']

    print("\nFetching from yfinance...")
    data_yf = fetch_5m_data(tickers=test_ticker, period='7d', source='yfinance')

    print("\nFetching from Fyers...")
    data_fyers = fetch_5m_data(tickers=test_ticker, period='7d', source='fyers')

    if test_ticker[0] in data_yf and test_ticker[0] in data_fyers:
        df_yf = data_yf[test_ticker[0]]
        df_fyers = data_fyers[test_ticker[0]]

        print(f"\nyfinance: {len(df_yf)} bars")
        print(f"Fyers:    {len(df_fyers)} bars")

        print(f"\nyfinance date range: {df_yf.index[0]} to {df_yf.index[-1]}")
        print(f"Fyers date range:    {df_fyers.index[0]} to {df_fyers.index[-1]}")

        print("\nSample comparison (latest 3 bars):")
        print("\nyfinance:")
        print(df_yf.tail(3))
        print("\nFyers:")
        print(df_fyers.tail(3))


def example_4_direct_fyers_api():
    """Example 4: Using Fyers API directly"""
    print("\n" + "="*80)
    print("Example 4: Direct Fyers API usage")
    print("="*80)

    from fyers_data_fetcher import fetch_5m_data_fyers

    # Fetch specific stocks
    stocks = ['HDFCBANK.NS', 'RELIANCE.NS']

    print(f"Fetching {len(stocks)} stocks with Fyers API...")
    data_dict = fetch_5m_data_fyers(stocks, days=30)

    for ticker, df in data_dict.items():
        print(f"\n{ticker}:")
        print(f"  Total bars: {len(df)}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
        print(f"  Last close: ₹{df['Close'].iloc[-1]:.2f}")


def example_5_env_variable_usage():
    """Example 5: Using environment variable to set data source"""
    print("\n" + "="*80)
    print("Example 5: Environment variable configuration")
    print("="*80)

    import os

    # Set environment variable
    os.environ['DATA_SOURCE'] = 'fyers'

    print("Set DATA_SOURCE=fyers")
    print("Now all calls to fetch_5m_data() will use Fyers by default")

    # Fetch will automatically use Fyers
    data_dict = fetch_5m_data(tickers=['HDFCBANK.NS'], period='7d')

    print(f"\nFetched {len(data_dict)} stocks using configured source")


if __name__ == "__main__":
    print("\n🚀 Fyers Data Integration Examples\n")

    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_simple_fetch()
        elif example_num == '2':
            example_2_backtest_with_fyers()
        elif example_num == '3':
            example_3_compare_sources()
        elif example_num == '4':
            example_4_direct_fyers_api()
        elif example_num == '5':
            example_5_env_variable_usage()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python examples_fyers_usage.py [1|2|3|4|5]")
    else:
        print("Available examples:")
        print("  1 - Simple data fetch with Fyers")
        print("  2 - Backtest with Fyers data")
        print("  3 - Compare Fyers vs yfinance")
        print("  4 - Direct Fyers API usage")
        print("  5 - Environment variable configuration")
        print("\nUsage: python examples_fyers_usage.py <number>")
        print("Example: python examples_fyers_usage.py 1")
