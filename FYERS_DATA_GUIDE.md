# Using Fyers API for Training Data

This guide explains how to use your Fyers account to fetch historical data for training the Lorentzian ML model instead of using yfinance.

## Why Use Fyers for Training Data?

1. **More Reliable**: Direct broker API access, no rate limits from yfinance
2. **Better Quality**: Exchange-grade data from your broker
3. **Consistent Source**: Same data source for both training and live trading
4. **Faster**: Efficient API designed for trading platforms

## Setup Instructions

### 1. Authenticate with Fyers

First, make sure you have a valid Fyers access token:

```bash
python fyers_auth.py
```

This will:
- Open your browser for Fyers login
- Generate an `access_token.txt` file
- Token is valid for the trading session

### 2. Test Fyers Data Fetching

Test with a few stocks to ensure it works:

```bash
python fyers_data_fetcher.py test
```

This will fetch 30 days of 5-minute data for 5 stocks and display sample output.

### 3. Fetch Full Training Dataset

Fetch data for all Nifty 100 stocks:

```bash
python fyers_data_fetcher.py
```

Or from Python:

```python
from fyers_data_fetcher import fetch_5m_data_fyers
from data_fetcher import NIFTY_100

# Fetch 30 days of data
data_dict = fetch_5m_data_fyers(NIFTY_100, days=30)

# Use for backtesting or optimization
for ticker, df in data_dict.items():
    print(f"{ticker}: {len(df)} bars")
```

## Using Fyers Data in Your Scripts

### Method 1: Environment Variable

Set the data source globally:

```bash
export DATA_SOURCE=fyers
python backtest_engine.py
```

### Method 2: Direct Import

Use Fyers data fetcher directly:

```python
from fyers_data_fetcher import fetch_5m_data_fyers

# Fetch data
data_dict = fetch_5m_data_fyers(['HDFCBANK.NS', 'RELIANCE.NS'], days=30)
```

### Method 3: Override in Code

Pass `source='fyers'` to the fetch function:

```python
from data_fetcher import fetch_5m_data

# Use Fyers for this specific call
data_dict = fetch_5m_data(source='fyers', period='30d')
```

## Parameter Optimization with Fyers Data

To run parameter optimization using Fyers data:

```python
from parameter_tuner import run_optimization

# Edit parameter_tuner.py to use Fyers:
# Change line in run_optimization():
#   data_dict = fetch_5m_data(NIFTY_100, period="30d", source='fyers')
```

## Caching and Performance

### Cache Location

- Fyers data is cached in `data_fyers/` directory
- Each stock saved as `{SYMBOL}_5m_fyers.csv`
- Cache expires after 5 minutes (fresh data on each trading day)

### Rate Limiting

The fetcher includes automatic rate limiting:
- Small delay every 10 requests
- Respects Fyers API limits
- Can fetch 100 stocks in ~1-2 minutes

### Cache Management

Clear cache to force fresh data fetch:

```bash
rm -rf data_fyers/
python fyers_data_fetcher.py
```

## Comparison: Fyers vs yfinance

| Feature | Fyers API | yfinance |
|---------|-----------|----------|
| **Data Quality** | Exchange-grade | Good (from Yahoo) |
| **Reliability** | Very high | Moderate |
| **Rate Limits** | Broker limits | Frequent rate limits |
| **Speed** | Fast | Can be slow |
| **Authentication** | Required | None needed |
| **Cost** | Free (with account) | Free |
| **Historical Limit** | Depends on plan | Up to 60 days for 5m |
| **Consistency** | Same as live | Different source |

## Troubleshooting

### "Access token missing" error

Run authentication again:
```bash
python fyers_auth.py
```

### "Failed to fetch" errors

1. Check your Fyers account status
2. Verify access token is valid
3. Check if markets are open (for live data)
4. Ensure symbol format is correct

### "No data returned" errors

- Verify stock symbol is correct
- Check date range (weekends have no data)
- Ensure stock is actively traded

### Import errors

Install Fyers API:
```bash
pip install fyers-apiv3
```

## Data Format

Both Fyers and yfinance return the same DataFrame format:

```
Datetime (index) | Open | High | Low | Close | Volume
2024-01-01 09:15 | 1500 | 1505 | 1498| 1502  | 100000
2024-01-01 09:20 | 1502 | 1508 | 1501| 1506  | 120000
...
```

This ensures seamless integration with existing strategy code.

## Advanced Usage

### Custom Date Range

```python
from fyers_data_fetcher import fetch_fyers_historical_data
from fyers_auth import get_fyers_client

fyers = get_fyers_client()

df = fetch_fyers_historical_data(
    fyers,
    symbol='HDFCBANK',
    from_date='2024-01-01',
    to_date='2024-02-01',
    resolution='5'  # 5-minute candles
)
```

### Different Timeframes

Available resolutions:
- `'1'` - 1 minute
- `'5'` - 5 minutes (default)
- `'15'` - 15 minutes
- `'30'` - 30 minutes
- `'60'` - 1 hour
- `'D'` - 1 day

### Batch Processing

```python
from fyers_data_fetcher import fetch_5m_data_fyers

# Process in batches
batches = [
    NIFTY_100[:25],   # First 25
    NIFTY_100[25:50], # Next 25
    NIFTY_100[50:75], # Next 25
    NIFTY_100[75:]    # Last 25
]

all_data = {}
for batch in batches:
    data = fetch_5m_data_fyers(batch, days=30)
    all_data.update(data)
    print(f"Processed batch, total: {len(all_data)} stocks")
```

## Integration with Backtest Engine

The backtest engine automatically works with Fyers data:

```python
from data_fetcher import fetch_5m_data
from backtest_engine import run_backtest

# Set environment variable or use source parameter
data_dict = fetch_5m_data(source='fyers', period='30d')

# Run backtest as usual
for ticker, df in data_dict.items():
    trades = run_backtest(df, ticker)
    # Process trades...
```

## Best Practices

1. **Always authenticate before fetching data**
   - Run `fyers_auth.py` at the start of each trading session
   - Check `access_token.txt` exists

2. **Use caching effectively**
   - Let the system cache data for faster subsequent runs
   - Clear cache at start of new trading day

3. **Verify data quality**
   - Check for missing bars
   - Validate price ranges
   - Ensure volume data is present

4. **Monitor API usage**
   - Respect rate limits
   - Don't fetch unnecessarily
   - Use cache for backtesting

5. **Keep tokens secure**
   - Don't commit `access_token.txt` to git
   - Don't share your access token
   - Regenerate if compromised

## Support

For issues with:
- **Fyers API**: Contact Fyers support
- **Data fetcher**: Check logs and error messages
- **Integration**: Review this guide and code comments

---

**Last Updated**: 2026-04-07  
**Version**: 1.0  
**Status**: Production Ready
