import yfinance as yf
import pandas as pd
import os
import time
# Truncated list for simplicity, but covers major ones.
# Accurate Nifty 100 (Nifty 50 + Next 50) Source Components
NIFTY_100 = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
    "ADANIPOWER.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "DMART.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "BANKBARODA.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BOSCHLTD.NS",
    "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "COALINDIA.NS", "CUMMINSIND.NS", "DLF.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "ETERNAL.NS", "GAIL.NS", "GODREJCP.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HAL.NS", "HINDUNILVR.NS", "HINDZINC.NS", "HYUNDAI.NS", "ICICIBANK.NS",
    "ITC.NS", "INDHOTEL.NS", "IOC.NS", "IRFC.NS", "INFY.NS",
    "INDIGO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS",
    "LTM.NS", "LT.NS", "LODHA.NS", "M&M.NS", "MARUTI.NS",
    "MAXHEALTH.NS", "MAZDOCK.NS", "MUTHOOTFIN.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "PIDILITIND.NS", "PFC.NS", "POWERGRID.NS", "PNB.NS",
    "RECLTD.NS", "RELIANCE.NS", "SBILIFE.NS", "MOTHERSON.NS", "SHREECEM.NS",
    "SHRIRAMFIN.NS", "ENRIN.NS", "SIEMENS.NS", "SOLARINDS.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TVSMOTOR.NS", "TATACAP.NS", "TCS.NS", "TATACONSUM.NS",
    "TMCV.NS", "TMPV.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", "UNIONBANK.NS",
    "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "WIPRO.NS", "ZYDUSLIFE.NS"
]

def fetch_5m_data(tickers=NIFTY_100, period="60d", cache_dir="data"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    data_dict = {}
    print(f"Fetching {period} of 5m data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}_5m.csv")
        
        # FIX: If we restart the engine mid-day, the old CSV cache causes the AI to see a massive fake 'gap' candle.
        # So we delete the cache if it's older than 5 minutes to ensure continuous data is properly fetched.
        if os.path.exists(cache_file):
            file_age_minutes = (time.time() - os.path.getmtime(cache_file)) / 60
            if file_age_minutes > 5:
                os.remove(cache_file)
        
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col="Datetime", parse_dates=True)
            if not df.empty:
                data_dict[ticker] = df
            continue
            
        try:
            df = yf.download(ticker, period=period, interval="5m", progress=False)
            if not df.empty:
                # yfinance returns multi-index columns sometimes, flatten them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Standardize column names
                df.columns = [c.capitalize() for c in df.columns]
                df.index.name = "Datetime"
                # Strip timezone to avoid tz-naive vs tz-aware comparison issues
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_csv(cache_file)
                data_dict[ticker] = df
            else:
                print(f"Failed to fetch {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    return data_dict

if __name__ == "__main__":
    fetch_5m_data(NIFTY_100[:5]) # test with 5
