import time
import datetime
import pandas as pd
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from fyers_auth import get_fyers_client, CLIENT_ID
from live_executor import evaluate_and_trade
from paper_executor import set_initial_paper_fund, enter_paper_trade, evaluate_ticks_for_paper_exits, end_of_day_summary
from data_fetcher import fetch_5m_data, NIFTY_100
from strategy import calculate_strategy_signals

CURATED_ROSTER = NIFTY_100

# Memory structure to hold forming 5-min candles and historical data
data_cache = {}
IS_PAPER_TRADING = False
EOD_PRINTED = False
tick_count = 0
message_count = 0

def initialize_historical_data():
    """Fetches trailing data so our ML model has history to compute on instantly."""
    print("Pre-loading historical data for algorithms...")
    hist_dict = fetch_5m_data(CURATED_ROSTER, period="30d")
    for ticker, df in hist_dict.items():
        clean_ticker = ticker.replace('.NS', '')
        # yfinance stores timestamps in UTC; strip timezone info so they're naive-UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        data_cache[f"NSE:{clean_ticker}-EQ"] = df
    print(f"Historical data cached successfully for {len(data_cache)} symbols.")
    # Print a sample to verify timestamps
    sample_symbol = list(data_cache.keys())[0] if data_cache else None
    if sample_symbol:
        print(f"  Sample [{sample_symbol}] last candle: {data_cache[sample_symbol].index[-1]}")

def on_message(message):
    """
    Called whenever a tick is received from Fyers WebSocket.
    """
    global tick_count, message_count, EOD_PRINTED
    message_count += 1
    
    # Debug: print first few raw messages to understand format
    if message_count <= 5:
        print(f"[DEBUG MSG #{message_count}] {type(message).__name__}: {str(message)[:200]}")
    
    if not isinstance(message, dict):
        return
    if "symbol" not in message or "ltp" not in message:
        if message_count <= 10:
            print(f"[DEBUG] Skipping message without symbol/ltp: keys={list(message.keys())}")
        return
        
    symbol = message['symbol']
    try:
        ltp = float(message['ltp'])
    except (ValueError, TypeError) as e:
        print(f"[WARN] Bad ltp for {symbol}: {e}")
        return
    
    # Convert tick timestamp to naive-UTC (matching our yfinance data index)
    # Fyers sends epoch seconds → convert to UTC datetime (naive)
    if 'timestamp' in message:
        try:
            dt_utc = datetime.datetime.utcfromtimestamp(message['timestamp'])
        except (ValueError, TypeError, OSError):
            dt_utc = datetime.datetime.utcnow()
    else:
        dt_utc = datetime.datetime.utcnow()
    
    # IST time for display and EOD check
    dt_ist = dt_utc + datetime.timedelta(hours=5, minutes=30)
    
    tick_count += 1
    if tick_count <= 5 or tick_count % 500 == 0:
        print(f"[TICK #{tick_count}] {symbol} LTP=₹{ltp:.2f} UTC={dt_utc} IST={dt_ist.strftime('%H:%M:%S')}")
    
    # EOD Auto Summary Trigger (3:20 PM IST)
    if dt_ist.hour == 15 and dt_ist.minute >= 20 and not EOD_PRINTED:
        if IS_PAPER_TRADING:
            end_of_day_summary()
        EOD_PRINTED = True
        print("\n=== MARKET CLOSED | BOT ENTERING EOD STANDBY ===")
    
    # We snap the time to the floor of the 5 minute interval IN UTC
    minute = dt_utc.minute
    floor_minute = minute - (minute % 5)
    candle_time = dt_utc.replace(minute=floor_minute, second=0, microsecond=0)
    
    df = data_cache.get(symbol)
    if df is None:
        return
        
    # 0. If paper trading is active, check intra-candle Stop Loss / Take profit immediately
    if IS_PAPER_TRADING:
        # Strip NSE:-EQ part
        raw_ticker_tick = symbol.replace("NSE:", "").replace("-EQ", "")
        evaluate_ticks_for_paper_exits(raw_ticker_tick, ltp)
    
    # Check if this tick belongs to a new 5-minute candle than what's in our dataframe
    if df.index[-1] < candle_time:
        # A 5-minute boundary passed! The last candle is now CLOSED.
        # 1. Run the Algo on the closed history!
        print(f"\n[Candle Close] {symbol} @ {df.index[-1]} (new candle: {candle_time})")
        evaluated_df = calculate_strategy_signals(df)
        
        if evaluated_df is None or len(evaluated_df) < 200:
            # Not enough data for signal generation
            print(f"  [SKIP] Not enough data for {symbol} ({len(df)} bars)")
        else:
            last_row = evaluated_df.iloc[-1]
            raw_ticker = symbol.replace("NSE:", "").replace("-EQ", "")
            
            # 2. Trigger Executor if signal matches
            if last_row['startLongTrade']:
                print(f"  [SIGNAL] 🟢 LONG {raw_ticker} @ ₹{last_row['Close']:.2f}")
                if IS_PAPER_TRADING:
                    enter_paper_trade(raw_ticker, 1, last_row['Close'])
                else:
                    evaluate_and_trade(raw_ticker, 1, last_row['Close'])
            elif last_row['startShortTrade']:
                print(f"  [SIGNAL] 🔴 SHORT {raw_ticker} @ ₹{last_row['Close']:.2f}")
                if IS_PAPER_TRADING:
                    enter_paper_trade(raw_ticker, -1, last_row['Close'])
                else:
                    evaluate_and_trade(raw_ticker, -1, last_row['Close'])
            
        # 3. Create a new empty candle for the current timestamp
        new_row = pd.DataFrame([{
            'Open': ltp, 'High': ltp, 'Low': ltp, 'Close': ltp, 'Volume': 0
        }], index=[candle_time])
        new_row.index.name = "Datetime"
        data_cache[symbol] = pd.concat([df, new_row])
        
    else:
        # Update the currently forming candle dynamically
        data_cache[symbol].iloc[-1, data_cache[symbol].columns.get_loc('High')] = max(df.iloc[-1]['High'], ltp)
        data_cache[symbol].iloc[-1, data_cache[symbol].columns.get_loc('Low')] = min(df.iloc[-1]['Low'], ltp)
        data_cache[symbol].iloc[-1, data_cache[symbol].columns.get_loc('Close')] = ltp

def on_error(message):
    print("WebSocket Error:", message)

def on_close(message):
    print("WebSocket Connection Closed:", message)

def on_open():
    print("WebSocket Opened! Subscribing to symbols...")
    # Map the symbols expected by data websocket
    data_type = "SymbolUpdate" # Get Ltp, volume etc
    active_symbols = list(data_cache.keys())
    print(f"  Subscribing to {len(active_symbols)} symbols")
    print(f"  First 5: {active_symbols[:5]}")
    print(f"  Last 5: {active_symbols[-5:]}")
    if active_symbols:
        fyers_ws.subscribe(symbols=active_symbols, data_type=data_type)
        fyers_ws.keep_running()
    else:
        print("No active symbols to subscribe to. Exiting.")

if __name__ == "__main__":
    print("\n--- LORENTZIAN TRADING ENGINE ---")
    mode_input = input("Enter Paper Trading Initial Fund (e.g. 50000) or press Enter to skip and run REAL trading: ").strip()
    
    if mode_input and mode_input.replace('.', '', 1).isdigit() and float(mode_input) > 0:
        IS_PAPER_TRADING = True
        set_initial_paper_fund(mode_input)
    else:
        print("\n[WARNING] Executing in REAL Funds Live Mode.")
        
    initialize_historical_data()
    
    # WebSocket requires token
    with open("access_token.txt", "r") as f:
        token = f.read().strip()
    
    # Quick token validation
    print(f"\nToken length: {len(token)} chars")
    fyers_test = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=token, log_path="")
    profile = fyers_test.get_profile()
    print(f"Token validation: {profile}")
    
    if profile.get('s') != 'ok':
        print("\n[FATAL] Access token is INVALID or EXPIRED. Please re-run fyers_auth.py to generate a new token.")
        exit(1)
        
    # Fyers Websocket initialization standard format (AppId:Token)
    access_token_ws = f"{CLIENT_ID}:{token}"
    
    fyers_ws = data_ws.FyersDataSocket(
        access_token=access_token_ws,       
        log_path="",                     
        litemode=False,
        write_to_file=False,           
        reconnect=True,                  
        on_connect=on_open,               
        on_close=on_close,                
        on_error=on_error,                
        on_message=on_message             
    )

    fyers_ws.connect()
