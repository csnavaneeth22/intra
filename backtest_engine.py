import pandas as pd
import numpy as np
from data_fetcher import fetch_5m_data, NIFTY_100
from strategy import calculate_strategy_signals


def run_backtest(df, ticker, initial_capital=25000, risk_per_trade=0.01):
    """
    Simulate the strategy on a single dataframe.
    Returns trade log list.
    """
    df = calculate_strategy_signals(df)
    
    trades = []
    in_position = False
    entry_price = 0
    entry_index = 0
    position_type = 0 # 1 long, -1 short
    qty = 0
    capital = initial_capital
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check Exits if in position
        if in_position:
            bars_held = i - entry_index
            exit_signal = False
            exit_reason = ""
            current_price = row['Close']
            
            # Optimized Profit Target / Stop Loss (2.5% profit / 0.9% stop loss)
            if position_type == 1:
                if current_price >= entry_price * 1.025:
                    exit_signal, exit_reason = True, "Take Profit 2.5%"
                elif current_price <= entry_price * 0.991:
                    exit_signal, exit_reason = True, "Stop Loss 0.9%"
                elif row['kernel_bearish_change'] or row['startShortTrade'] or bars_held >= 15:
                    exit_signal, exit_reason = True, "Dynamic/Time limit"
            elif position_type == -1:
                if current_price <= entry_price * 0.975:
                    exit_signal, exit_reason = True, "Take Profit 2.5%"
                elif current_price >= entry_price * 1.009:
                    exit_signal, exit_reason = True, "Stop Loss 0.9%"
                elif row['kernel_bullish_change'] or row['startLongTrade'] or bars_held >= 15:
                    exit_signal, exit_reason = True, "Dynamic/Time limit"
                    
            if exit_signal:
                exit_price = current_price
                pnl = (exit_price - entry_price) * qty * position_type
                capital += pnl
                
                trades.append({
                    'Ticker': ticker,
                    'Date': df.index[entry_index].date(),
                    'EntryTime': df.index[entry_index],
                    'ExitTime': df.index[i],
                    'Type': 'Long' if position_type == 1 else 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl,
                    'Reason': exit_reason,
                    'Win': 1 if pnl > 0 else 0
                })
                in_position = False
                continue
                
        # Check Entries
        if not in_position:
            if row['startLongTrade']:
                position_type = 1
                entry_price = row['Close']
                entry_index = i
                in_position = True
            elif row['startShortTrade']:
                position_type = -1
                entry_price = row['Close']
                entry_index = i
                in_position = True
                
            if in_position:
                # 2. MIS Leverage System (1% Risk of capital, leveraged 5x via Intraday Margin)
                MARGIN_LEVERAGE = 5 
                risk_amount = (capital * risk_per_trade) * MARGIN_LEVERAGE
                assumed_sl_distance = entry_price * 0.01 
                qty = np.floor(risk_amount / assumed_sl_distance) if assumed_sl_distance > 0 else 0
                if qty <= 0:
                    qty = 1

    return trades

def analyze_all():
    # Only test the curated roster
    data_dict = fetch_5m_data(NIFTY_100, period="30d")
    
    all_trades = []
    for ticker, df in data_dict.items():
        # Using 2% Risk 
        trades = run_backtest(df, ticker, initial_capital=25000, risk_per_trade=0.02)
        all_trades.extend(trades)
        
    if not all_trades:
        print("No trades found.")
        return
        
    res_df = pd.DataFrame(all_trades)
    
    # 1. Total Win Rate
    overall_win_rate = res_df['Win'].mean() * 100
    total_pnl = res_df['PnL'].sum()
    total_trades = len(res_df)
    
    print("\n--- Portfolio Summary (15 Stocks) ---")
    print(f"Start Capital: 25,000 INR")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Total Trades over 30 days: {total_trades}")
    print(f"Total Portfolio PnL: {total_pnl:.2f} INR")
    
    # 2. Daily Analysis
    daily_stats = res_df.groupby('Date').agg(
        DailyPnL=('PnL', 'sum'),
        TradesPerDay=('PnL', 'count')
    )
    
    avg_daily_trades = daily_stats['TradesPerDay'].mean()
    avg_daily_pnl = daily_stats['DailyPnL'].mean()
    avg_daily_return_pct = (avg_daily_pnl / 25000) * 100
    
    print("\n--- Daily Performance ---")
    print(f"Average Trades per Day across portfolio: {avg_daily_trades:.1f} trades")
    print(f"Average Daily PnL: {avg_daily_pnl:.2f} INR (+{avg_daily_return_pct:.2f}%)")
    print(f"Max Trades in a single day: {daily_stats['TradesPerDay'].max()}")
    
    # To hit 20 trades a day:
    # 15 stocks gave X trades. We can extrapolate to 100 stocks.
    projected_daily = avg_daily_trades * (100 / 15)
    print(f"\nProjection for 100 Stocks: ~{projected_daily:.1f} trades daily.")

if __name__ == "__main__":
    analyze_all()
