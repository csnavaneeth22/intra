"""
Parameter Tuning Framework for Lorentzian Trading Strategy
Performs forward testing with trade limiting to find optimal parameters
"""
import pandas as pd
import numpy as np
from itertools import product
from data_fetcher import fetch_5m_data, NIFTY_100
from strategy import calculate_strategy_signals
from indicators import rsi, wt, cci, adx, rationalize_quadratic_kernel
from lorentzian_ml import get_lorentzian_predictions


def calculate_strategy_signals_tunable(df, params):
    """
    Modified version of calculate_strategy_signals that accepts tunable parameters
    """
    if len(df) < 200:
        return df

    close = df['Close']
    high = df['High']
    low = df['Low']

    # 1. Feature Engineering with tunable parameters
    df['f1_rsi'] = rsi(close, params['rsi_period_1'])
    df['f2_wt'] = wt(high, low, close, params['wt_n1'], params['wt_n2'])
    df['f3_cci'] = cci(high, low, close, params['cci_period'])
    df['f4_adx'] = adx(high, low, close, params['adx_period'])
    df['f5_rsi'] = rsi(close, params['rsi_period_2'])

    # Fill NAs
    df.bfill(inplace=True)

    # Normalize features
    features_df = df[['f1_rsi', 'f2_wt', 'f3_cci', 'f4_adx', 'f5_rsi']].copy()
    features_df = (features_df - features_df.rolling(200).min()) / (features_df.rolling(200).max() - features_df.rolling(200).min() + 1e-8)
    features_df.fillna(0, inplace=True)

    # 2. Lorentzian ML Predictions with tunable parameters
    df['ml_prediction'] = get_lorentzian_predictions(
        features_df, close,
        neighbors_count=params['neighbors_count'],
        max_bars_back=params['max_bars_back']
    )

    # 3. Filters with tunable parameters
    df['ema'] = close.ewm(span=params['ema_period'], adjust=False).mean()
    df['sma'] = close.rolling(window=params['sma_period']).mean()

    df['is_ema_uptrend'] = close > df['ema']
    df['is_ema_downtrend'] = close < df['ema']
    df['is_sma_uptrend'] = close > df['sma']
    df['is_sma_downtrend'] = close < df['sma']

    # ADX filter with tunable threshold
    df['adx_filter'] = df['f4_adx'] > params['adx_threshold']

    # Kernel Regression Filter with tunable parameters
    df['yhat1'] = rationalize_quadratic_kernel(close, h=params['kernel_h'], r=params['kernel_r'], x=25)
    df['yhat1_lag'] = df['yhat1'].shift(1)
    df['is_bullish'] = df['yhat1'] > df['yhat1_lag']
    df['is_bearish'] = df['yhat1'] < df['yhat1_lag']

    df['kernel_bullish_change'] = (df['yhat1'] > df['yhat1_lag']) & (df['yhat1'].shift(1) < df['yhat1'].shift(2))
    df['kernel_bearish_change'] = (df['yhat1'] < df['yhat1_lag']) & (df['yhat1'].shift(1) > df['yhat1'].shift(2))

    # Signal generation with tunable ml_threshold
    df['signal'] = 0
    df.loc[(df['ml_prediction'] > params['ml_threshold']) & df['adx_filter'], 'signal'] = 1
    df.loc[(df['ml_prediction'] < -params['ml_threshold']) & df['adx_filter'], 'signal'] = -1

    df['signal_changed'] = df['signal'] != df['signal'].shift(1)

    # Entry conditions
    df['startLongTrade'] = (df['signal'] == 1) & df['signal_changed'] & df['is_bullish'] & df['is_ema_uptrend'] & df['is_sma_uptrend']
    df['startShortTrade'] = (df['signal'] == -1) & df['signal_changed'] & df['is_bearish'] & df['is_ema_downtrend'] & df['is_sma_downtrend']

    return df


def run_backtest_tunable(df, ticker, params, initial_capital=25000):
    """
    Backtest with tunable exit parameters and trade tracking
    """
    df = calculate_strategy_signals_tunable(df.copy(), params)

    trades = []
    in_position = False
    entry_price = 0
    entry_index = 0
    position_type = 0
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

            # Tunable Profit Target / Stop Loss
            if position_type == 1:
                if current_price >= entry_price * (1 + params['take_profit']):
                    exit_signal, exit_reason = True, "Take Profit"
                elif current_price <= entry_price * (1 - params['stop_loss']):
                    exit_signal, exit_reason = True, "Stop Loss"
                elif row['kernel_bearish_change'] or row['startShortTrade'] or bars_held >= params['max_hold_bars']:
                    exit_signal, exit_reason = True, "Dynamic/Time"
            elif position_type == -1:
                if current_price <= entry_price * (1 - params['take_profit']):
                    exit_signal, exit_reason = True, "Take Profit"
                elif current_price >= entry_price * (1 + params['stop_loss']):
                    exit_signal, exit_reason = True, "Stop Loss"
                elif row['kernel_bullish_change'] or row['startLongTrade'] or bars_held >= params['max_hold_bars']:
                    exit_signal, exit_reason = True, "Dynamic/Time"

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
                MARGIN_LEVERAGE = params['margin_leverage']
                risk_amount = (capital * params['risk_per_trade']) * MARGIN_LEVERAGE
                assumed_sl_distance = entry_price * params['stop_loss']
                qty = np.floor(risk_amount / assumed_sl_distance) if assumed_sl_distance > 0 else 0
                if qty <= 0:
                    qty = 1

    return trades, capital


def limit_trades_per_day(trades, max_trades_per_day=20):
    """
    Simulate real trading by limiting trades per day to match live conditions
    Takes the first N trades per day to simulate what would actually happen
    """
    if not trades:
        return []

    df = pd.DataFrame(trades)
    limited_trades = []

    for date in df['Date'].unique():
        day_trades = df[df['Date'] == date].sort_values('EntryTime')
        # Take only the first max_trades_per_day trades
        limited_trades.extend(day_trades.head(max_trades_per_day).to_dict('records'))

    return limited_trades


def evaluate_parameters(params, data_dict, max_trades_per_day=20, initial_capital=25000):
    """
    Evaluate a parameter set across all stocks with trade limiting
    """
    all_trades = []

    for ticker, df in data_dict.items():
        trades, _ = run_backtest_tunable(df, ticker, params, initial_capital)
        all_trades.extend(trades)

    # Apply daily trade limit to simulate live conditions
    limited_trades = limit_trades_per_day(all_trades, max_trades_per_day)

    if not limited_trades:
        return {
            'total_pnl': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_daily_pnl': 0,
            'avg_daily_trades': 0,
            'monthly_return_pct': 0,
            'trades_per_day': 0
        }

    res_df = pd.DataFrame(limited_trades)

    # Calculate metrics
    total_pnl = res_df['PnL'].sum()
    win_rate = res_df['Win'].mean() * 100
    total_trades = len(res_df)

    # Daily statistics
    daily_stats = res_df.groupby('Date').agg(
        DailyPnL=('PnL', 'sum'),
        TradesPerDay=('PnL', 'count')
    )

    avg_daily_pnl = daily_stats['DailyPnL'].mean()
    avg_daily_trades = daily_stats['TradesPerDay'].mean()

    # Monthly projection (30 days)
    monthly_pnl = avg_daily_pnl * 30
    monthly_return_pct = (monthly_pnl / initial_capital) * 100

    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_daily_pnl': avg_daily_pnl,
        'avg_daily_trades': avg_daily_trades,
        'monthly_return_pct': monthly_return_pct,
        'trades_per_day': avg_daily_trades
    }


def get_default_params():
    """Current baseline parameters"""
    return {
        # Feature parameters
        'rsi_period_1': 14,
        'rsi_period_2': 9,
        'wt_n1': 10,
        'wt_n2': 11,
        'cci_period': 20,
        'adx_period': 20,

        # ML parameters
        'neighbors_count': 8,
        'max_bars_back': 2000,
        'ml_threshold': 0,

        # Filter parameters
        'ema_period': 50,
        'sma_period': 50,
        'adx_threshold': 20,

        # Kernel parameters
        'kernel_h': 8,
        'kernel_r': 8.0,

        # Exit parameters
        'take_profit': 0.02,  # 2%
        'stop_loss': 0.01,    # 1%
        'max_hold_bars': 12,

        # Risk parameters
        'risk_per_trade': 0.02,  # 2%
        'margin_leverage': 5
    }


def grid_search_optimization(data_dict, max_trades_per_day=20):
    """
    Perform grid search over key parameters
    Focus on parameters with highest impact
    """
    print("Starting parameter optimization...")
    print(f"Target: {max_trades_per_day} trades per day\n")

    # Define parameter grid (focused on high-impact parameters)
    param_grid = {
        # ML parameters - most critical
        'neighbors_count': [6, 8, 10, 12],
        'ml_threshold': [0, 1],

        # Exit parameters - high impact on returns
        'take_profit': [0.015, 0.02, 0.025, 0.03],  # 1.5%, 2%, 2.5%, 3%
        'stop_loss': [0.008, 0.01, 0.012],    # 0.8%, 1%, 1.2%

        # Filter parameters
        'adx_threshold': [15, 20, 25],
        'ema_period': [40, 50],

        # Risk parameters
        'risk_per_trade': [0.015, 0.02],
        'max_hold_bars': [10, 12, 15],
    }

    # Get baseline parameters
    base_params = get_default_params()

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    print(f"Testing {len(combinations)} parameter combinations...\n")

    best_params = None
    best_score = -float('inf')
    best_metrics = None
    results = []

    for i, combination in enumerate(combinations):
        # Create parameter set
        params = base_params.copy()
        for key, value in zip(keys, combination):
            params[key] = value

        # Evaluate
        metrics = evaluate_parameters(params, data_dict, max_trades_per_day)

        # Score: prioritize monthly return, but consider win rate and trade count
        score = metrics['monthly_return_pct'] + (metrics['win_rate'] * 0.1) + (min(metrics['trades_per_day'], max_trades_per_day) * 0.5)

        results.append({
            'params': params.copy(),
            'metrics': metrics,
            'score': score
        })

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = metrics.copy()

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Tested {i + 1}/{len(combinations)} combinations... Best monthly return so far: {best_metrics['monthly_return_pct']:.2f}%")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    return best_params, best_metrics, results


def run_optimization(use_synthetic=True):
    """Main optimization function"""
    if use_synthetic:
        print("Generating synthetic data for 100 stocks (30 days)...")
        from synthetic_data_generator import generate_nifty100_synthetic_data
        data_dict = generate_nifty100_synthetic_data(days=30)
        print(f"Generated data for {len(data_dict)} stocks\n")
    else:
        print("Fetching data for 100 stocks (30 days)...")
        data_dict = fetch_5m_data(NIFTY_100, period="30d")
        print(f"Loaded data for {len(data_dict)} stocks\n")

    # Get baseline performance
    print("Evaluating baseline parameters...")
    baseline_params = get_default_params()
    baseline_metrics = evaluate_parameters(baseline_params, data_dict, max_trades_per_day=20)

    print("\nBASELINE PERFORMANCE:")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2f}%")
    print(f"  Total Trades: {baseline_metrics['total_trades']}")
    print(f"  Avg Trades/Day: {baseline_metrics['avg_daily_trades']:.1f}")
    print(f"  Avg Daily PnL: ₹{baseline_metrics['avg_daily_pnl']:.2f}")
    print(f"  Monthly Return: {baseline_metrics['monthly_return_pct']:.2f}%")
    print(f"  Total PnL: ₹{baseline_metrics['total_pnl']:.2f}\n")

    # Run optimization
    best_params, best_metrics, all_results = grid_search_optimization(data_dict, max_trades_per_day=20)

    print("\nOPTIMIZED PERFORMANCE:")
    print(f"  Win Rate: {best_metrics['win_rate']:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    print(f"  Avg Trades/Day: {best_metrics['avg_daily_trades']:.1f}")
    print(f"  Avg Daily PnL: ₹{best_metrics['avg_daily_pnl']:.2f}")
    print(f"  Monthly Return: {best_metrics['monthly_return_pct']:.2f}%")
    print(f"  Total PnL: ₹{best_metrics['total_pnl']:.2f}")

    print("\n" + "="*80)
    print("IMPROVEMENT:")
    print("="*80)
    daily_improvement = ((best_metrics['avg_daily_pnl'] - baseline_metrics['avg_daily_pnl']) / abs(baseline_metrics['avg_daily_pnl'])) * 100 if baseline_metrics['avg_daily_pnl'] != 0 else 0
    monthly_improvement = ((best_metrics['monthly_return_pct'] - baseline_metrics['monthly_return_pct']) / abs(baseline_metrics['monthly_return_pct'])) * 100 if baseline_metrics['monthly_return_pct'] != 0 else 0

    print(f"  Daily PnL Improvement: {daily_improvement:+.2f}%")
    print(f"  Monthly Return Improvement: {monthly_improvement:+.2f}%")
    print(f"  Win Rate Change: {best_metrics['win_rate'] - baseline_metrics['win_rate']:+.2f}%")

    print("\n" + "="*80)
    print("OPTIMIZED PARAMETERS:")
    print("="*80)
    for key, value in best_params.items():
        if baseline_params[key] != value:
            print(f"  {key}: {baseline_params[key]} → {value} *CHANGED*")
        else:
            print(f"  {key}: {value}")

    # Save results
    print("\nSaving optimization results...")
    results_df = pd.DataFrame([{
        **r['params'],
        **{f'metric_{k}': v for k, v in r['metrics'].items()},
        'score': r['score']
    } for r in all_results])
    results_df.to_csv('/home/runner/work/intra/intra/optimization_results.csv', index=False)

    # Save best parameters
    import json
    with open('/home/runner/work/intra/intra/best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    print("Results saved to optimization_results.csv and best_parameters.json")

    return best_params, best_metrics, baseline_metrics


if __name__ == "__main__":
    best_params, best_metrics, baseline_metrics = run_optimization()
