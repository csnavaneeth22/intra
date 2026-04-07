"""
Quick parameter optimization with reduced grid
"""
from parameter_tuner import *
from synthetic_data_generator import generate_nifty100_synthetic_data

def quick_optimization():
    """Run a fast optimization with smaller grid"""
    print("Generating synthetic data for 20 stocks (30 days)...")
    from data_fetcher import NIFTY_100

    # Use only 20 stocks for faster testing
    selected_stocks = NIFTY_100[:20]

    data_dict = {}
    base_prices = [1000, 1500, 2000, 800, 1200, 900, 1800, 1100, 1600, 1300,
                   950, 1400, 1700, 850, 1250, 1050, 1550, 1150, 1450, 1350]

    for i, ticker in enumerate(selected_stocks):
        from synthetic_data_generator import generate_synthetic_stock_data
        df = generate_synthetic_stock_data(ticker, days=30, start_price=base_prices[i])
        data_dict[ticker] = df

    print(f"Generated data for {len(data_dict)} stocks\n")

    # Simplified parameter grid
    param_grid = {
        'neighbors_count': [6, 10],
        'ml_threshold': [0, 1],
        'take_profit': [0.02, 0.025],
        'stop_loss': [0.008, 0.01],
        'adx_threshold': [15, 25],
        'risk_per_trade': [0.015, 0.02],
    }

    print("Running quick optimization...")
    base_params = get_default_params()

    # Generate combinations
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    print(f"Testing {len(combinations)} parameter combinations...\n")

    best_params = None
    best_score = -float('inf')
    best_metrics = None

    # Get baseline
    baseline_params = get_default_params()
    baseline_metrics = evaluate_parameters(baseline_params, data_dict, max_trades_per_day=20)

    print("BASELINE PERFORMANCE:")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2f}%")
    print(f"  Avg Trades/Day: {baseline_metrics['avg_daily_trades']:.1f}")
    print(f"  Avg Daily PnL: ₹{baseline_metrics['avg_daily_pnl']:.2f}")
    print(f"  Monthly Return: {baseline_metrics['monthly_return_pct']:.2f}%\n")

    for i, combination in enumerate(combinations):
        params = base_params.copy()
        for key, value in zip(keys, combination):
            params[key] = value

        metrics = evaluate_parameters(params, data_dict, max_trades_per_day=20)
        score = metrics['monthly_return_pct'] + (metrics['win_rate'] * 0.1)

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = metrics.copy()

        if (i + 1) % 16 == 0:
            print(f"Progress: {i + 1}/{len(combinations)} - Best return: {best_metrics['monthly_return_pct']:.2f}%")

    print("\n" + "="*80)
    print("OPTIMIZED PERFORMANCE:")
    print(f"  Win Rate: {best_metrics['win_rate']:.2f}%")
    print(f"  Avg Trades/Day: {best_metrics['avg_daily_trades']:.1f}")
    print(f"  Avg Daily PnL: ₹{best_metrics['avg_daily_pnl']:.2f}")
    print(f"  Monthly Return: {best_metrics['monthly_return_pct']:.2f}%")

    print("\n" + "="*80)
    print("IMPROVEMENT:")
    daily_improvement = ((best_metrics['avg_daily_pnl'] - baseline_metrics['avg_daily_pnl']) / abs(baseline_metrics['avg_daily_pnl'] + 1)) * 100
    monthly_improvement = ((best_metrics['monthly_return_pct'] - baseline_metrics['monthly_return_pct']) / abs(baseline_metrics['monthly_return_pct'] + 1)) * 100

    print(f"  Daily PnL Improvement: {daily_improvement:+.2f}%")
    print(f"  Monthly Return Improvement: {monthly_improvement:+.2f}%")

    print("\n" + "="*80)
    print("OPTIMIZED PARAMETERS (changed only):")
    for key in param_grid.keys():
        if baseline_params[key] != best_params[key]:
            print(f"  {key}: {baseline_params[key]} → {best_params[key]}")

    # Save
    import json
    with open('/home/runner/work/intra/intra/best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    print("\nBest parameters saved to best_parameters.json")

    return best_params, best_metrics, baseline_metrics


if __name__ == "__main__":
    quick_optimization()
