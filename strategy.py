from lorentzian_backtester import compute_signals


def calculate_strategy_signals(df):
    """
    Thin wrapper to reuse the Lorentzian backtester’s signal computation so
    live trading and backtests share identical logic.
    """
    return compute_signals(df)
