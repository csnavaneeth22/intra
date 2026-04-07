import math
from datetime import datetime

# Global state for Paper Trading
paper_state = {
    "initial_fund": 0.0,
    "current_capital": 0.0,
    "daily_trades_count": 0,
    "open_trades": {},  # { "symbol": { entry_price, qty, side, time } }
    "closed_trades": [],
    "max_trades": 20
}

def set_initial_paper_fund(fund):
    paper_state["initial_fund"] = float(fund)
    paper_state["current_capital"] = float(fund)
    print(f"\n[PAPER TRADING] Initial Capital Set: ₹{paper_state['initial_fund']:,.2f}")

def calculate_indian_intraday_taxes(entry, exit, qty, side):
    """
    Calculates precise Indian Equity Intraday (MIS) Charges.
    """
    buy_value = entry * qty if side == 1 else exit * qty
    sell_value = exit * qty if side == 1 else entry * qty
    total_turnover = buy_value + sell_value
    
    # 1. Brokerage: Lower of ₹20 or 0.03% per leg.
    buy_brokerage = min(20, buy_value * 0.0003)
    sell_brokerage = min(20, sell_value * 0.0003)
    total_brokerage = buy_brokerage + sell_brokerage
    
    # 2. STT: 0.025% on sell side
    stt = round(sell_value * 0.00025)
    
    # 3. Exchange Transaction Charges (NSE roughly 0.00325%)
    txn_charge = total_turnover * 0.0000325
    
    # 4. SEBI Charges (₹10 / Crore)
    sebi_charge = total_turnover * 0.000001
    
    # 5. GST (18% on Brokerage + Txn + SEBI)
    gst = (total_brokerage + txn_charge + sebi_charge) * 0.18
    
    # 6. Stamp Duty (0.003% on Buy side only)
    stamp_duty = round(buy_value * 0.00003)
    
    total_taxes_and_brokerage = total_brokerage + stt + txn_charge + sebi_charge + gst + stamp_duty
    return total_taxes_and_brokerage

def evaluate_ticks_for_paper_exits(symbol, ltp):
    """
    Called on EVERY tick to simulate perfect Live Order Execution against Limits.
    """
    if symbol not in paper_state["open_trades"]:
        return
        
    trade = paper_state["open_trades"][symbol]
    entry = trade["entry_price"]
    side = trade["side"]
    qty = trade["qty"]
    
    exit_triggered = False
    reason = ""
    
    # Optimized exit thresholds
    if side == 1:
        if ltp >= entry * 1.025:
            exit_triggered, reason = True, "Take Profit 2.5%"
        elif ltp <= entry * 0.991:
            exit_triggered, reason = True, "Stop Loss 0.9%"
    elif side == -1:
        if ltp <= entry * 0.975:
            exit_triggered, reason = True, "Take Profit 2.5%"
        elif ltp >= entry * 1.009:
            exit_triggered, reason = True, "Stop Loss 0.9%"
            
    if exit_triggered:
        close_paper_trade(symbol, ltp, reason)


def close_paper_trade(symbol, exit_price, reason):
    """Closes trade and commits the net PnL."""
    trade = paper_state["open_trades"].pop(symbol)
    entry = trade["entry_price"]
    side = trade["side"]
    qty = trade["qty"]
    
    # Gross PnL
    gross_pnl = (exit_price - entry) * qty * side
    
    # Taxes
    intraday_taxes = calculate_indian_intraday_taxes(entry, exit_price, qty, side)
    
    # Net PnL
    net_pnl = gross_pnl - intraday_taxes
    
    # Update Capital
    paper_state["current_capital"] += net_pnl
    
    log = {
        "symbol": symbol,
        "side": "LONG" if side == 1 else "SHORT",
        "entry": entry,
        "exit": exit_price,
        "qty": qty,
        "gross": gross_pnl,
        "taxes": intraday_taxes,
        "net": net_pnl,
        "reason": reason,
        "time": datetime.now()
    }
    paper_state["closed_trades"].append(log)
    
    print(f"\n[PAPER EXIT] {symbol} {log['side']} Closed @ ₹{exit_price:.2f} ({reason})")
    print(f"Gross: ₹{gross_pnl:.2f} | Taxes/Brokerage: ₹{intraday_taxes:.2f} | NET: ₹{net_pnl:.2f}")
    print(f"Current Virtual Capital: ₹{paper_state['current_capital']:,.2f}\n")

def enter_paper_trade(symbol, side, signal_price):
    if paper_state["daily_trades_count"] >= paper_state["max_trades"]:
        return
        
    if symbol in paper_state["open_trades"]:
        return # Already in position for this stock 
        
    # Math Sizing directly resembling our App's 5x Limit sizing
    margin_leverage = 5
    risk_amount = (paper_state["current_capital"] * 0.01) * margin_leverage
    assumed_sl_distance = signal_price * 0.01 
    
    qty = math.floor(risk_amount / assumed_sl_distance) if assumed_sl_distance > 0 else 1
    if qty <= 0: qty = 1
        
    paper_state["open_trades"][symbol] = {
        "entry_price": signal_price,
        "side": side,
        "qty": qty,
        "time": datetime.now()
    }
    
    paper_state["daily_trades_count"] += 1
    
    side_str = "LONG" if side == 1 else "SHORT"
    print(f"[PAPER ENTRY] {symbol} {side_str} Executed @ ₹{signal_price:.2f} | Qty: {qty}")

def end_of_day_summary():
    """Prints the final net summary."""
    print("\n" + "="*40)
    print(" END OF DAY PAPER TRADING SUMMARY")
    print("="*40)
    
    total_trades = len(paper_state["closed_trades"])
    gross_total = sum(t["gross"] for t in paper_state["closed_trades"])
    taxes_total = sum(t["taxes"] for t in paper_state["closed_trades"])
    net_total = sum(t["net"] for t in paper_state["closed_trades"])
    
    wins = sum(1 for t in paper_state["closed_trades"] if t["net"] > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Initial Fund:    ₹{paper_state['initial_fund']:,.2f}")
    print(f"Final Fund:      ₹{paper_state['current_capital']:,.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print("-" * 40)
    print(f"Gross Profit:   ₹{gross_total:,.2f}")
    print(f"Total Charges: -₹{taxes_total:,.2f}")
    print(f" NET PROFIT:    ₹{net_total:,.2f}")
    print("="*40 + "\n")
