import math
import time
from fyers_auth import get_fyers_client

# Constants
MAX_TRADES_PER_DAY = 20
RISK_PERCENT = 0.01 # 1% per trade risk
MARGIN_LEVERAGE = 5 

state = {
    "daily_trades": 0,
    "current_capital": 25000.0 # Fallback
}

def update_live_capital(fyers):
    """Fetches the real available balance in your Fyers account dynamically."""
    try:
        response = fyers.funds()
        if "fund_limit" in response:
            for item in response["fund_limit"]:
                # 'Available Balance' or 'Total Available Balance' holds the usable equity
                if item["title"] == "Available Balance" or item["id"] == 10:
                    state["current_capital"] = float(item["equityAmount"])
                    print(f"Live Capital Synced: ₹{state['current_capital']}")
                    return
    except Exception as e:
        print(f"Failed to sync live capital, using fallback: {e}")

def place_order_with_slippage_control(fyers, symbol, side, qty, signal_price):
    """
    Minimizes slippage by placing LIMIT orders. 
    Instead of using MARKET which gets filled at bad prices on 5-min intervals,
    we place a Limit order exactly at the current signal price. 
    If the momentum is real, you fill instantly or on minor retracement.
    """
    try:
        # 1 for Buy, -1 for Sell
        fyers_side = 1 if side == 1 else -1
        
        # Limit price set strictly to signal price to avoid bad bid-asks
        data = {
            "symbol": f"NSE:{symbol}-EQ",
            "qty": int(qty),
            "type": 1, # 1: Limit Order, 2: Market
            "side": fyers_side,
            "productType": "INTRADAY", 
            "limitPrice": round(signal_price, 2),
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        
        response = fyers.place_order(data=data)
        print(f"Order Placed [{symbol}]:", response)
        
        if "s" in response and response['s'] == 'ok':
            state["daily_trades"] += 1
            return response
            
    except Exception as e:
        print(f"Failed to place order for {symbol}. Error: {e}")
        
    return None

def evaluate_and_trade(symbol, current_signal, signal_price):
    """
    Called by the websocket data stream loop for every 5-minute close.
    """
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        print("Daily max trades reached (20). Ignored.")
        return
        
    if current_signal not in [1, -1]:
        return # No entry signal

    try:
        fyers = get_fyers_client()
        
        # Sync the capital once per order to always risk 1% of the latest account balance
        update_live_capital(fyers)
        
        # Sizing Calculation 
        risk_amount = (state["current_capital"] * RISK_PERCENT) * MARGIN_LEVERAGE
        assumed_sl_distance = signal_price * 0.01 
        qty = math.floor(risk_amount / assumed_sl_distance) if assumed_sl_distance > 0 else 1
        
        print(f"Executing {symbol} | Signal {current_signal} | Qty {qty} | Price {signal_price}")
        place_order_with_slippage_control(fyers, symbol, current_signal, qty, signal_price)

    except Exception as e:
        print(f"Trade Evaluation failed: {e}")

if __name__ == "__main__":
    print("Live Executor ready to listen for ML Signals...")
    # Test Mock execution locally
    # evaluate_and_trade("HDFCBANK", 1, 1500.50)
