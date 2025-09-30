# Dynamic + Signal-Driven Grid Bot for Kraken
# Supports: Grid trading, Trailing TP, RSI+MACD+Bollinger signal mode via config switch

import os
import time
import csv
import json
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from notifier import send_notification
from logger import get_logger

# ========= CONFIG ==========
CONFIG = {
    "symbol": "BTC/CAD",
    "timeframe": "5m",
    "cad_per_trade": 15,
    "min_trade_btc": 0.0001,
    "grid_levels": 10,
    "base_spacing": 50,
    "trailing_tp_pct": 0.004,
    "grid_state_file": "grid_state.json",
    "log_file": "logs/grid_bot.log",
    "trade_hours": [8, 21],
    "mode": "grid",  # options: grid, signal
    "rsi_buy": 25,
    "rsi_sell": 75,
    "bollinger_squeeze_threshold": 0.005,
    "stop_loss_pct": 0.03,
    "use_fvg": True,
    "use_structure": True,
    "csv_log_file": "trade_log.csv",
    "pnl_state_file": "pnl_state.json",
    "min_trade_interval_sec": 300  # New cooldown interval (5 minutes)
}

buy_price = None
last_trade = None
load_dotenv()
logger = get_logger("GridBot")

exchange = ccxt.kraken({
    'apiKey': os.getenv("KRAKEN_API_KEY"),
    'secret': os.getenv("KRAKEN_SECRET"),
    'enableRateLimit': True,
})

# ========= State ==========
def load_state():
    if CONFIG['mode'] == 'grid':
        if os.path.exists(CONFIG['grid_state_file']):
            with open(CONFIG['grid_state_file'], 'r') as f:
                return json.load(f)
        return {"btc_held": 0.0, "cash": 1000.0, "active_trades": [], "last_buy_time": 0}
    else:
        if os.path.exists(CONFIG['pnl_state_file']):
            with open(CONFIG['pnl_state_file'], 'r') as f:
                return json.load(f)
        return {"last_buy_price": 0.0, "cumulative_pnl": 0.0}

def save_state(state):
    path = CONFIG['grid_state_file'] if CONFIG['mode'] == 'grid' else CONFIG['pnl_state_file']
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

state = load_state()

# ========= Common ==========
def get_price():
    return exchange.fetch_ticker(CONFIG['symbol'])['last']

def is_within_trade_time():
    hour = time.localtime().tm_hour
    return CONFIG['trade_hours'][0] <= hour <= CONFIG['trade_hours'][1]

def fetch_ohlcv():
    data = exchange.fetch_ohlcv(CONFIG['symbol'], CONFIG['timeframe'], limit=100)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    return df

# ========= Grid Mode ==========
def get_grid_levels(center):
    return [center + i * CONFIG['base_spacing'] for i in range(-CONFIG['grid_levels'], CONFIG['grid_levels'] + 1)]

def check_buy(price, levels):
    return any(price <= lvl for lvl in levels)

def check_sell(price, trades):
    for trade in trades:
        if trade['type'] == 'BUY':
            target = trade['entry_price'] * (1 + CONFIG['trailing_tp_pct'])
            if price >= target:
                return trade
    return None


def place_trade(action, amount, price):
    logger.info(f"{action} {amount:.6f} BTC @ {price:.2f}")

    if action == "BUY":
        logger.info(f"💰 Cash after buy: {state['cash']:.2f}")
        logger.info(f"📊 BTC held after buy: {state['btc_held'] + amount:.6f}")
    elif action == "SELL":
        logger.info(f"💸 Cash after sell: {state['cash'] + (amount * price):.2f}")
        logger.info(f"📉 BTC held after sell: {state['btc_held'] - amount:.6f}")

    send_notification(f"{action} Triggered", f"{amount:.6f} BTC at {price:.2f} CAD")


# ========= Signal Mode ==========
def log_trade(action, amount, price, pnl=None):
    try:
        with open(CONFIG['csv_log_file'], mode="a", newline="") as file:
            writer = csv.writer(file)
            row = [time.strftime("%Y-%m-%d %H:%M:%S"), action, amount, f"{price:.2f}"]
            if pnl is not None:
                row.append(f"{pnl:.2f}")
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")

def detect_structure(df):
    last = df['close'].iloc[-1]
    prev = df['close'].iloc[-2]
    ph = df['high'].iloc[-2]
    pl = df['low'].iloc[-2]
    ch = df['high'].iloc[-1]
    cl = df['low'].iloc[-1]
    if pl < cl and last > ph:
        return "CHOCH_BULL"
    elif ph > ch and last < pl:
        return "CHOCH_BEAR"
    return None

def detect_fvg(df):
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if c1['high'] < c3['low']: return "FVG_BULL"
    if c1['low'] > c3['high']: return "FVG_BEAR"
    return None

def is_bollinger_squeeze(df):
    bb = BollingerBands(df['close'])
    width = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    return width.iloc[-1] <= width.rolling(20).min().iloc[-1] * (1 + CONFIG['bollinger_squeeze_threshold'])

def get_signal(df):
    try:
        close = df['close']
        rsi = RSIIndicator(close, 14).rsi().iloc[-1]
        ema9 = close.ewm(span=9).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        macd_line = MACD(close).macd().iloc[-1]
        signal_line = MACD(close).macd_signal().iloc[-1]
        bb = BollingerBands(close)
        bb_u = bb.bollinger_hband().iloc[-1]
        bb_l = bb.bollinger_lband().iloc[-1]
        price = close.iloc[-1]

        score = 0
        if rsi < CONFIG['rsi_buy']: score += 1
        if ema9 > ema21: score += 1
        if macd_line > signal_line: score += 1
        if price <= bb_l * 1.01: score += 1
        if is_bollinger_squeeze(df): score += 1
        if CONFIG['use_structure'] and detect_structure(df) == 'CHOCH_BULL': score += 1
        if CONFIG['use_fvg'] and detect_fvg(df) == 'FVG_BULL': score += 1
        if score >= 4: return "BUY"

        score = 0
        if rsi > CONFIG['rsi_sell']: score += 1
        if ema9 < ema21: score += 1
        if macd_line < signal_line: score += 1
        if price >= bb_u * 0.99: score += 1
        if CONFIG['use_structure'] and detect_structure(df) == 'CHOCH_BEAR': score += 1
        if CONFIG['use_fvg'] and detect_fvg(df) == 'FVG_BEAR': score += 1
        if score >= 4: return "SELL"

        return "HOLD"
    except Exception as e:
        logger.error(f"Signal error: {e}")
        return "HOLD"

def should_trade(signal):
    global last_trade
    if signal == last_trade or signal == "HOLD": return False
    last_trade = signal
    return True

def place_signal_order(signal):
    global buy_price, state
    price = get_price()
    if signal == "BUY":
        amount = CONFIG['cad_per_trade'] / price
        if amount < CONFIG['min_trade_btc']: return
        exchange.create_market_buy_order(CONFIG['symbol'], round(amount, 6))
        buy_price = price
        state['last_buy_price'] = price
        log_trade("BUY", amount, price)
        send_notification("BUY", f"{amount:.6f} BTC @ {price:.2f}")
    elif signal == "SELL":
        btc = exchange.fetch_balance()['free'].get('BTC', 0)
        if btc < CONFIG['min_trade_btc']: return
        pnl = (price - state.get('last_buy_price', 0)) * btc
        exchange.create_market_sell_order(CONFIG['symbol'], btc)
        state['cumulative_pnl'] += pnl
        log_trade("SELL", btc, price, pnl)
        send_notification("SELL", f"{btc:.6f} BTC @ {price:.2f}\nPnL: {pnl:.2f}")
        buy_price = None
    save_state(state)

# ========= Main ==========
if __name__ == "__main__":
    logger.info("🚀 Starting hybrid bot on Kraken")
    while True:
        try:
            if not is_within_trade_time():
                time.sleep(60); continue
            df = fetch_ohlcv()
            price = df['close'].iloc[-1]

            if CONFIG['mode'] == 'grid':
                levels = get_grid_levels(price)
                now = time.time()
                if check_buy(price, levels) and now - state.get('last_buy_time', 0) >= CONFIG['min_trade_interval_sec']:
                    available_cad = exchange.fetch_balance()['free'].get('CAD', 0)
                    if available_cad < CONFIG['cad_per_trade']:
                        logger.warning(f"❌ Not enough CAD to trade: {available_cad:.2f} < {CONFIG['cad_per_trade']}")
                        continue

                    amt = CONFIG['cad_per_trade'] / price
                    if amt >= CONFIG['min_trade_btc']:
                        place_trade("BUY", amt, price)
                        state['btc_held'] += amt
                        state['active_trades'].append({
                            "type": "BUY",
                            "entry_price": price,
                            "amount": amt
                        })
                        state['last_buy_time'] = now

                sell = check_sell(price, state['active_trades'])
                if sell and state['btc_held'] >= sell['amount']:
                    place_trade("SELL", sell['amount'], price)
                    state['btc_held'] -= sell['amount']
                    state['cash'] += sell['amount'] * price
                    state['active_trades'].remove(sell)

            elif CONFIG['mode'] == 'signal':
                signal = get_signal(df)
                if buy_price:
                    stop = buy_price * (1 - CONFIG['stop_loss_pct'])
                    if price < stop:
                        logger.warning(f"STOP-LOSS hit: {price:.2f} < {stop:.2f}")
                        place_signal_order("SELL")
                        continue
                if should_trade(signal):
                    place_signal_order(signal)

            save_state(state)
            time.sleep(60)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
