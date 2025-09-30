from flask import Flask, render_template
import json
import csv
import os

app = Flask(__name__)

# Relative paths to the project root (biTrade/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PNL_STATE_PATH = os.path.join(ROOT_DIR, 'pnl_state.json')
TRADE_LOG_PATH = os.path.join(ROOT_DIR, 'trade_log.csv')

@app.route('/')
def dashboard():
    # Load PnL state
    try:
        with open(PNL_STATE_PATH, 'r') as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {'cumulative_pnl': 0.0, 'last_buy_price': 0.0}

    # Load trades
    trades = []
    try:
        with open(TRADE_LOG_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            trades = list(reader)[-50:]  # Last 50 trades
    except FileNotFoundError:
        pass

    return render_template('index.html', state=state, trades=trades)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
