import os
import logging
from logging.handlers import RotatingFileHandler

class ColorFormatter(logging.Formatter):
    COLORS = {
        "BUY": "\033[92m",          # Green
        "SELL": "\033[91m",         # Red
        "HOLD": "\033[93m",         # Yellow
        "STOP-LOSS": "\033[95m",    # Magenta
        "CHoC": "\033[96m",         # Light Cyan
        "FVG": "\033[94m",          # Light Blue
        "Structure: 🟢": "\033[92m", # Bullish CHoC
        "Structure: 🔴": "\033[91m", # Bearish CHoC
        "Bullish Gap": "\033[92m",  # FVG Bull
        "Bearish Gap": "\033[91m",  # FVG Bear
        "Signal: HOLD": "\033[93m",
        "Signal: BUY": "\033[92m",
        "Signal: SELL": "\033[91m",
        "[INFO]": "\033[96m",       # Cyan
        "[WARNING]": "\033[95m",    # Magenta
        "[ERROR]": "\033[91m",      # Red
        "[DEBUG]": "\033[90m",      # Grey
        "RESET": "\033[0m"
    }

    def format(self, record):
        msg = super().format(record)
        for key, color in self.COLORS.items():
            if key in msg:
                return f"{color}{msg}{self.COLORS['RESET']}"
        return msg


def get_logger(name):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{name.lower()}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    # File handler
    fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
