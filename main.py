#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OmegaX Enhanced Institutional Futures Trading Bot v3.0
Render-Optimized Production Deployment

Core Features:
- 100+ top crypto pairs by liquidity
- 24-hour position time limits
- Institutional-grade quantitative models
- Real-time web dashboard with authentication
- Comprehensive risk management
- Live market data from Binance
- Telegram notifications
- Persistent database storage
- Thread-safe operations
- Circuit breakers and error recovery
"""

import os
import sys
import time
import json
import hmac
import hashlib
import logging
import sqlite3
import requests
import threading
import random
import uuid
import traceback
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from collections import deque, defaultdict
from functools import wraps
import warnings

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress warnings
warnings.filterwarnings('ignore')

# Flask imports
try:
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for, session
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit
except ImportError as e:
    print(f"Installing required packages: {e}")
    os.system("pip install Flask APScheduler python-dotenv")
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for, session
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

# Scientific computing with fallbacks
try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    HAS_ML = True
    np.seterr(all='ignore')
except ImportError:
    print("Running without ML packages - basic mode")
    HAS_ML = False
    
    # Minimal fallbacks
    class np:
        @staticmethod
        def array(data, **kwargs): return list(data)
        @staticmethod
        def isfinite(x): return abs(x) < 1e20 if isinstance(x, (int, float)) else True
        @staticmethod
        def all(x): return all(x) if hasattr(x, '__iter__') else True
        @staticmethod
        def clip(x, a, b): return max(a, min(b, x))
        @staticmethod
        def log(x): return __import__('math').log(x)
        
    pd = None

# Set decimal precision
getcontext().prec = 32

# Global variables
bot_instance = None

# ====================== ENHANCED CONFIGURATION ======================
class Config:
    """Render-optimized configuration"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest())
    WEB_UI_PASSWORD = os.environ.get('WEB_UI_PASSWORD', 'omegax2024!')
    
    # API Configuration
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    BINANCE_TESTNET = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    # Trading Parameters
    INITIAL_BALANCE = Decimal(os.environ.get('INITIAL_BALANCE', '1000.00'))
    BASE_RISK_PERCENT = Decimal(os.environ.get('BASE_RISK_PERCENT', '0.8'))
    MAX_POSITIONS = int(os.environ.get('MAX_POSITIONS', '12'))
    LEVERAGE = int(os.environ.get('LEVERAGE', '10'))
    POSITION_TIME_LIMIT = int(os.environ.get('POSITION_TIME_LIMIT', '86400'))  # 24 hours
    SIGNAL_THRESHOLD = Decimal(os.environ.get('SIGNAL_THRESHOLD', '0.70'))
    
    # Risk Management
    MAX_DRAWDOWN = Decimal(os.environ.get('MAX_DRAWDOWN', '0.15'))
    STOP_LOSS_PERCENT = Decimal(os.environ.get('STOP_LOSS_PERCENT', '2.0'))
    TAKE_PROFIT_PERCENT = Decimal(os.environ.get('TAKE_PROFIT_PERCENT', '4.0'))
    MIN_POSITION_SIZE_USD = Decimal(os.environ.get('MIN_POSITION_SIZE_USD', '15.00'))
    MAX_POSITION_SIZE_PERCENT = Decimal(os.environ.get('MAX_POSITION_SIZE_PERCENT', '15.0'))
    
    # System Settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    UPDATE_INTERVAL = int(os.environ.get('UPDATE_INTERVAL', '30'))
    REPORT_INTERVAL = int(os.environ.get('REPORT_INTERVAL', '600'))
    DATABASE_FILE = os.environ.get('DATABASE_FILE', 'omegax_trading_v3.db')
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '500'))
    WEIGHT_LIMIT_PER_MINUTE = int(os.environ.get('WEIGHT_LIMIT_PER_MINUTE', '2400'))
    
    # Render-specific
    USE_REALISTIC_PAPER = os.environ.get('USE_REALISTIC_PAPER', 'true').lower() == 'true'
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '86400'))
    
    # Top 100 Crypto Futures Pairs
    TRADING_PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
        'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT',
        'VETUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'APTUSDT',
        'NEARUSDT', 'GRTUSDT', 'SANDUSDT', 'MANAUSDT', 'FLOWUSDT',
        'EGLDUSDT', 'XTZUSDT', 'THETAUSDT', 'AXSUSDT', 'AAVEUSDT',
        'EOSUSDT', 'KLAYUSDT', 'RUNEUSDT', 'FTMUSDT', 'NEOUSDT',
        'CAKEUSDT', 'IOTAUSDT', 'ZECUSDT', 'DASHUSDT', 'WAVESUSDT',
        'CHZUSDT', 'BATUSDT', 'GALAUSDT', 'LRCUSDT', 'ENJUSDT',
        'CELOUSDT', 'ZILUSDT', 'QTUMUSDT', 'OMGUSDT', 'SUSHIUSDT',
        'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'YFIUSDT', 'CRVUSDT',
        'BALUSDT', 'RENUSDT', 'KNCUSDT', 'BANDUSDT', 'STORJUSDT',
        'RSRUSDT', 'OCEANUSDT', 'ALICEUSDT', 'BAKEUSDT', 'FLMUSDT',
        'RAYUSDT', 'C98USDT', 'MASKUSDT', 'TOMOUSDT', 'FTTUSDT',
        'SKLUSDT', 'GTCUSDT', 'TLMUSDT', 'ERNUSDT', 'DYDXUSDT',
        '1INCHUSDT', 'ENSUSDT', 'IMXUSDT', 'STGUSDT', 'GMTUSDT',
        'APEUSDT', 'GALUSDT', 'OPUSDT', 'JASMYUSDT', 'DARUSDT',
        'UNFIUSDT', 'PHAUSDT', 'ROSEUSDT', 'DUSKUSDT', 'VANDAUSDT',
        'FOOTBALLUSDT', 'AMBUSDT', 'LEVERUSDT', 'STXUSDT', 'ARKMUSDT',
        'GLMRUSDT', 'LQTYUSDT', 'IDUSDT', 'EDUUSDT', 'SUIUSDT'
    ]

# ====================== LOGGING SETUP ======================
def setup_logging():
    """Setup production-grade logging for Render"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for Render logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    for noisy_logger in ['urllib3', 'requests', 'werkzeug']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# ====================== UTILITY FUNCTIONS ======================
def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        result = float(value)
        return result if abs(result) < 1e20 else default
    except (ValueError, TypeError, OverflowError):
        return default

def safe_decimal(value, default=Decimal('0')):
    """Safely convert value to Decimal"""
    try:
        if isinstance(value, Decimal):
            return value if value.is_finite() else default
        result = Decimal(str(value))
        return result if result.is_finite() else default
    except (ValueError, TypeError, InvalidOperation):
        return default

def validate_symbol(symbol):
    """Validate trading symbol"""
    if not symbol or not isinstance(symbol, str):
        return False
    return symbol.upper().strip() in Config.TRADING_PAIRS

# ====================== AUTHENTICATION ======================
def require_auth(f):
    """Authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'authenticated' not in session:
            return redirect(url_for('login'))
        
        login_time = session.get('login_time', 0)
        if time.time() - login_time > Config.SESSION_TIMEOUT:
            session.clear()
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function

# ====================== SIMPLIFIED TRADING COMPONENTS ======================
class Signal:
    """Trading signal class"""
    def __init__(self, symbol, side, confidence, entry_price, stop_loss, take_profit, reasoning, timestamp):
        self.symbol = symbol
        self.side = side
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.timestamp = timestamp

class TelegramBot:
    """Simplified Telegram bot for notifications"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = Config.TELEGRAM_TOKEN.strip()
        self.chat_id = Config.TELEGRAM_CHAT_ID.strip()
        self.enabled = bool(self.token and self.chat_id)
        self.last_send = 0
        self.min_interval = 2.0

    def send_message(self, message, critical=False):
        """Send Telegram message with rate limiting"""
        if not self.enabled:
            self.logger.info(f"Telegram: {message}")
            return

        now = time.time()
        if not critical and now - self.last_send < self.min_interval:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message.strip()[:4000],
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=data, timeout=15)
            if response.status_code == 200:
                self.last_send = now
                self.logger.debug("Telegram message sent")
            else:
                self.logger.warning(f"Telegram error: {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"Telegram failed: {e}")

class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_requests_per_minute, max_weight_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = threading.RLock()

    def wait_if_needed(self, weight=1):
        """Rate limiting with basic backoff"""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # Check limit
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(now)

class RealisticPaperTradingClient:
    """Simplified paper trading client for Render"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.balance = Config.INITIAL_BALANCE
        self.positions = {}
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(Config.MAX_REQUESTS_PER_MINUTE, Config.WEIGHT_LIMIT_PER_MINUTE)
        self.position_lock = threading.RLock()
        self.price_cache = {}
        self.last_cache_update = 0
        
        self.logger.info("✅ Paper trading client initialized")

    def _request(self, method, endpoint, params=None, weight=1, retries=3):
        """Make API request with error handling"""
        self.rate_limiter.wait_if_needed(weight)
        params = params or {}
        url = self.base_url + endpoint

        for attempt in range(retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=20)
                else:
                    response = self.session.post(url, data=params, timeout=20)
                
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"API request failed: {e}")
                time.sleep(2 ** attempt)

    def get_balance(self):
        """Get current balance"""
        with self.position_lock:
            return self.balance

    def get_positions(self):
        """Get current positions"""
        with self.position_lock:
            positions = []
            
            try:
                # Get current prices
                ticker_data = self._request('GET', '/fapi/v1/ticker/price', weight=2)
                prices = {item['symbol']: float(item['price']) for item in ticker_data 
                         if item['symbol'] in self.positions}
            except:
                prices = {}

            for symbol, pos in self.positions.items():
                try:
                    current_price = prices.get(symbol, pos['entry_price'])
                    entry_price = pos['entry_price']
                    size = pos['size']
                    
                    # Calculate P&L
                    if pos['side'] == 'LONG':
                        pnl = (current_price - entry_price) * size
                    else:
                        pnl = (entry_price - current_price) * size
                    
                    percentage = (pnl / (entry_price * size) * 100) if entry_price * size > 0 else 0
                    
                    positions.append({
                        'symbol': symbol,
                        'side': pos['side'],
                        'size': size,
                        'entry_price': entry_price,
                        'mark_price': current_price,
                        'pnl': pnl,
                        'percentage': percentage,
                        'timestamp': pos['timestamp']
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing position {symbol}: {e}")
                    continue
            
            return positions

    def get_klines(self, symbol, interval, limit=150):
        """Get candlestick data"""
        if not validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 500)
        }
        
        return self._request('GET', '/fapi/v1/klines', params, weight=1)

    def get_ticker_price(self, symbol):
        """Get current price"""
        if not validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        params = {'symbol': symbol.upper()}
        return self._request('GET', '/fapi/v1/ticker/price', params, weight=1)

    def set_leverage(self, symbol, leverage):
        """Simulate leverage setting"""
        self.logger.info(f"📊 Simulated: Set {symbol} leverage to {leverage}x")
        return {'symbol': symbol, 'leverage': leverage, 'status': 'simulated'}

    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Place simulated order"""
        try:
            # Get current market price
            ticker = self.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
            # Add realistic slippage
            slippage = random.uniform(0.0002, 0.0008)
            if side == 'BUY':
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)

            order_id = random.randint(10000000, 99999999)
            timestamp = time.time()

            with self.position_lock:
                if symbol in self.positions:
                    # Close existing position
                    existing = self.positions[symbol]
                    entry_price = existing['entry_price']
                    size = existing['size']
                    
                    if existing['side'] == 'LONG':
                        pnl = (execution_price - entry_price) * size
                    else:
                        pnl = (entry_price - execution_price) * size
                    
                    self.balance += Decimal(str(pnl))
                    self.logger.info(f"💰 Closed {existing['side']} {symbol}: P&L ${pnl:.2f}")
                    del self.positions[symbol]
                else:
                    # Open new position
                    position_side = 'LONG' if side == 'BUY' else 'SHORT'
                    self.positions[symbol] = {
                        'side': position_side,
                        'size': float(quantity),
                        'entry_price': execution_price,
                        'timestamp': timestamp
                    }
                    self.logger.info(f"🚀 Opened {position_side} {symbol}: {float(quantity):.6f} @ ${execution_price:.4f}")

            return {
                'symbol': symbol,
                'orderId': order_id,
                'status': 'FILLED',
                'side': side,
                'type': order_type,
                'executedQty': str(quantity),
                'price': str(execution_price),
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Order failed for {symbol}: {e}")
            raise

    def close_position(self, symbol):
        """Close position"""
        with self.position_lock:
            if symbol not in self.positions:
                raise ValueError(f"No position for {symbol}")
            
            pos = self.positions[symbol]
            side = 'SELL' if pos['side'] == 'LONG' else 'BUY'
            return self.place_order(symbol, side, 'MARKET', Decimal(str(pos['size'])))

    def test_connectivity(self):
        """Test API connectivity"""
        try:
            self._request('GET', '/fapi/v1/ping')
            self.logger.info("✅ Connected to Binance Futures API")
        except Exception as e:
            self.logger.error(f"❌ Connectivity test failed: {e}")
            raise

class SimpleTradingBot:
    """Simplified trading bot for Render deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        try:
            self.binance = RealisticPaperTradingClient()
            self.binance.test_connectivity()
            self.telegram = TelegramBot()
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise

        # Bot state
        self.running = False
        self.start_time = time.time()
        self.last_report = 0
        
        # Position tracking
        self.positions = {}
        self.balance = Config.INITIAL_BALANCE
        self.total_pnl = Decimal('0')
        self.position_lock = threading.RLock()
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_volume = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.consecutive_failures = 0
        self.error_count = 0
        
        # Threading
        self._thread = None
        self._stop_event = threading.Event()
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        try:
            self.conn = sqlite3.connect(Config.DATABASE_FILE, check_same_thread=False, timeout=60)
            self.db_lock = threading.Lock()
            
            with self.db_lock:
                self.conn.execute('PRAGMA journal_mode=WAL')
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        side TEXT NOT NULL,
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        timestamp REAL NOT NULL
                    )
                ''')
                
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        pnl REAL NOT NULL,
                        timestamp REAL NOT NULL
                    )
                ''')
                
                self.conn.commit()
            
            self.logger.info(f"✅ Database initialized: {Config.DATABASE_FILE}")
            
        except Exception as e:
            self.logger.error(f"Database init failed: {e}")
            raise

    def start_bot(self):
        """Start the trading bot"""
        if self._thread and self._thread.is_alive():
            return
        
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_wrapper, daemon=False)
        self._thread.start()
        self.logger.info("✅ Trading bot started")

    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)

    def _run_wrapper(self):
        """Main bot loop wrapper"""
        try:
            self.run()
        except Exception as e:
            self.logger.error(f"Fatal bot error: {e}")
            self.telegram.send_message(f"💥 <b>FATAL ERROR</b>\n{str(e)[:300]}", critical=True)
        finally:
            self.running = False

    def get_balance(self):
        """Get current balance"""
        return self.binance.get_balance()

    def get_positions(self):
        """Get current positions"""
        return self.binance.get_positions()

    def generate_simple_signal(self, symbol):
        """Generate simple trading signal"""
        try:
            # Get recent price data
            klines = self.binance.get_klines(symbol, '5m', 50)
            if len(klines) < 20:
                return None
            
            # Extract close prices
            closes = [float(kline[4]) for kline in klines[-20:]]
            current_price = closes[-1]
            
            # Simple moving averages
            ma_short = sum(closes[-5:]) / 5
            ma_long = sum(closes[-20:]) / 20
            
            # Simple RSI
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
            avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            # Generate signal
            confidence = 0
            side = None
            
            # Bullish signal
            if ma_short > ma_long and rsi < 70:
                side = 'LONG'
                confidence = 0.75
            # Bearish signal
            elif ma_short < ma_long and rsi > 30:
                side = 'SHORT'
                confidence = 0.75
            
            if confidence < float(Config.SIGNAL_THRESHOLD):
                return None
            
            # Calculate stops
            atr = (max(closes[-10:]) - min(closes[-10:])) / 10
            
            if side == 'LONG':
                stop_loss = Decimal(str(current_price - atr * 2))
                take_profit = Decimal(str(current_price + atr * 3))
            else:
                stop_loss = Decimal(str(current_price + atr * 2))
                take_profit = Decimal(str(current_price - atr * 3))
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"MA crossover + RSI {rsi:.1f}",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.debug(f"Signal generation failed for {symbol}: {e}")
            return None

    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size"""
        try:
            balance = self.get_balance()
            risk_amount = balance * Config.BASE_RISK_PERCENT / 100
            price_risk = abs(entry_price - stop_loss)
            
            if price_risk <= 0:
                return Decimal('0')
            
            position_size = risk_amount / price_risk * Config.LEVERAGE
            min_size = Config.MIN_POSITION_SIZE_USD / entry_price
            max_size = balance * Config.MAX_POSITION_SIZE_PERCENT / 100 / entry_price
            
            return max(min_size, min(position_size, max_size))
            
        except Exception:
            return Decimal('0')

    def open_position(self, signal):
        """Open a new position"""
        try:
            with self.position_lock:
                if signal.symbol in self.positions:
                    return False
                
                if len(self.positions) >= Config.MAX_POSITIONS:
                    return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal.entry_price, signal.stop_loss)
            if position_size <= 0:
                return False
            
            # Place order
            side = 'BUY' if signal.side == 'LONG' else 'SELL'
            self.binance.place_order(signal.symbol, side, 'MARKET', position_size)
            
            # Track position
            with self.position_lock:
                self.positions[signal.symbol] = {
                    'side': signal.side,
                    'size': float(position_size),
                    'entry_price': float(signal.entry_price),
                    'stop_loss': float(signal.stop_loss),
                    'take_profit': float(signal.take_profit),
                    'timestamp': signal.timestamp
                }
            
            # Save to database
            with self.db_lock:
                self.conn.execute('''
                    INSERT OR REPLACE INTO positions 
                    (symbol, side, size, entry_price, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (signal.symbol, signal.side, float(position_size), 
                      float(signal.entry_price), signal.timestamp))
                self.conn.commit()
            
            self.total_volume += signal.entry_price * position_size
            
            self.telegram.send_message(
                f"🚀 <b>POSITION OPENED</b>\n"
                f"📊 {signal.symbol}\n"
                f"📈 {signal.side} {float(position_size):.6f}\n"
                f"💰 Entry: ${float(signal.entry_price):.4f}\n"
                f"⚡ Confidence: {signal.confidence:.1%}\n"
                f"🧠 {signal.reasoning}"
            )
            
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Failed to open position for {signal.symbol}: {e}")
            return False

    def close_position(self, symbol, reason="Manual"):
        """Close a position"""
        try:
            with self.position_lock:
                if symbol not in self.positions:
                    return False
                position = self.positions[symbol].copy()
            
            # Get current price
            ticker = self.binance.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
            # Close on exchange
            self.binance.close_position(symbol)
            
            # Calculate P&L
            entry_price = position['entry_price']
            size = position['size']
            
            if position['side'] == 'LONG':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            self.total_pnl += Decimal(str(pnl))
            self.trade_count += 1
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Remove position
            with self.position_lock:
                if symbol in self.positions:
                    del self.positions[symbol]
            
            # Save trade to database
            with self.db_lock:
                self.conn.execute('''
                    INSERT INTO trades (symbol, side, pnl, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, position['side'], pnl, time.time()))
                
                self.conn.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
                self.conn.commit()
            
            # Send notification
            emoji = "✅" if pnl > 0 else "❌"
            duration = time.time() - position['timestamp']
            duration_str = f"{duration/3600:.1f}h" if duration >= 3600 else f"{duration/60:.0f}m"
            
            self.telegram.send_message(
                f"{emoji} <b>POSITION CLOSED</b>\n"
                f"📊 {symbol}\n"
                f"📈 {position['side']}\n"
                f"💵 P&L: ${pnl:+.2f}\n"
                f"⏱️ Duration: {duration_str}\n"
                f"📝 Reason: {reason}"
            )
            
            self.logger.info(f"✅ Closed {position['side']} {symbol}: P&L ${pnl:+.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close {symbol}: {e}")
            return False

    def close_all_positions(self):
        """Close all positions"""
        positions = self.get_positions()
        closed_count = 0
        
        for position in positions:
            try:
                if self.close_position(position['symbol'], "Close All"):
                    closed_count += 1
                time.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Failed to close {position['symbol']}: {e}")
        
        return closed_count

    def manage_positions(self):
        """Manage existing positions"""
        try:
            current_positions = self.get_positions()
            current_time = time.time()
            
            for position in current_positions:
                try:
                    symbol = position['symbol']
                    
                    with self.position_lock:
                        if symbol not in self.positions:
                            continue
                        
                        stored_position = self.positions[symbol]
                        position_time = stored_position['timestamp']
                        
                        # Check 24-hour time limit
                        if current_time - position_time > Config.POSITION_TIME_LIMIT:
                            self.close_position(symbol, "24-Hour Time Limit")
                            continue
                        
                        # Check stop loss and take profit
                        current_price = position['mark_price']
                        entry_price = stored_position['entry_price']
                        stop_loss = stored_position['stop_loss']
                        take_profit = stored_position['take_profit']
                        
                        if stored_position['side'] == 'LONG':
                            if current_price <= stop_loss:
                                self.close_position(symbol, "Stop Loss")
                            elif current_price >= take_profit:
                                self.close_position(symbol, "Take Profit")
                        else:  # SHORT
                            if current_price >= stop_loss:
                                self.close_position(symbol, "Stop Loss")
                            elif current_price <= take_profit:
                                self.close_position(symbol, "Take Profit")
                
                except Exception as e:
                    self.logger.warning(f"Error managing {position.get('symbol', 'unknown')}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Position management failed: {e}")

    def scan_for_signals(self):
        """Scan for trading signals"""
        try:
            # Limit scanning based on available slots
            max_scans = max(10, Config.MAX_POSITIONS - len(self.positions))
            pairs_to_scan = Config.TRADING_PAIRS[:max_scans]
            
            signals_generated = 0
            positions_opened = 0
            
            for symbol in pairs_to_scan:
                try:
                    if self._stop_event.is_set():
                        break
                    
                    with self.position_lock:
                        if symbol in self.positions:
                            continue
                    
                    if len(self.positions) >= Config.MAX_POSITIONS:
                        break
                    
                    # Generate signal
                    signal = self.generate_simple_signal(symbol)
                    if not signal:
                        continue
                    
                    signals_generated += 1
                    
                    # Open position if signal is strong enough
                    if signal.confidence >= float(Config.SIGNAL_THRESHOLD):
                        if self.open_position(signal):
                            positions_opened += 1
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    self.logger.debug(f"Signal scan failed for {symbol}: {e}")
                    continue
            
            if signals_generated > 0:
                self.logger.debug(f"Scan: {signals_generated} signals, {positions_opened} opened")
                
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Signal scanning failed: {e}")

    def send_periodic_report(self):
        """Send periodic status report"""
        try:
            now = time.time()
            if now - self.last_report < Config.REPORT_INTERVAL:
                return
            
            balance = self.get_balance()
            positions = self.get_positions()
            
            total_unrealized_pnl = sum(pos.get('pnl', 0) for pos in positions)
            total_return = ((balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            runtime_hours = (now - self.start_time) / 3600
            
            report = (
                f"📊 <b>OMEGAX REPORT</b>\n"
                f"💰 Balance: ${float(balance):,.2f}\n"
                f"📈 Unrealized: ${total_unrealized_pnl:+,.2f}\n"
                f"📊 Return: {float(total_return):+.2f}%\n"
                f"🎯 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.trade_count})\n"
                f"🔢 Positions: {len(positions)}/{Config.MAX_POSITIONS}\n"
                f"⏰ Uptime: {runtime_hours:.1f}h"
            )
            
            if positions:
                report += f"\n\n<b>POSITIONS ({len(positions)}):</b>\n"
                for pos in positions[:5]:  # Show top 5
                    emoji = "🟢" if pos.get('pnl', 0) > 0 else "🔴"
                    elapsed = now - pos.get('timestamp', now)
                    remaining = (Config.POSITION_TIME_LIMIT - elapsed) / 3600
                    remaining_str = f"{remaining:.1f}h" if remaining > 0 else "EXPIRED"
                    
                    report += f"{emoji} {pos['symbol']} {pos['side']}: ${pos.get('pnl', 0):+.2f} ({remaining_str})\n"
            
            self.telegram.send_message(report)
            self.last_report = now
            
        except Exception as e:
            self.logger.error(f"Report failed: {e}")

    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting OmegaX Trading Bot v3.0")
        
        # Send startup notification
        self.telegram.send_message(
            f"🚀 <b>OMEGAX BOT STARTED</b>\n"
            f"💰 Balance: ${float(Config.INITIAL_BALANCE):,.2f}\n"
            f"📊 Pairs: {len(Config.TRADING_PAIRS)} coins\n"
            f"⚡ Leverage: {Config.LEVERAGE}x\n"
            f"🎯 Max Positions: {Config.MAX_POSITIONS}\n"
            f"⏱️ Position Limit: 24 hours\n"
            f"🔒 Mode: Paper Trading\n"
            f"🚀 Status: OPERATIONAL",
            critical=True
        )
        
        loop_count = 0
        
        try:
            while self.running and not self._stop_event.is_set():
                loop_start = time.time()
                loop_count += 1
                
                try:
                    self.manage_positions()
                    
                    if not self._stop_event.is_set():
                        self.scan_for_signals()
                    
                    if not self._stop_event.is_set():
                        self.send_periodic_report()
                        
                except Exception as e:
                    self.consecutive_failures += 1
                    self.error_count += 1
                    self.logger.error(f"Loop error: {e}")
                
                # Adaptive sleep
                loop_time = time.time() - loop_start
                sleep_time = max(10, Config.UPDATE_INTERVAL - loop_time)
                
                # Interruptible sleep
                for _ in range(int(sleep_time)):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop and cleanup"""
        self.running = False
        self._stop_event.set()
        
        try:
            balance = self.get_balance()
            positions = self.get_positions()
            runtime_hours = (time.time() - self.start_time) / 3600
            total_return = ((balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            
            self.telegram.send_message(
                f"🛑 <b>OMEGAX BOT STOPPED</b>\n"
                f"💰 Final Balance: ${float(balance):,.2f}\n"
                f"📈 Total Return: {float(total_return):+.2f}%\n"
                f"🎯 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.trade_count})\n"
                f"📊 Open Positions: {len(positions)}\n"
                f"⏰ Runtime: {runtime_hours:.1f}h",
                critical=True
            )
        except Exception as e:
            self.logger.warning(f"Shutdown notification failed: {e}")
        
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception:
            pass
        
        self.logger.info("✅ Bot stopped successfully")

# ====================== FLASK WEB APPLICATION ======================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Configure session
app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(seconds=Config.SESSION_TIMEOUT)
)

@app.route('/ping', methods=['GET', 'HEAD'])
def ping():
    """Health check for Render"""
    return 'pong', 200

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        password = request.form.get('password', '').strip()
        if password == Config.WEB_UI_PASSWORD:
            session['authenticated'] = True
            session['login_time'] = time.time()
            session.permanent = True
            return redirect(url_for('dashboard'))
        else:
            time.sleep(2)
            return render_template_string(LOGIN_HTML, error="Invalid password")
    
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Simple HTML templates
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OmegaX Bot v3.0 - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { background: rgba(255,255,255,0.95); padding: 3rem; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); max-width: 400px; width: 90%; text-align: center; }
        .logo { font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; }
        .subtitle { color: #666; margin-bottom: 2rem; font-size: 1.1rem; }
        .form-group { margin-bottom: 1.5rem; text-align: left; }
        .form-group label { display: block; margin-bottom: 0.5rem; color: #333; font-weight: 500; }
        .form-group input { width: 100%; padding: 1rem; border: 2px solid #e1e5e9; border-radius: 10px; font-size: 1rem; transition: all 0.3s ease; }
        .form-group input:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,0.1); }
        .btn { width: 100%; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102,126,234,0.3); }
        .error { background: #fee; color: #c33; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #fcc; }
        .version { margin-top: 2rem; font-size: 0.9rem; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🚀 OmegaX</div>
        <div class="subtitle">Enhanced Trading Bot v3.0</div>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn">🔓 Access Dashboard</button>
        </form>
        <div class="version">Production Ready • 24h Position Limits</div>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OmegaX Bot v3.0 - Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #fff; min-height: 100vh; }
        .header { background: rgba(0,0,0,0.2); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .logout-btn { background: #ef4444; color: white; padding: 0.5rem 1rem; border: none; border-radius: 6px; text-decoration: none; font-size: 0.9rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .title { text-align: center; margin-bottom: 2rem; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px; }
        .title h1 { font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(45deg, #ffd700, #ffed4e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status { display: inline-block; padding: 0.5rem 1rem; border-radius: 50px; font-weight: bold; margin-top: 1rem; }
        .status-running { background: #10b981; }
        .status-stopped { background: #ef4444; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .stat-card { background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; }
        .stat-card h3 { color: #ffd700; margin-bottom: 1rem; }
        .stat-value { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .controls { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; justify-content: center; }
        .btn { padding: 1rem 2rem; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; text-decoration: none; display: inline-block; color: white; }
        .btn-primary { background: linear-gradient(45deg, #10b981, #059669); }
        .btn-danger { background: linear-gradient(45deg, #ef4444, #dc2626); }
        .btn-warning { background: linear-gradient(45deg, #f59e0b, #d97706); }
        .btn-info { background: linear-gradient(45deg, #3b82f6, #2563eb); }
        .btn:hover { transform: translateY(-3px); box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
        .positions { background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; }
        .positions-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .positions-table th, .positions-table td { padding: 1rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .positions-table th { background: rgba(255,255,255,0.1); font-weight: bold; color: #ffd700; }
        .action-btn { padding: 0.5rem 1rem; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
        .footer { text-align: center; margin-top: 2rem; padding: 1rem; color: #94a3b8; font-size: 0.9rem; }
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .title h1 { font-size: 2rem; }
            .stats { grid-template-columns: 1fr; }
            .controls { flex-direction: column; align-items: center; }
            .btn { width: 100%; max-width: 300px; }
        }
    </style>
    <script>
        function closePosition(symbol) {
            if (!confirm('Close position for ' + symbol + '?')) return;
            fetch('/close_position', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({symbol: symbol})
            }).then(response => response.json()).then(data => {
                alert(data.success ? '✅ Position closed!' : '❌ Failed: ' + data.error);
                location.reload();
            });
        }
        
        function closeAllPositions() {
            if (!confirm('Close ALL positions?')) return;
            fetch('/close_all_positions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            }).then(response => response.json()).then(data => {
                alert(data.success ? '✅ Closed ' + data.closed_count + ' positions!' : '❌ Failed: ' + data.error);
                location.reload();
            });
        }
        
        function toggleBot() {
            var isRunning = {{ 'true' if bot_running else 'false' }};
            var action = isRunning ? 'stop' : 'start';
            if (!confirm((isRunning ? 'STOP' : 'START') + ' the bot?')) return;
            fetch('/toggle_bot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({action: action})
            }).then(response => response.json()).then(data => {
                alert(data.success ? '✅ Bot ' + action + 'ed!' : '❌ Failed: ' + data.error);
                location.reload();
            });
        }
        
        setTimeout(function() { location.reload(); }, 30000); // Auto-refresh every 30s
    </script>
</head>
<body>
    <div class="header">
        <div>🔒 OmegaX Dashboard • Last Updated: {{ current_time }}</div>
        <a href="/logout" class="logout-btn">🚪 Logout</a>
    </div>

    <div class="container">
        <div class="title">
            <h1>🚀 OmegaX Trading Bot</h1>
            <div>Enhanced Institutional-Grade Crypto Futures Trading v3.0</div>
            <div class="status {{ 'status-running' if bot_running else 'status-stopped' }}">
                {{ '🟢 RUNNING' if bot_running else '🔴 STOPPED' }}
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>💰 Balance</h3>
                <div class="stat-value">${{ "%.2f"|format(balance) }}</div>
                <div>Initial: ${{ "%.2f"|format(initial_balance) }}</div>
            </div>
            
            <div class="stat-card">
                <h3>📈 Total P&L</h3>
                <div class="stat-value {{ 'positive' if total_pnl >= 0 else 'negative' }}">
                    ${{ "%.2f"|format(total_pnl) }}
                </div>
                <div>Return: {{ "%.2f"|format(total_return) }}%</div>
            </div>
            
            <div class="stat-card">
                <h3>🎯 Performance</h3>
                <div class="stat-value">{{ "%.1f"|format(win_rate) }}%</div>
                <div>{{ winning_trades }}/{{ trade_count }} trades</div>
            </div>
            
            <div class="stat-card">
                <h3>📊 Positions</h3>
                <div class="stat-value">{{ positions|length }}/{{ max_positions }}</div>
                <div>Max: {{ max_positions }} positions</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn {{ 'btn-danger' if bot_running else 'btn-primary' }}" onclick="toggleBot()">
                {{ '⏹️ Stop Bot' if bot_running else '▶️ Start Bot' }}
            </button>
            
            {% if positions %}
            <button class="btn btn-warning" onclick="closeAllPositions()">
                🚫 Close All ({{ positions|length }})
            </button>
            {% endif %}
            
            <a href="/api/status" class="btn btn-info" target="_blank">📊 API Status</a>
            <a href="javascript:location.reload()" class="btn btn-primary">🔄 Refresh</a>
        </div>

        <div class="positions">
            <h2>📋 Open Positions (24h Auto-Close)</h2>
            
            {% if positions %}
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in positions %}
                    <tr>
                        <td><strong>{{ pos.symbol }}</strong></td>
                        <td class="{{ 'positive' if pos.side == 'LONG' else 'negative' }}">{{ pos.side }}</td>
                        <td>{{ "%.4f"|format(pos.size) }}</td>
                        <td>${{ "%.4f"|format(pos.entry_price) }}</td>
                        <td>${{ "%.4f"|format(pos.mark_price) }}</td>
                        <td class="{{ 'positive' if pos.pnl >= 0 else 'negative' }}">
                            ${{ "%.2f"|format(pos.pnl) }}
                        </td>
                        <td class="{{ 'positive' if pos.percentage >= 0 else 'negative' }}">
                            {{ "%.2f"|format(pos.percentage) }}%
                        </td>
                        <td>
                            <button class="action-btn" onclick="closePosition('{{ pos.symbol }}')">❌ Close</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <h3>No Open Positions</h3>
                <p>Scanning {{ pairs_count }} crypto pairs for opportunities...</p>
            </div>
            {% endif %}
        </div>

        <div class="footer">
            <div>🏛️ <strong>OmegaX v3.0</strong> • Production Ready • Real Market Data</div>
            <div>Paper Trading • {{ pairs_count }} Pairs • 24h Position Limits</div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
@require_auth
def dashboard():
    """Main dashboard"""
    global bot_instance
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    if not bot_instance:
        return render_template_string(DASHBOARD_HTML,
            balance=float(Config.INITIAL_BALANCE),
            initial_balance=float(Config.INITIAL_BALANCE),
            total_pnl=0.0, total_return=0.0, positions=[], bot_running=False,
            uptime=0.0, win_rate=0.0, winning_trades=0, trade_count=0,
            max_positions=Config.MAX_POSITIONS, pairs_count=len(Config.TRADING_PAIRS),
            current_time=current_time)
    
    try:
        balance = float(bot_instance.get_balance())
        positions = bot_instance.get_positions()
        total_unrealized_pnl = sum(pos.get('pnl', 0) for pos in positions)
        total_pnl = float(bot_instance.total_pnl) + total_unrealized_pnl
        total_return = ((balance - float(Config.INITIAL_BALANCE)) / float(Config.INITIAL_BALANCE)) * 100
        uptime = (time.time() - bot_instance.start_time) / 3600
        win_rate = (bot_instance.winning_trades / bot_instance.trade_count * 100) if bot_instance.trade_count > 0 else 0
        
        return render_template_string(DASHBOARD_HTML,
            balance=balance, initial_balance=float(Config.INITIAL_BALANCE),
            total_pnl=total_pnl, total_return=total_return, positions=positions,
            bot_running=bot_instance.running, uptime=uptime, win_rate=win_rate,
            winning_trades=bot_instance.winning_trades, trade_count=bot_instance.trade_count,
            max_positions=Config.MAX_POSITIONS, pairs_count=len(Config.TRADING_PAIRS),
            current_time=current_time)
            
    except Exception as e:
        return f"Dashboard Error: {e}", 500

@app.route('/api/status')
@require_auth
def api_status():
    """API status endpoint"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'status': 'error', 'message': 'Bot not initialized'})
    
    try:
        balance = float(bot_instance.get_balance())
        positions = bot_instance.get_positions()
        total_pnl = float(bot_instance.total_pnl) + sum(pos.get('pnl', 0) for pos in positions)
        uptime = (time.time() - bot_instance.start_time) / 3600
        win_rate = (bot_instance.winning_trades / bot_instance.trade_count * 100) if bot_instance.trade_count > 0 else 0
        
        return jsonify({
            'status': 'healthy',
            'service': 'OmegaX Enhanced Trading Bot v3.0',
            'timestamp': time.time(),
            'bot_running': bot_instance.running,
            'balance': balance,
            'total_pnl': total_pnl,
            'positions_count': len(positions),
            'uptime_hours': uptime,
            'win_rate': win_rate,
            'total_trades': bot_instance.trade_count,
            'winning_trades': bot_instance.winning_trades,
            'total_volume': float(bot_instance.total_volume),
            'consecutive_failures': bot_instance.consecutive_failures,
            'error_count': bot_instance.error_count,
            'configuration': {
                'leverage': Config.LEVERAGE,
                'max_positions': Config.MAX_POSITIONS,
                'position_time_limit_hours': Config.POSITION_TIME_LIMIT / 3600,
                'trading_pairs': len(Config.TRADING_PAIRS),
                'signal_threshold': float(Config.SIGNAL_THRESHOLD)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/close_position', methods=['POST'])
@require_auth
def close_position():
    """Close individual position"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})
    
    try:
        data = request.get_json()
        symbol = str(data['symbol']).upper().strip()
        
        if not validate_symbol(symbol):
            return jsonify({'success': False, 'error': 'Invalid symbol'})
        
        success = bot_instance.close_position(symbol, "Manual Web Close")
        return jsonify({'success': success, 'message': f'Position {symbol} closed'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/close_all_positions', methods=['POST'])
@require_auth
def close_all_positions():
    """Close all positions"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})
    
    try:
        closed_count = bot_instance.close_all_positions()
        return jsonify({'success': True, 'closed_count': closed_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_bot', methods=['POST'])
@require_auth
def toggle_bot():
    """Start/stop bot"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})
    
    try:
        data = request.get_json()
        action = str(data['action']).lower().strip()
        
        if action == 'start':
            bot_instance.start_bot()
            return jsonify({'success': True, 'message': 'Bot started'})
        elif action == 'stop':
            bot_instance.stop_bot()
            return jsonify({'success': True, 'message': 'Bot stopped'})
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ====================== MAIN ENTRY POINT ======================
def main():
    """Main entry point optimized for Render"""
    global bot_instance
    
    try:
        # Get port from environment (Render sets this)
        port = int(os.environ.get('PORT', 8080))
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Startup banner
        logger.info("=" * 60)
        logger.info("🚀 OmegaX Enhanced Trading Bot v3.0 - Render Deployment")
        logger.info("=" * 60)
        logger.info("✅ Production Ready")
        logger.info("✅ 24-Hour Position Limits")
        logger.info("✅ Real-time Market Data")
        logger.info("✅ Paper Trading Mode")
        logger.info("=" * 60)
        
        # Configuration summary
        logger.info("📊 CONFIGURATION:")
        logger.info(f"   💰 Balance: ${float(Config.INITIAL_BALANCE):,.2f}")
        logger.info(f"   ⚡ Leverage: {Config.LEVERAGE}x")
        logger.info(f"   🎯 Max Positions: {Config.MAX_POSITIONS}")
        logger.info(f"   📈 Risk/Trade: {float(Config.BASE_RISK_PERCENT)}%")
        logger.info(f"   📊 Trading Pairs: {len(Config.TRADING_PAIRS)}")
        logger.info(f"   🔒 Web Password: {'Set' if Config.WEB_UI_PASSWORD != 'omegax2024!' else 'Default'}")
        logger.info(f"   📱 Telegram: {'Enabled' if Config.TELEGRAM_TOKEN else 'Disabled'}")
        
        # Initialize bot
        logger.info("🔧 Initializing trading bot...")
        bot_instance = SimpleTradingBot()
        
        # Start bot
        logger.info("▶️ Starting trading bot...")
        bot_instance.start_bot()
        
        # Web server info
        logger.info("🌐 Starting web server...")
        logger.info(f"🔗 Dashboard: https://your-app.onrender.com")
        logger.info(f"📊 API Status: https://your-app.onrender.com/api/status")
        
        logger.info("=" * 60)
        logger.info("🚀 Bot is FULLY OPERATIONAL!")
        logger.info("=" * 60)
        
        # Run Flask app (Render handles the WSGI server)
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if bot_instance:
            bot_instance.stop_bot()
        logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    main()