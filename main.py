#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OmegaX Trading Bot v3.0 - Render Free Tier Optimized
Lightweight version without heavy ML dependencies
"""

import os
import sys
import time
import json
import logging
import sqlite3
import requests
import threading
import random
import traceback
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, InvalidOperation, ROUND_DOWN
from collections import deque
from functools import wraps
import warnings

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings('ignore')

# Flask imports
try:
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for, session
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit
except ImportError as e:
    print(f"Installing Flask: {e}")
    os.system("pip install Flask APScheduler")
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for, session
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

getcontext().prec = 32
bot_instance = None

# ====================== CONFIGURATION ======================
class Config:
    """Lightweight configuration for Render free tier"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 
        ''.join([str(random.randint(0,9)) for _ in range(32)]))
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
    DATABASE_FILE = os.environ.get('DATABASE_FILE', 'omegax.db')
    
    # Render optimized
    USE_REALISTIC_PAPER = os.environ.get('USE_REALISTIC_PAPER', 'true').lower() == 'true'
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '86400'))
    
    # Essential crypto pairs (reduced for faster processing)
    TRADING_PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
        'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT',
        'VETUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'APTUSDT'
    ]

# ====================== LIGHTWEIGHT MATH FUNCTIONS ======================
def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        result = float(value)
        return result if abs(result) < 1e20 else default
    except (ValueError, TypeError, OverflowError):
        return default

def safe_decimal(value, default=Decimal('0')):
    """Safely convert to Decimal"""
    try:
        if isinstance(value, Decimal):
            return value if value.is_finite() else default
        result = Decimal(str(value))
        return result if result.is_finite() else default
    except (ValueError, TypeError, InvalidOperation):
        return default

def calculate_sma(values, period):
    """Simple moving average"""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def calculate_rsi(prices, period=14):
    """Simple RSI calculation"""
    if len(prices) < period + 1:
        return 50
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, change) for change in changes[-period:]]
    losses = [max(0, -change) for change in changes[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def validate_symbol(symbol):
    """Validate trading symbol"""
    if not symbol or not isinstance(symbol, str):
        return False
    return symbol.upper().strip() in Config.TRADING_PAIRS

# ====================== SIMPLIFIED COMPONENTS ======================
def setup_logging():
    """Simple logging setup"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Quiet noisy loggers
    for noisy_logger in ['urllib3', 'requests', 'werkzeug']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

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

class Signal:
    """Simple signal class"""
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
    """Lightweight Telegram bot"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = Config.TELEGRAM_TOKEN.strip()
        self.chat_id = Config.TELEGRAM_CHAT_ID.strip()
        self.enabled = bool(self.token and self.chat_id)
        self.last_send = 0

    def send_message(self, message, critical=False):
        """Send message with rate limiting"""
        if not self.enabled:
            self.logger.info(f"Telegram: {message}")
            return

        now = time.time()
        if not critical and now - self.last_send < 2.0:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message.strip()[:4000],
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                self.last_send = now
                
        except Exception as e:
            self.logger.warning(f"Telegram failed: {e}")

class LightweightTradingClient:
    """Minimal trading client for Render free tier"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.balance = Config.INITIAL_BALANCE
        self.positions = {}
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'OmegaX-Bot/3.0'})
        self.position_lock = threading.RLock()
        
        self.logger.info("✅ Trading client ready")

    def _request(self, method, endpoint, params=None, retries=2):
        """Simple API request"""
        params = params or {}
        url = self.base_url + endpoint

        for attempt in range(retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=15)
                else:
                    response = self.session.post(url, data=params, timeout=15)
                
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"API failed: {e}")
                time.sleep(1)

    def get_balance(self):
        return self.balance

    def get_positions(self):
        """Get positions with current prices"""
        with self.position_lock:
            positions = []
            
            # Get current prices
            try:
                ticker_data = self._request('GET', '/fapi/v1/ticker/price')
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
                except Exception:
                    continue
            
            return positions

    def get_klines(self, symbol, interval, limit=50):
        """Get candlestick data"""
        params = {'symbol': symbol.upper(), 'interval': interval, 'limit': min(limit, 100)}
        return self._request('GET', '/fapi/v1/klines', params)

    def get_ticker_price(self, symbol):
        """Get current price"""
        params = {'symbol': symbol.upper()}
        return self._request('GET', '/fapi/v1/ticker/price', params)

    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Place simulated order"""
        try:
            # Get current price
            ticker = self.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
            # Add minimal slippage
            slippage = random.uniform(0.0001, 0.0005)
            if side == 'BUY':
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)

            timestamp = time.time()

            with self.position_lock:
                if symbol in self.positions:
                    # Close existing
                    existing = self.positions[symbol]
                    entry_price = existing['entry_price']
                    size = existing['size']
                    
                    if existing['side'] == 'LONG':
                        pnl = (execution_price - entry_price) * size
                    else:
                        pnl = (entry_price - execution_price) * size
                    
                    self.balance += Decimal(str(pnl))
                    self.logger.info(f"Closed {existing['side']} {symbol}: P&L ${pnl:.2f}")
                    del self.positions[symbol]
                else:
                    # Open new
                    position_side = 'LONG' if side == 'BUY' else 'SHORT'
                    self.positions[symbol] = {
                        'side': position_side,
                        'size': float(quantity),
                        'entry_price': execution_price,
                        'timestamp': timestamp
                    }
                    self.logger.info(f"Opened {position_side} {symbol}: {float(quantity):.6f} @ ${execution_price:.4f}")

            return {
                'symbol': symbol,
                'orderId': random.randint(10000000, 99999999),
                'status': 'FILLED',
                'side': side,
                'executedQty': str(quantity),
                'price': str(execution_price),
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Order failed: {e}")
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
        """Test connection"""
        try:
            self._request('GET', '/fapi/v1/ping')
            self.logger.info("✅ Connected to Binance API")
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            raise

class MinimalTradingBot:
    """Lightweight trading bot for Render free tier"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.binance = LightweightTradingClient()
            self.binance.test_connectivity()
            self.telegram = TelegramBot()
        except Exception as e:
            self.logger.error(f"Init failed: {e}")
            raise

        # State
        self.running = False
        self.start_time = time.time()
        self.last_report = 0
        self.positions = {}
        self.balance = Config.INITIAL_BALANCE
        self.total_pnl = Decimal('0')
        self.position_lock = threading.RLock()
        self.trade_count = 0
        self.winning_trades = 0
        self.total_volume = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.consecutive_failures = 0
        self.error_count = 0
        self._thread = None
        self._stop_event = threading.Event()
        
        self._init_database()

    def _init_database(self):
        """Simple database setup"""
        try:
            self.conn = sqlite3.connect(Config.DATABASE_FILE, check_same_thread=False, timeout=30)
            self.db_lock = threading.Lock()
            
            with self.db_lock:
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
            
            self.logger.info("✅ Database ready")
            
        except Exception as e:
            self.logger.error(f"Database failed: {e}")
            raise

    def start_bot(self):
        """Start bot"""
        if self._thread and self._thread.is_alive():
            return
        
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_wrapper, daemon=False)
        self._thread.start()
        self.logger.info("✅ Bot started")

    def stop_bot(self):
        """Stop bot"""
        self.running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)

    def _run_wrapper(self):
        """Run wrapper"""
        try:
            self.run()
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.telegram.send_message(f"💥 ERROR: {str(e)[:200]}", critical=True)
        finally:
            self.running = False

    def get_balance(self):
        return self.binance.get_balance()

    def get_positions(self):
        return self.binance.get_positions()

    def generate_simple_signal(self, symbol):
        """Lightweight signal generation"""
        try:
            klines = self.binance.get_klines(symbol, '5m', 30)
            if len(klines) < 20:
                return None
            
            closes = [float(kline[4]) for kline in klines[-20:]]
            current_price = closes[-1]
            
            # Simple strategy: MA crossover + RSI
            ma_short = calculate_sma(closes, 5)
            ma_long = calculate_sma(closes, 15)
            rsi = calculate_rsi(closes, 14)
            
            if not ma_short or not ma_long:
                return None
            
            # Signal logic
            confidence = 0
            side = None
            
            if ma_short > ma_long and rsi < 70:
                side = 'LONG'
                confidence = 0.75
            elif ma_short < ma_long and rsi > 30:
                side = 'SHORT'
                confidence = 0.75
            
            if confidence < float(Config.SIGNAL_THRESHOLD):
                return None
            
            # Simple stops
            price_range = max(closes) - min(closes)
            stop_distance = price_range * 0.02
            profit_distance = price_range * 0.04
            
            if side == 'LONG':
                stop_loss = Decimal(str(current_price - stop_distance))
                take_profit = Decimal(str(current_price + profit_distance))
            else:
                stop_loss = Decimal(str(current_price + stop_distance))
                take_profit = Decimal(str(current_price - profit_distance))
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"MA+RSI {rsi:.0f}",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.debug(f"Signal failed for {symbol}: {e}")
            return None

    def calculate_position_size(self, entry_price, stop_loss):
        """Simple position sizing"""
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
        """Open position"""
        try:
            with self.position_lock:
                if signal.symbol in self.positions:
                    return False
                
                if len(self.positions) >= Config.MAX_POSITIONS:
                    return False
            
            position_size = self.calculate_position_size(signal.entry_price, signal.stop_loss)
            if position_size <= 0:
                return False
            
            side = 'BUY' if signal.side == 'LONG' else 'SELL'
            self.binance.place_order(signal.symbol, side, 'MARKET', position_size)
            
            with self.position_lock:
                self.positions[signal.symbol] = {
                    'side': signal.side,
                    'size': float(position_size),
                    'entry_price': float(signal.entry_price),
                    'stop_loss': float(signal.stop_loss),
                    'take_profit': float(signal.take_profit),
                    'timestamp': signal.timestamp
                }
            
            self.total_volume += signal.entry_price * position_size
            
            self.telegram.send_message(
                f"🚀 OPENED {signal.symbol} {signal.side}\n"
                f"Size: {float(position_size):.6f}\n"
                f"Entry: ${float(signal.entry_price):.4f}\n"
                f"Confidence: {signal.confidence:.1%}"
            )
            
            return True
            
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Open failed {signal.symbol}: {e}")
            return False

    def close_position(self, symbol, reason="Manual"):
        """Close position"""
        try:
            with self.position_lock:
                if symbol not in self.positions:
                    return False
                position = self.positions[symbol].copy()
            
            ticker = self.binance.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
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
            
            # Save trade
            with self.db_lock:
                self.conn.execute('''
                    INSERT INTO trades (symbol, side, pnl, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, position['side'], pnl, time.time()))
                self.conn.commit()
            
            # Notify
            emoji = "✅" if pnl > 0 else "❌"
            self.telegram.send_message(
                f"{emoji} CLOSED {symbol} {position['side']}\n"
                f"P&L: ${pnl:+.2f}\n"
                f"Reason: {reason}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Close failed {symbol}: {e}")
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
            except Exception:
                pass
        
        return closed_count

    def manage_positions(self):
        """Manage positions"""
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
                        
                        # 24-hour limit
                        if current_time - position_time > Config.POSITION_TIME_LIMIT:
                            self.close_position(symbol, "24h Limit")
                            continue
                        
                        # Stop/profit checks
                        current_price = position['mark_price']
                        stop_loss = stored_position['stop_loss']
                        take_profit = stored_position['take_profit']
                        
                        if stored_position['side'] == 'LONG':
                            if current_price <= stop_loss:
                                self.close_position(symbol, "Stop Loss")
                            elif current_price >= take_profit:
                                self.close_position(symbol, "Take Profit")
                        else:
                            if current_price >= stop_loss:
                                self.close_position(symbol, "Stop Loss")
                            elif current_price <= take_profit:
                                self.close_position(symbol, "Take Profit")
                
                except Exception as e:
                    self.logger.warning(f"Manage error {position.get('symbol')}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Position management failed: {e}")

    def scan_for_signals(self):
        """Lightweight signal scanning"""
        try:
            max_scans = max(5, Config.MAX_POSITIONS - len(self.positions))
            pairs_to_scan = Config.TRADING_PAIRS[:max_scans]
            
            for symbol in pairs_to_scan:
                try:
                    if self._stop_event.is_set():
                        break
                    
                    with self.position_lock:
                        if symbol in self.positions:
                            continue
                    
                    if len(self.positions) >= Config.MAX_POSITIONS:
                        break
                    
                    signal = self.generate_simple_signal(symbol)
                    if signal and signal.confidence >= float(Config.SIGNAL_THRESHOLD):
                        self.open_position(signal)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    self.logger.debug(f"Scan failed {symbol}: {e}")
                    continue
                
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Scanning failed: {e}")

    def send_periodic_report(self):
        """Send reports"""
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
                f"📊 OMEGAX REPORT\n"
                f"Balance: ${float(balance):,.2f}\n"
                f"P&L: ${total_unrealized_pnl:+,.2f}\n"
                f"Return: {float(total_return):+.2f}%\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Positions: {len(positions)}/{Config.MAX_POSITIONS}\n"
                f"Uptime: {runtime_hours:.1f}h"
            )
            
            self.telegram.send_message(report)
            self.last_report = now
            
        except Exception as e:
            self.logger.error(f"Report failed: {e}")

    def run(self):
        """Main loop"""
        self.logger.info("🚀 Starting OmegaX Bot v3.0")
        
        self.telegram.send_message(
            f"🚀 OMEGAX STARTED\n"
            f"Balance: ${float(Config.INITIAL_BALANCE):,.2f}\n"
            f"Pairs: {len(Config.TRADING_PAIRS)}\n"
            f"Status: OPERATIONAL",
            critical=True
        )
        
        try:
            while self.running and not self._stop_event.is_set():
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
                
                # Sleep
                for _ in range(Config.UPDATE_INTERVAL):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Fatal: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop and cleanup"""
        self.running = False
        self._stop_event.set()
        
        try:
            self.telegram.send_message("🛑 BOT STOPPED", critical=True)
        except Exception:
            pass
        
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception:
            pass
        
        self.logger.info("✅ Bot stopped")

# ====================== FLASK APP ======================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '').strip()
        if password == Config.WEB_UI_PASSWORD:
            session['authenticated'] = True
            session['login_time'] = time.time()
            return redirect(url_for('dashboard'))
        else:
            return render_template_string(LOGIN_HTML, error="Invalid password")
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Minimal templates
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head><title>OmegaX Login</title>
<style>
body{font-family:Arial,sans-serif;background:#667eea;min-height:100vh;display:flex;align-items:center;justify-content:center;margin:0}
.container{background:#fff;padding:2rem;border-radius:10px;max-width:400px;width:90%;text-align:center}
.logo{font-size:2rem;margin-bottom:1rem;color:#667eea}
input{width:100%;padding:0.75rem;margin:0.5rem 0;border:1px solid #ddd;border-radius:5px}
.btn{width:100%;padding:0.75rem;background:#667eea;color:#fff;border:none;border-radius:5px;cursor:pointer}
.error{background:#fee;color:#c33;padding:0.75rem;margin:0.5rem 0;border-radius:5px}
</style></head>
<body>
<div class="container">
<div class="logo">🚀 OmegaX</div>
<h2>Trading Bot v3.0</h2>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<form method="POST">
<input type="password" name="password" placeholder="Password" required>
<button type="submit" class="btn">Login</button>
</form>
</div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head><title>OmegaX Dashboard</title>
<style>
body{font-family:Arial,sans-serif;background:linear-gradient(135deg,#1e3c72,#2a5298);color:#fff;margin:0;min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:2rem}
.header{background:rgba(0,0,0,0.2);padding:1rem 2rem;display:flex;justify-content:space-between;align-items:center}
.title{text-align:center;margin-bottom:2rem;padding:2rem;background:rgba(255,255,255,0.1);border-radius:10px}
.title h1{font-size:2.5rem;margin-bottom:1rem;color:#ffd700}
.status{display:inline-block;padding:0.5rem 1rem;border-radius:25px;font-weight:bold;margin-top:1rem}
.status-running{background:#10b981}.status-stopped{background:#ef4444}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-bottom:2rem}
.stat-card{background:rgba(255,255,255,0.1);padding:1.5rem;border-radius:10px}
.stat-card h3{color:#ffd700;margin-bottom:0.5rem}
.stat-value{font-size:1.8rem;font-weight:bold;margin-bottom:0.25rem}
.positive{color:#4ade80}.negative{color:#f87171}
.controls{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap;justify-content:center}
.btn{padding:0.75rem 1.5rem;border:none;border-radius:5px;cursor:pointer;text-decoration:none;color:#fff}
.btn-primary{background:#10b981}.btn-danger{background:#ef4444}.btn-warning{background:#f59e0b}
.positions{background:rgba(255,255,255,0.1);padding:1.5rem;border-radius:10px}
.positions-table{width:100%;border-collapse:collapse;margin-top:1rem}
.positions-table th,.positions-table td{padding:0.75rem;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1)}
.positions-table th{background:rgba(255,255,255,0.1);color:#ffd700}
.action-btn{padding:0.5rem 1rem;background:#ef4444;color:#fff;border:none;border-radius:3px;cursor:pointer}
.empty-state{text-align:center;padding:3rem;color:#94a3b8}
.logout-btn{background:#ef4444;color:#fff;padding:0.5rem 1rem;text-decoration:none;border-radius:5px}
</style>
<script>
function closePosition(symbol){if(!confirm('Close '+symbol+'?'))return;fetch('/close_position',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:symbol})}).then(r=>r.json()).then(d=>{alert(d.success?'✅ Closed!':'❌ Failed');location.reload()})}
function closeAll(){if(!confirm('Close ALL?'))return;fetch('/close_all_positions',{method:'POST',headers:{'Content-Type':'application/json'}}).then(r=>r.json()).then(d=>{alert(d.success?'✅ Closed '+d.closed_count:'❌ Failed');location.reload()})}
function toggleBot(){var running={{ 'true' if bot_running else 'false' }};var action=running?'stop':'start';if(!confirm((running?'STOP':'START')+' bot?'))return;fetch('/toggle_bot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:action})}).then(r=>r.json()).then(d=>{alert(d.success?'✅ Done!':'❌ Failed');location.reload()})}
setTimeout(()=>location.reload(),30000);
</script>
</head>
<body>
<div class="header">
<div>🚀 OmegaX Dashboard</div>
<a href="/logout" class="logout-btn">Logout</a>
</div>
<div class="container">
<div class="title">
<h1>🚀 OmegaX Bot</h1>
<div>Crypto Trading v3.0</div>
<div class="status {{ 'status-running' if bot_running else 'status-stopped' }}">
{{ '🟢 RUNNING' if bot_running else '🔴 STOPPED' }}
</div>
</div>
<div class="stats">
<div class="stat-card"><h3>💰 Balance</h3><div class="stat-value">${{ "%.2f"|format(balance) }}</div></div>
<div class="stat-card"><h3>📈 P&L</h3><div class="stat-value {{ 'positive' if total_pnl >= 0 else 'negative' }}">${{ "%.2f"|format(total_pnl) }}</div></div>
<div class="stat-card"><h3>🎯 Win Rate</h3><div class="stat-value">{{ "%.1f"|format(win_rate) }}%</div></div>
<div class="stat-card"><h3>📊 Positions</h3><div class="stat-value">{{ positions|length }}/{{ max_positions }}</div></div>
</div>
<div class="controls">
<button class="btn {{ 'btn-danger' if bot_running else 'btn-primary' }}" onclick="toggleBot()">{{ '⏹️ Stop' if bot_running else '▶️ Start' }}</button>
{% if positions %}<button class="btn btn-warning" onclick="closeAll()">🚫 Close All</button>{% endif %}
<a href="javascript:location.reload()" class="btn btn-primary">🔄 Refresh</a>
</div>
<div class="positions">
<h2>📋 Positions</h2>
{% if positions %}
<table class="positions-table">
<tr><th>Symbol</th><th>Side</th><th>P&L</th><th>Action</th></tr>
{% for pos in positions %}
<tr>
<td><strong>{{ pos.symbol }}</strong></td>
<td class="{{ 'positive' if pos.side == 'LONG' else 'negative' }}">{{ pos.side }}</td>
<td class="{{ 'positive' if pos.pnl >= 0 else 'negative' }}">${{ "%.2f"|format(pos.pnl) }}</td>
<td><button class="action-btn" onclick="closePosition('{{ pos.symbol }}')">❌</button></td>
</tr>
{% endfor %}
</table>
{% else %}
<div class="empty-state">
<h3>🔍 No Positions</h3>
<p>Scanning {{ pairs_count }} pairs...</p>
</div>
{% endif %}
</div>
</div>
</body>
</html>
"""

@app.route('/')
@require_auth
def dashboard():
    global bot_instance
    
    if not bot_instance:
        return render_template_string(DASHBOARD_HTML,
            balance=float(Config.INITIAL_BALANCE), total_pnl=0.0, positions=[], 
            bot_running=False, win_rate=0.0, max_positions=Config.MAX_POSITIONS,
            pairs_count=len(Config.TRADING_PAIRS))
    
    try:
        balance = float(bot_instance.get_balance())
        positions = bot_instance.get_positions()
        total_pnl = float(bot_instance.total_pnl) + sum(pos.get('pnl', 0) for pos in positions)
        win_rate = (bot_instance.winning_trades / bot_instance.trade_count * 100) if bot_instance.trade_count > 0 else 0
        
        return render_template_string(DASHBOARD_HTML,
            balance=balance, total_pnl=total_pnl, positions=positions,
            bot_running=bot_instance.running, win_rate=win_rate,
            max_positions=Config.MAX_POSITIONS, pairs_count=len(Config.TRADING_PAIRS))
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/close_position', methods=['POST'])
@require_auth
def close_position():
    global bot_instance
    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})
    
    try:
        data = request.get_json()
        symbol = str(data['symbol']).upper().strip()
        success = bot_instance.close_position(symbol, "Manual")
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/close_all_positions', methods=['POST'])
@require_auth
def close_all_positions():
    global bot_instance
    if not bot_instance:
        return jsonify({'success': False})
    
    try:
        closed_count = bot_instance.close_all_positions()
        return jsonify({'success': True, 'closed_count': closed_count})
    except Exception:
        return jsonify({'success': False})

@app.route('/toggle_bot', methods=['POST'])
@require_auth
def toggle_bot():
    global bot_instance
    if not bot_instance:
        return jsonify({'success': False})
    
    try:
        data = request.get_json()
        action = str(data['action']).lower()
        
        if action == 'start':
            bot_instance.start_bot()
        elif action == 'stop':
            bot_instance.stop_bot()
        
        return jsonify({'success': True})
    except Exception:
        return jsonify({'success': False})

# ====================== MAIN ======================
def main():
    global bot_instance
    
    try:
        port = int(os.environ.get('PORT', 8080))
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 OmegaX Bot v3.0 - Render Free Tier")
        logger.info(f"Pairs: {len(Config.TRADING_PAIRS)}")
        
        bot_instance = MinimalTradingBot()
        bot_instance.start_bot()
        
        logger.info(f"Starting on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)
    finally:
        if bot_instance:
            bot_instance.stop_bot()

if __name__ == "__main__":
    main()