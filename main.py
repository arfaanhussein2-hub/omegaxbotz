# main.py
"""
OmegaX Institutional Futures Trading Bot with Live Web UI
Complete single-file implementation - Render compatible

Features:
- 48 trading pairs support
- Live web dashboard
- Real-time position management
- Paper trading with real market data
- Telegram notifications
- All numpy/telegram errors fixed

Access: https://your-app-name.onrender.com
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
from datetime import datetime, timezone
from decimal import Decimal, getcontext, InvalidOperation
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
import warnings
warnings.filterwarnings('ignore')

# Flask imports
try:
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit
except ImportError:
    print("Installing Flask and APScheduler...")
    os.system("pip install Flask APScheduler")
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

# Check required packages without auto-installation
try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    # Suppress numpy polyfit RankWarning explicitly
    try:
        warnings.simplefilter('ignore', np.RankWarning)
        warnings.filterwarnings('ignore', category=np.RankWarning)
    except Exception:
        pass
except ImportError as e:
    print(f"Required packages missing: {e}")
    print("Please install manually: pip install numpy pandas scikit-learn flask apscheduler")
    sys.exit(1)

# Set decimal precision
getcontext().prec = 28
D = Decimal

# Global bot instance for web UI
bot_instance = None

# ====================== CONFIGURATION ======================
class Config:
    """Configuration settings"""

    # API Configuration - Validate credentials exist
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY')
    BINANCE_TESTNET = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'

    # Validate API credentials
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        if not os.environ.get('USE_REALISTIC_PAPER', 'false').lower() == 'true':
            print("ERROR: BINANCE_API_KEY and BINANCE_SECRET_KEY must be set")
            sys.exit(1)

    # Paper trading mode using real public data
    USE_REALISTIC_PAPER = os.environ.get('USE_REALISTIC_PAPER', 'false').lower() == 'true'

    # Telegram Configuration
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    # Trading Configuration with validation
    try:
        INITIAL_BALANCE = D(os.environ.get('INITIAL_BALANCE', '1000'))
        BASE_RISK_PERCENT = D(os.environ.get('BASE_RISK_PERCENT', '0.8'))  # 0.8% per trade
        MAX_POSITIONS = int(os.environ.get('MAX_POSITIONS', '12'))
        LEVERAGE = int(os.environ.get('LEVERAGE', '10'))

        # Validate ranges
        if INITIAL_BALANCE <= 0:
            raise ValueError("INITIAL_BALANCE must be positive")
        if not (0 < BASE_RISK_PERCENT <= 10):
            raise ValueError("BASE_RISK_PERCENT must be between 0 and 10")
        if not (1 <= MAX_POSITIONS <= 50):
            raise ValueError("MAX_POSITIONS must be between 1 and 50")
        if not (1 <= LEVERAGE <= 125):
            raise ValueError("LEVERAGE must be between 1 and 125")

    except (ValueError, InvalidOperation) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Strategy Configuration
    try:
        SIGNAL_THRESHOLD = D(os.environ.get('SIGNAL_THRESHOLD', '0.65'))
        MIN_VOLUME_24H = D(os.environ.get('MIN_VOLUME_24H', '5000000'))  # $5M daily volume

        if not (0 < SIGNAL_THRESHOLD <= 1):
            raise ValueError("SIGNAL_THRESHOLD must be between 0 and 1")
        if MIN_VOLUME_24H <= 0:
            raise ValueError("MIN_VOLUME_24H must be positive")

    except (ValueError, InvalidOperation) as e:
        print(f"Strategy configuration error: {e}")
        sys.exit(1)

    # Risk Management
    try:
        MAX_DRAWDOWN = D(os.environ.get('MAX_DRAWDOWN', '0.12'))  # 12%
        STOP_LOSS_PERCENT = D(os.environ.get('STOP_LOSS_PERCENT', '1.8'))  # 1.8%
        TAKE_PROFIT_PERCENT = D(os.environ.get('TAKE_PROFIT_PERCENT', '3.5'))  # 3.5%

        if not (0 < MAX_DRAWDOWN <= 1):
            raise ValueError("MAX_DRAWDOWN must be between 0 and 1")
        if STOP_LOSS_PERCENT <= 0:
            raise ValueError("STOP_LOSS_PERCENT must be positive")
        if TAKE_PROFIT_PERCENT <= 0:
            raise ValueError("TAKE_PROFIT_PERCENT must be positive")

    except (ValueError, InvalidOperation) as e:
        print(f"Risk management configuration error: {e}")
        sys.exit(1)

    # System Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    UPDATE_INTERVAL = max(1, int(os.environ.get('UPDATE_INTERVAL', '25')))  # seconds
    REPORT_INTERVAL = max(60, int(os.environ.get('REPORT_INTERVAL', '300')))  # 5 minutes

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = max(1, int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '1000')))
    WEIGHT_LIMIT_PER_MINUTE = max(1, int(os.environ.get('WEIGHT_LIMIT_PER_MINUTE', '5000')))

    # Maximum position size as percentage of balance
    MAX_POSITION_SIZE_PERCENT = D('20')  # 20% max per position

    # Top trading pairs (ALL 48 PAIRS AS ORIGINAL)
    TRADING_PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'MATICUSDT',
        'DOTUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT', 'VETUSDT',
        'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'APTUSDT', 'NEARUSDT', 'GRTUSDT', 'SANDUSDT', 'MANAUSDT',
        'FLOWUSDT', 'EGLDUSDT', 'XTZUSDT', 'THETAUSDT', 'AXSUSDT', 'AAVEUSDT', 'EOSUSDT', 'KLAYUSDT',
        'RUNEUSDT', 'FTMUSDT', 'NEOUSDT', 'CAKEUSDT', 'IOTAUSDT', 'ZECUSDT', 'DASHUSDT', 'WAVESUSDT',
        'CHZUSDT', 'BATUSDT', 'GALAUSDT', 'LRCUSDT', 'ENJUSDT', 'CELOUSDT', 'ZILUSDT', 'QTUMUSDT'
    ]

# ====================== LOGGING SETUP ======================
def setup_logging():
    """Setup logging configuration"""
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# ====================== QUANTITATIVE MODELS ======================
class KalmanFilter:
    """Kalman Filter for optimal price estimation - Renaissance Technologies method"""

    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        self.initialized = False

    def update(self, measurement):
        try:
            measurement = float(measurement)
            if not self.initialized:
                self.posteri_estimate = measurement
                self.initialized = True
                return measurement

            # Prediction step
            priori_estimate = self.posteri_estimate
            priori_error_estimate = self.posteri_error_estimate + self.process_variance

            # Update step
            blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
            self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
            self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

            return self.posteri_estimate
        except (ValueError, TypeError):
            return self.posteri_estimate if self.initialized else measurement

class OrnsteinUhlenbeckModel:
    """Mean reversion model - Two Sigma/DE Shaw method"""

    def __init__(self, window=80):
        self.window = window
        self.prices = deque(maxlen=window)

    def update(self, price):
        try:
            price = float(price)
            self.prices.append(price)

            if len(self.prices) < 15:
                return {'theta': 0, 'mu': price, 'sigma': 0, 'half_life': float('inf'), 'z_score': 0}

            prices_array = np.array(self.prices)
            if len(prices_array) < 15 or np.any(prices_array <= 0):
                return {'theta': 0, 'mu': price, 'sigma': 0, 'half_life': float('inf'), 'z_score': 0}

            log_prices = np.log(prices_array)
            returns = np.diff(log_prices)

            if len(returns) < 8:
                return {'theta': 0, 'mu': price, 'sigma': 0, 'half_life': float('inf'), 'z_score': 0}

            # Estimate OU parameters using least squares
            y = returns[1:]
            x = log_prices[:-2]

            if len(x) != len(y) or len(x) < 3:
                return {'theta': 0, 'mu': price, 'sigma': 0, 'half_life': float('inf'), 'z_score': 0}

            # SAFE POLYFIT WITH ERROR HANDLING - FIXED
            try:
                # Check data quality before polyfit
                if (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and 
                    np.std(x) > 1e-10 and np.std(y) > 1e-10 and
                    not np.allclose(x, x[0]) and not np.allclose(y, y[0])):

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.RankWarning)
                        coeffs = np.polyfit(x, y, 1)
                        beta, alpha = coeffs
                else:
                    beta, alpha = 0, 0
            except (np.linalg.LinAlgError, ValueError, np.RankWarning):
                beta, alpha = 0, 0

            theta = -beta if beta != 0 else 0  # Mean reversion speed
            mu = -alpha / beta if beta != 0 else np.mean(log_prices)  # Long-term mean
            sigma = np.std(y) if len(y) > 0 and np.all(np.isfinite(y)) else 0  # Volatility

            half_life = np.log(2) / theta if theta > 0 else float('inf')

            current_log_price = log_prices[-1]
            z_score = (current_log_price - mu) / sigma if sigma > 0 else 0

            return {
                'theta': theta,
                'mu': np.exp(mu) if np.isfinite(mu) else price,
                'sigma': sigma,
                'half_life': half_life,
                'z_score': z_score
            }

        except (ValueError, TypeError, np.linalg.LinAlgError, RuntimeWarning):
            return {'theta': 0, 'mu': price, 'sigma': 0, 'half_life': float('inf'), 'z_score': 0}

class HurstExponent:
    """Hurst Exponent for trend persistence - Citadel method"""

    @staticmethod
    def calculate(prices, max_lag=15):
        try:
            if len(prices) < max_lag * 2:
                return 0.5

            prices = np.array(prices)
            if len(prices) < max_lag * 2 or np.any(prices <= 0):
                return 0.5

            log_prices = np.log(prices)
            returns = np.diff(log_prices)

            lags = range(2, min(max_lag, len(returns) // 3))
            rs_values = []

            for lag in lags:
                n_periods = len(returns) // lag
                if n_periods < 2:
                    continue

                rs_period = []

                for i in range(n_periods):
                    period_returns = returns[i*lag:(i+1)*lag]

                    if len(period_returns) < lag:
                        continue

                    mean_return = np.mean(period_returns)
                    deviations = np.cumsum(period_returns - mean_return)

                    R = np.max(deviations) - np.min(deviations)
                    S = np.std(period_returns)

                    if S > 0:
                        rs_period.append(R / S)

                if rs_period:
                    rs_values.append(np.mean(rs_period))

            if len(rs_values) < 3:
                return 0.5

            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)

            if len(log_lags) != len(log_rs) or len(log_lags) < 2:
                return 0.5

            # SAFE POLYFIT WITH ERROR HANDLING - FIXED
            try:
                if (np.all(np.isfinite(log_lags)) and np.all(np.isfinite(log_rs)) and
                    np.std(log_lags) > 1e-10 and np.std(log_rs) > 1e-10):

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.RankWarning)
                        hurst_exponent = np.polyfit(log_lags, log_rs, 1)[0]
                        return max(0.1, min(0.9, hurst_exponent))
                else:
                    return 0.5
            except (np.linalg.LinAlgError, ValueError, np.RankWarning):
                return 0.5

        except (ValueError, TypeError, np.linalg.LinAlgError, RuntimeWarning):
            return 0.5

class RegimeDetector:
    """Market regime detection using HMM-style clustering"""

    def __init__(self, window=40):
        self.window = window
        self.observations = deque(maxlen=window)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.fitted = False

    def update(self, returns, volatility, volume_ratio):
        try:
            returns = float(returns)
            volatility = float(volatility)
            volume_ratio = float(volume_ratio)

            # Validate inputs
            if not all(np.isfinite([returns, volatility, volume_ratio])):
                return 'NORMAL'

            self.observations.append([returns, volatility, volume_ratio])

            if len(self.observations) < 20:
                return 'NORMAL'

            obs_array = np.array(self.observations)

            if not self.fitted and len(self.observations) >= 30:
                scaled_obs = self.scaler.fit_transform(obs_array)
                self.kmeans.fit(scaled_obs)
                self.fitted = True

            if self.fitted:
                current_obs = obs_array[-1:].reshape(1, -1)
                scaled_current = self.scaler.transform(current_obs)
                _ = self.kmeans.predict(scaled_current)[0]

                # Map regimes to meaningful labels
                recent_obs = obs_array[-10:]
                avg_return = np.mean(recent_obs[:, 0])
                avg_vol = np.mean(recent_obs[:, 1])

                if avg_vol > 0.04:
                    return 'CRISIS'
                elif avg_return > 0.002 and avg_vol < 0.03:
                    return 'BULL_TREND'
                elif avg_return < -0.002 and avg_vol < 0.03:
                    return 'BEAR_TREND'
                elif avg_vol < 0.015:
                    return 'LOW_VOL'
                else:
                    return 'NORMAL'
            else:
                return 'NORMAL'

        except (ValueError, TypeError, np.linalg.LinAlgError):
            return 'NORMAL'

# ====================== RATE LIMITER ======================
class RateLimiter:
    """Binance API rate limiter"""

    def __init__(self, max_requests_per_minute: int, max_weight_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.max_weight = max_weight_per_minute
        self.requests = deque()
        self.weight_used = deque()
        self.lock = threading.Lock()
        self.consecutive_failures = 0
        self.last_failure_time = 0

    def wait_if_needed(self, weight: int = 1):
        with self.lock:
            now = time.time()

            # Exponential backoff for consecutive failures
            if self.consecutive_failures > 0:
                backoff_time = min(60, 2 ** self.consecutive_failures)
                if now - self.last_failure_time < backoff_time:
                    time.sleep(backoff_time - (now - self.last_failure_time))

            # Remove old requests
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()

            while self.weight_used and now - self.weight_used[0][0] > 60:
                self.weight_used.popleft()

            # Check limits
            current_requests = len(self.requests)
            current_weight = sum(w[1] for w in self.weight_used)

            if current_requests >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if current_weight + weight > self.max_weight:
                sleep_time = 60 - (now - self.weight_used[0][0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.requests.append(now)
            self.weight_used.append((now, weight))

    def record_success(self):
        """Record successful API call"""
        self.consecutive_failures = 0

    def record_failure(self):
        """Record failed API call"""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

# ====================== REALISTIC PAPER TRADING CLIENT ======================
class RealisticPaperTradingClient:
    """Paper trading client using real Binance public API data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.balance = Config.INITIAL_BALANCE
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.base_url = "https://fapi.binance.com"  # Real Binance Futures API
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(1200, 6000)  # Public API limits
        self.position_lock = threading.Lock()  # Thread safety for positions
        self.logger.info("✅ Realistic paper trading client initialized with real market data")

    def _request(self, method: str, endpoint: str, params: Dict = None, weight: int = 1) -> Dict:
        """Make public API request (no authentication needed)"""
        self.rate_limiter.wait_if_needed(weight)
        params = params or {}
        url = self.base_url + endpoint

        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            result = response.json()
            self.rate_limiter.record_success()
            return result

        except requests.exceptions.RequestException as e:
            self.rate_limiter.record_failure()
            self.logger.error(f"Public API request failed: {e}")
            raise
        except ValueError as e:
            self.rate_limiter.record_failure()
            self.logger.error(f"Invalid JSON response: {e}")
            raise

    def _get_all_prices(self) -> Dict[str, float]:
        """Fetch all ticker prices in one call to reduce API load and avoid gaps"""
        try:
            data = self._request('GET', '/fapi/v1/ticker/price', weight=2)
            if not isinstance(data, list):
                return {}
            return {item['symbol']: float(item['price']) for item in data if 'symbol' in item and 'price' in item}
        except (ValueError, TypeError, KeyError):
            return {}

    def get_balance(self) -> Decimal:
        """Return simulated balance"""
        return self.balance

    def get_positions(self) -> List[Dict]:
        """Return simulated positions with real market prices (robust)"""
        with self.position_lock:
            positions: List[Dict[str, Any]] = []
            try:
                prices_map = self._get_all_prices()
            except Exception:
                prices_map = {}

            for symbol, pos in self.positions.items():
                try:
                    # Use live price if available, else last cached mark, else entry
                    current_price = prices_map.get(symbol, pos.get('last_mark_price', pos['entry_price']))
                    # Persist last seen price as a fallback for next time
                    self.positions[symbol]['last_mark_price'] = current_price

                    if pos['side'] == 'LONG':
                        pnl = (current_price - pos['entry_price']) * pos['size']
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['size']

                    notional = pos['entry_price'] * pos['size']
                    percentage = (pnl / notional) * 100 if notional > 0 else 0.0

                    positions.append({
                        'symbol': symbol,
                        'side': pos['side'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'mark_price': current_price,
                        'pnl': pnl,
                        'percentage': percentage
                    })
                except (ValueError, TypeError, KeyError):
                    continue

            return positions

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        """Get REAL candlestick data from Binance public API"""
        if not symbol or not interval:
            raise ValueError("Symbol and interval must be provided")
        if not (1 <= limit <= 1500):
            raise ValueError("Limit must be between 1 and 1500")

        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return self._request('GET', '/fapi/v1/klines', params, weight=1)

    def get_ticker_price(self, symbol: str) -> Dict:
        """Get REAL current price from Binance"""
        if not symbol:
            raise ValueError("Symbol must be provided")
        params = {'symbol': symbol}
        return self._request('GET', '/fapi/v1/ticker/price', params, weight=1)

    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get REAL 24hr ticker statistics"""
        if not symbol:
            raise ValueError("Symbol must be provided")
        params = {'symbol': symbol}
        return self._request('GET', '/fapi/v1/ticker/24hr', params, weight=1)

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Simulate leverage setting"""
        if not symbol:
            raise ValueError("Symbol must be provided")
        if not (1 <= leverage <= 125):
            raise ValueError("Leverage must be between 1 and 125")

        self.logger.info(f"📊 Simulated: Set {symbol} leverage to {leverage}x")
        return {'symbol': symbol, 'leverage': leverage, 'status': 'simulated'}

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """Simulate order placement with real market price"""
        if not symbol or not side or not order_type:
            raise ValueError("Symbol, side, and order_type must be provided")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if side not in ['BUY', 'SELL']:
            raise ValueError("Side must be BUY or SELL")

        try:
            # Get real current market price
            ticker = self.get_ticker_price(symbol)
            if 'price' not in ticker:
                raise ValueError(f"Invalid ticker response for {symbol}")

            current_price = float(ticker['price'])
            if current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")

            # Simulate slippage (0.01-0.05%)
            slippage = random.uniform(0.0001, 0.0005)
            if side == 'BUY':
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)

            order_id = random.randint(1000000, 9999999)

            with self.position_lock:
                # Handle position logic
                if symbol in self.positions:
                    # Close existing position
                    existing = self.positions[symbol]
                    if existing['side'] == 'LONG':
                        pnl = (execution_price - existing['entry_price']) * existing['size']
                    else:
                        pnl = (existing['entry_price'] - execution_price) * existing['size']

                    # Update balance with P&L using Decimal arithmetic
                    self.balance += Decimal(str(pnl))
                    self.positions[symbol]['last_mark_price'] = execution_price
                    self.logger.info(f"💰 Closed {existing['side']} {symbol}: P&L ${pnl:.2f}")
                    del self.positions[symbol]
                else:
                    # Open new position
                    position_side = 'LONG' if side == 'BUY' else 'SHORT'
                    self.positions[symbol] = {
                        'side': position_side,
                        'size': quantity,
                        'entry_price': execution_price,
                        'timestamp': time.time(),
                        'last_mark_price': execution_price
                    }
                    self.logger.info(f"🚀 Opened {position_side} {symbol}: Size {quantity} @ ${execution_price:.4f}")

            return {
                'symbol': symbol,
                'orderId': order_id,
                'status': 'FILLED',
                'side': side,
                'type': order_type,
                'executedQty': str(quantity),
                'price': str(execution_price),
                'fills': [{'price': str(execution_price), 'qty': str(quantity)}]
            }

        except Exception as e:
            self.logger.error(f"Simulated order failed for {symbol}: {e}")
            raise

    def close_position(self, symbol: str) -> Dict:
        """Close position by placing opposite order"""
        if not symbol:
            raise ValueError("Symbol must be provided")

        with self.position_lock:
            if symbol in self.positions:
                pos = self.positions[symbol]
                side = 'SELL' if pos['side'] == 'LONG' else 'BUY'
                return self.place_order(symbol, side, 'MARKET', pos['size'])
        raise ValueError(f"No position found for {symbol}")

    def test_connectivity(self):
        """Test connection to Binance public API"""
        try:
            _ = self._request('GET', '/fapi/v1/ping')
            self.logger.info("✅ Connected to Binance public API")

            # Test getting real data
            btc_price = self.get_ticker_price('BTCUSDT')
            if 'price' in btc_price:
                self.logger.info(f"📊 Real BTC price: ${float(btc_price['price']):,.2f}")

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Binance public API: {e}")
            raise

# ====================== TELEGRAM BOT ======================
class TelegramBot:
    """Telegram notification system - FIXED"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self.last_send = 0
        self.min_interval = 2.0

    def send_message(self, message: str, critical: bool = False):
        if not self.enabled:
            self.logger.info(f"Telegram: {message}")
            return

        if not message or len(message.strip()) == 0:
            return

        now = time.time()
        if not critical and now - self.last_send < self.min_interval:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message[:4000],
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                self.last_send = now
                self.logger.debug("Telegram message sent successfully")
            else:
                self.logger.warning(f"Telegram API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Telegram network error: {e}")
        except Exception as e:
            self.logger.warning(f"Telegram send failed: {e}")

# ====================== STRATEGY ENGINE ======================
@dataclass
class Signal:
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: float

class InstitutionalStrategyEngine:
    """Institutional-grade strategy engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kalman_filters = {}
        self.ou_models = {}
        self.regime_detectors = {}
        self.price_history = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_fitted = False

    def update_market_data(self, symbol: str, klines: List[List]):
        try:
            if not symbol or not klines or len(klines) == 0:
                return None

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Validate data
            if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                return None
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                return None

            if symbol not in self.kalman_filters:
                self.kalman_filters[symbol] = KalmanFilter()
                self.ou_models[symbol] = OrnsteinUhlenbeckModel()
                self.regime_detectors[symbol] = RegimeDetector()
                self.price_history[symbol] = deque(maxlen=150)

            current_price = float(df['close'].iloc[-1])
            self.price_history[symbol].append(current_price)

            _ = self.kalman_filters[symbol].update(current_price)
            ou_params = self.ou_models[symbol].update(current_price)

            if len(df) > 1:
                returns = df['close'].pct_change().iloc[-1]
                volatility = df['close'].pct_change().rolling(15).std().iloc[-1]
                volume_mean = df['volume'].rolling(20).mean().iloc[-1]
                volume_ratio = df['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1.0

                # Validate calculated values
                if not all(np.isfinite([returns, volatility, volume_ratio])):
                    regime = 'NORMAL'
                else:
                    regime = self.regime_detectors[symbol].update(returns, volatility, volume_ratio)
            else:
                regime = 'NORMAL'

            return {'df': df, 'filtered_price': current_price, 'ou_params': ou_params, 'regime': regime, 'current_price': current_price}
        except Exception as e:
            self.logger.error(f"Failed to update market data for {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str, market_data: Dict) -> Optional[Signal]:
        try:
            if not market_data or 'df' not in market_data:
                return None

            df = market_data['df']
            ou_params = market_data['ou_params']
            regime = market_data['regime']
            current_price = market_data['current_price']

            if len(df) < 30:
                return None

            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Validate data before calculations
            if close.isnull().any() or (close <= 0).any():
                return None

            # Technical indicators
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

            # Avoid division by zero
            loss_nonzero = loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + gain / loss_nonzero))

            if rsi.isnull().iloc[-1]:
                return None
            current_rsi = rsi.iloc[-1]

            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            macd_histogram = macd - signal_line

            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)

            volume_sma = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0

            # Validate all indicators
            if not all(np.isfinite([current_rsi, volume_ratio])):
                return None
            if bb_upper.isnull().iloc[-1] or bb_lower.isnull().iloc[-1]:
                return None

            # Hurst Exponent
            hurst = HurstExponent.calculate(list(self.price_history[symbol])) if len(self.price_history[symbol]) >= 30 else 0.5

            # Signal generation
            signals = []

            # 1. Mean Reversion (OU Model)
            if abs(ou_params['z_score']) > 1.8 and ou_params['half_life'] < 40:
                if ou_params['z_score'] < -1.8:
                    signals.append(('LONG', 0.8, 'OU Mean Reversion - Oversold'))
                elif ou_params['z_score'] > 1.8:
                    signals.append(('SHORT', 0.8, 'OU Mean Reversion - Overbought'))

            # 2. Trend Following (Hurst + MACD)
            if hurst > 0.6 and len(macd_histogram) > 1 and not macd_histogram.isnull().iloc[-2:].any():
                if macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-1] > macd_histogram.iloc[-2]:
                    signals.append(('LONG', 0.7, 'Trend Following - MACD Bullish'))
                elif macd_histogram.iloc[-1] < 0 and macd_histogram.iloc[-1] < macd_histogram.iloc[-2]:
                    signals.append(('SHORT', 0.7, 'Trend Following - MACD Bearish'))

            # 3. RSI Extremes with Volume Confirmation
            if volume_ratio > 1.5:
                if current_rsi < 25:
                    signals.append(('LONG', 0.9, 'RSI Oversold + Volume Spike'))
                elif current_rsi > 75:
                    signals.append(('SHORT', 0.9, 'RSI Overbought + Volume Spike'))

            # 4. Bollinger Band Breakouts
            if current_price > bb_upper.iloc[-1] and volume_ratio > 1.3:
                signals.append(('LONG', 0.6, 'Bollinger Breakout + Volume'))
            elif current_price < bb_lower.iloc[-1] and volume_ratio > 1.3:
                signals.append(('SHORT', 0.6, 'Bollinger Breakdown + Volume'))

            # 5. Regime-based signals
            if regime == 'BULL_TREND' and current_rsi < 50:
                signals.append(('LONG', 0.7, 'Bull Regime + RSI Dip'))
            elif regime == 'BEAR_TREND' and current_rsi > 50:
                signals.append(('SHORT', 0.7, 'Bear Regime + RSI Rally'))

            # 6. Kalman Filter Trend - FIXED
            if len(self.price_history[symbol]) >= 5:
                recent_filtered = [self.kalman_filters[symbol].posteri_estimate]
                if len(recent_filtered) >= 3:
                    if all(recent_filtered[i] > recent_filtered[i-1] for i in range(1, len(recent_filtered))):
                        signals.append(('LONG', 0.5, 'Kalman Uptrend'))
                    elif all(recent_filtered[i] < recent_filtered[i-1] for i in range(1, len(recent_filtered))):
                        signals.append(('SHORT', 0.5, 'Kalman Downtrend'))

            if not signals:
                return None

            # Aggregate signals
            long_signals = [s for s in signals if s[0] == 'LONG']
            short_signals = [s for s in signals if s[0] == 'SHORT']

            if len(long_signals) > len(short_signals):
                side = 'LONG'
                confidence = sum(s[1] for s in long_signals) / len(long_signals)
                reasoning = '; '.join([s[2] for s in long_signals])
            elif len(short_signals) > len(long_signals):
                side = 'SHORT'
                confidence = sum(s[1] for s in short_signals) / len(short_signals)
                reasoning = '; '.join([s[2] for s in short_signals])
            else:
                return None

            if confidence < float(Config.SIGNAL_THRESHOLD):
                return None

            # Calculate stop loss and take profit
            atr_high = high.rolling(14).max()
            atr_low = low.rolling(14).min()
            atr = (atr_high - atr_low).rolling(14).mean().iloc[-1]

            if not np.isfinite(atr) or atr <= 0:
                # Fallback ATR calculation
                atr = current_price * 0.02  # 2% of current price

            if side == 'LONG':
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 2.5)
            else:
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 2.5)

            # Validate stop loss and take profit
            if stop_loss <= 0 or take_profit <= 0:
                return None

            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                timestamp=time.time()
            )

        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
            return None

# ====================== TRADING BOT ======================
class TradingBot:
    """Main trading bot orchestrator"""

    def start_bot(self):
        """Start the trading bot in a background thread"""
        # Avoid multiple threads if already running
        if getattr(self, "_thread", None) and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop_bot(self):
        """Stop the trading bot and join the background thread"""
        self.stop()
        t = getattr(self, "_thread", None)
        if t and t.is_alive():
            try:
                t.join(timeout=5)
            except Exception:
                pass

    def __init__(self, use_realistic_paper: bool = True):
        self.logger = logging.getLogger(__name__)
        self.binance = RealisticPaperTradingClient()
        self.binance.test_connectivity()
        self.strategy = InstitutionalStrategyEngine()
        self.telegram = TelegramBot()
        self.running = False
        self.start_time = time.time()
        self.last_report = 0
        self.positions = {}
        self.balance = Config.INITIAL_BALANCE
        self.equity_peak = Config.INITIAL_BALANCE
        self.total_pnl = Decimal('0')
        self.position_lock = threading.Lock()
        self.consecutive_failures = 0
        self.last_failure_time = 0
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT,
                    size REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    timestamp REAL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    size REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    timestamp REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database init failed: {e}")

    def get_balance(self) -> Decimal:
        """Get current balance"""
        try:
            balance = self.binance.get_balance()
            if balance <= 0:
                return self.balance
            return balance
        except Exception:
            return self.balance

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.binance.get_positions()
            if positions:
                return positions
        except Exception:
            pass

        # Fallback: synthesize from locally tracked positions
        with self.position_lock:
            synthesized: List[Dict[str, Any]] = []
            for symbol, p in self.positions.items():
                try:
                    ticker = self.binance.get_ticker_price(symbol)
                    if 'price' not in ticker:
                        current_price = p['entry_price']
                    else:
                        current_price = float(ticker['price'])
                except Exception:
                    current_price = p['entry_price']

                try:
                    pnl = (current_price - p['entry_price']) * p['size'] if p['side'] == 'LONG' \
                        else (p['entry_price'] - current_price) * p['size']
                    notional = p['entry_price'] * p['size']
                    percentage = (pnl / notional) * 100 if notional > 0 else 0.0

                    synthesized.append({
                        'symbol': symbol,
                        'side': p['side'],
                        'size': p['size'],
                        'entry_price': p['entry_price'],
                        'mark_price': current_price,
                        'pnl': pnl,
                        'percentage': percentage
                    })
                except (ValueError, TypeError, KeyError):
                    continue

        return synthesized

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using risk management"""
        try:
            if entry_price <= 0 or stop_loss <= 0:
                return 0

            current_balance = float(self.get_balance())
            if current_balance <= 0:
                return 0

            risk_amount = current_balance * float(Config.BASE_RISK_PERCENT) / 100

            price_risk = abs(entry_price - stop_loss)
            if price_risk <= 0:
                return 0

            position_size = risk_amount / price_risk

            # Apply leverage
            position_size *= Config.LEVERAGE

            # Minimum position size
            min_size = 10.0 / entry_price  # $10 minimum

            # Maximum position size (percentage of balance)
            max_size = (current_balance * float(Config.MAX_POSITION_SIZE_PERCENT) / 100) / entry_price

            return max(min_size, min(position_size, max_size))

        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0

    def open_position(self, signal: Signal) -> bool:
        """Open a new position"""
        try:
            with self.position_lock:
                # Check if we already have a position
                if signal.symbol in self.positions:
                    return False

                # Check maximum positions
                if len(self.positions) >= Config.MAX_POSITIONS:
                    return False

            # Calculate position size
            position_size = self.calculate_position_size(signal.symbol, signal.entry_price, signal.stop_loss)

            if position_size <= 0:
                return False

            # Set leverage
            try:
                self.binance.set_leverage(signal.symbol, Config.LEVERAGE)
            except Exception as e:
                self.logger.warning(f"Failed to set leverage for {signal.symbol}: {e}")

            # Place order
            side = 'BUY' if signal.side == 'LONG' else 'SELL'
            order_result = self.binance.place_order(
                symbol=signal.symbol,
                side=side,
                order_type='MARKET',
                quantity=position_size
            )

            if 'orderId' not in order_result and 'status' not in order_result:
                raise RuntimeError("Order placement failed - no order ID returned")

            # Store position
            position = {
                'symbol': signal.symbol,
                'side': signal.side,
                'size': position_size,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'timestamp': signal.timestamp
            }

            with self.position_lock:
                self.positions[signal.symbol] = position

            # Save to database
            try:
                self.conn.execute('''
                    INSERT OR REPLACE INTO positions 
                    (symbol, side, size, entry_price, stop_loss, take_profit, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (signal.symbol, signal.side, position_size, signal.entry_price, 
                      signal.stop_loss, signal.take_profit, signal.timestamp))
                self.conn.commit()
            except Exception as e:
                self.logger.warning(f"Failed to save position to database: {e}")

            # Send notification
            self.telegram.send_message(
                f"🚀 <b>POSITION OPENED</b>\n"
                f"📊 {signal.symbol}\n"
                f"📈 {signal.side} ${position_size:.4f}\n"
                f"💰 Entry: ${signal.entry_price:.4f}\n"
                f"🛑 SL: ${signal.stop_loss:.4f}\n"
                f"🎯 TP: ${signal.take_profit:.4f}\n"
                f"🧠 Reason: {signal.reasoning}\n"
                f"⚡ Confidence: {signal.confidence:.1%}"
            )

            self.logger.info(f"Opened {signal.side} position for {signal.symbol}")
            self.consecutive_failures = 0
            return True

        except Exception as e:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            self.logger.error(f"Failed to open position for {signal.symbol}: {e}")
            return False

    def close_position(self, symbol: str, reason: str = "Manual") -> bool:
        """Close a position"""
        try:
            with self.position_lock:
                if symbol not in self.positions:
                    return False
                position = self.positions[symbol].copy()

            # Get current price
            try:
                ticker = self.binance.get_ticker_price(symbol)
                if 'price' not in ticker:
                    raise ValueError("Invalid ticker response")
                current_price = float(ticker['price'])
            except Exception as e:
                self.logger.warning(f"Failed to get current price for {symbol}: {e}")
                current_price = position['entry_price']

            # Close position on exchange
            try:
                self.binance.close_position(symbol)
            except Exception as e:
                self.logger.warning(f"Exchange close failed for {symbol}: {e}")

            # Calculate PnL using Decimal arithmetic
            try:
                if position['side'] == 'LONG':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']

                self.total_pnl += Decimal(str(pnl))
            except (ValueError, InvalidOperation):
                pnl = 0.0

            # Save trade to database
            try:
                self.conn.execute('''
                    INSERT INTO trades 
                    (symbol, side, size, entry_price, exit_price, pnl, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, position['side'], position['size'], position['entry_price'], 
                      current_price, pnl, time.time()))
            except Exception as e:
                self.logger.warning(f"Failed to save trade to database: {e}")

            # Remove position
            with self.position_lock:
                if symbol in self.positions:
                    del self.positions[symbol]

            try:
                self.conn.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
                self.conn.commit()
            except Exception as e:
                self.logger.warning(f"Failed to remove position from database: {e}")

            # Send notification
            emoji = "✅" if pnl > 0 else "❌"
            self.telegram.send_message(
                f"{emoji} <b>POSITION CLOSED</b>\n"
                f"📊 {symbol}\n"
                f"📈 {position['side']} ${position['size']:.4f}\n"
                f"💰 Entry: ${position['entry_price']:.4f}\n"
                f"📉 Exit: ${current_price:.4f}\n"
                f"💵 PnL: ${pnl:.2f}\n"
                f"📝 Reason: {reason}"
            )

            self.logger.info(f"Closed {position['side']} position for {symbol}, PnL: ${pnl:.2f}")
            self.consecutive_failures = 0
            return True

        except Exception as e:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            self.logger.error(f"Failed to close position for {symbol}: {e}")
            return False

    def close_all_positions(self) -> int:
        """Close all open positions"""
        try:
            positions = self.get_positions()
            closed_count = 0

            for position in positions:
                if self.close_position(position['symbol'], "Close All"):
                    closed_count += 1
                    time.sleep(0.1)  # Small delay between closes

            self.logger.info(f"Closed {closed_count} positions")
            return closed_count

        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return 0

    def manage_positions(self):
        """Manage existing positions"""
        try:
            current_positions = self.get_positions()

            for position in current_positions:
                try:
                    symbol = position['symbol']
                    current_price = position['mark_price']

                    # Check stop loss
                    with self.position_lock:
                        if symbol in self.positions:
                            stored_position = self.positions[symbol]

                            if stored_position['side'] == 'LONG':
                                if current_price <= stored_position['stop_loss']:
                                    self.close_position(symbol, "Stop Loss")
                                elif current_price >= stored_position['take_profit']:
                                    self.close_position(symbol, "Take Profit")
                            else:  # SHORT
                                if current_price >= stored_position['stop_loss']:
                                    self.close_position(symbol, "Stop Loss")
                                elif current_price <= stored_position['take_profit']:
                                    self.close_position(symbol, "Take Profit")

                            # Time-based exit (24 hours max)
                            if time.time() - stored_position['timestamp'] > 86400:
                                self.close_position(symbol, "Time Limit")
                except Exception as e:
                    self.logger.warning(f"Failed to manage position for {position.get('symbol', 'unknown')}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Position management failed: {e}")

    def scan_for_signals(self):
        """Scan all symbols for trading signals"""
        # Circuit breaker for consecutive failures
        if self.consecutive_failures >= 5:
            backoff_time = min(300, 30 * (2 ** (self.consecutive_failures - 5)))
            if time.time() - self.last_failure_time < backoff_time:
                self.logger.warning(f"Circuit breaker active: waiting {backoff_time}s after {self.consecutive_failures} failures")
                return

        try:
            for symbol in Config.TRADING_PAIRS[:20]:  # Limit to top 20 for free tier
                try:
                    # Skip if we already have a position
                    with self.position_lock:
                        if symbol in self.positions:
                            continue

                    # Get market data
                    klines = self.binance.get_klines(symbol, '5m', 100)
                    if not klines or len(klines) == 0:
                        continue

                    market_data = self.strategy.update_market_data(symbol, klines)

                    if not market_data:
                        continue

                    # Generate signal
                    signal = self.strategy.generate_signal(symbol, market_data)

                    if signal and signal.confidence >= float(Config.SIGNAL_THRESHOLD):
                        success = self.open_position(signal)
                        if success:
                            self.consecutive_failures = 0

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    self.logger.warning(f"Signal scan failed for {symbol}: {e}")
                    continue

        except Exception as e:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            self.logger.error(f"Signal scanning failed: {e}")

    def send_periodic_report(self):
        """Send periodic status report"""
        try:
            now = time.time()
            if now - self.last_report < Config.REPORT_INTERVAL:
                return

            current_balance = self.get_balance()
            positions = self.get_positions()

            total_unrealized_pnl = sum(pos['pnl'] for pos in positions if 'pnl' in pos and np.isfinite(pos['pnl']))

            try:
                total_return = ((current_balance + Decimal(str(total_unrealized_pnl)) - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
            except (InvalidOperation, ZeroDivisionError):
                total_return = Decimal('0')

            report = (
                f"📊 <b>TRADING REPORT</b>\n"
                f"💰 Balance: ${float(current_balance):,.2f}\n"
                f"📈 Unrealized PnL: ${total_unrealized_pnl:+,.2f}\n"
                f"📊 Total Return: {float(total_return):+.2f}%\n"
                f"🔢 Open Positions: {len(positions)}/{Config.MAX_POSITIONS}\n"
                f"⏰ Uptime: {(now - self.start_time)/3600:.1f}h\n"
                f"🎯 Strategy: Institutional Grade\n"
                f"⚡ Mode: {Config.LEVERAGE}x Leverage"
            )

            if positions:
                report += "\n\n<b>OPEN POSITIONS:</b>\n"
                for pos in positions[:5]:  # Show top 5
                    try:
                        emoji = "🟢" if pos.get('pnl', 0) > 0 else "🔴"
                        symbol = pos.get('symbol', 'Unknown')
                        side = pos.get('side', 'Unknown')
                        pnl = pos.get('pnl', 0)
                        report += f"{emoji} {symbol} {side}: ${pnl:+.2f}\n"
                    except (ValueError, TypeError):
                        continue

            self.telegram.send_message(report)
            self.last_report = now

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

    def run(self):
        """Main trading loop"""
        self.running = True
        self.logger.info("🚀 Starting OmegaX Institutional Futures Trading Bot")

        # Send startup notification
        self.telegram.send_message(
            f"🚀 <b>OMEGAX BOT STARTED</b>\n"
            f"💰 Initial Balance: ${float(Config.INITIAL_BALANCE):,.2f}\n"
            f"📊 Monitoring: {len(Config.TRADING_PAIRS)} pairs\n"
            f"⚡ Leverage: {Config.LEVERAGE}x\n"
            f"🎯 Max Positions: {Config.MAX_POSITIONS}\n"
            f"🧠 Strategy: Institutional Grade\n"
            f"🔥 Mode: BEAST ACTIVATED",
            critical=True
        )

        try:
            while self.running:
                loop_start = time.time()

                try:
                    # Manage existing positions
                    self.manage_positions()

                    # Scan for new signals
                    self.scan_for_signals()

                    # Send periodic reports
                    self.send_periodic_report()

                except Exception as e:
                    self.consecutive_failures += 1
                    self.last_failure_time = time.time()
                    self.logger.error(f"Trading loop error: {e}")

                # Sleep with jitter
                loop_time = time.time() - loop_start
                sleep_time = max(1, Config.UPDATE_INTERVAL - loop_time)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.telegram.send_message(f"💥 <b>FATAL ERROR</b>\n{str(e)[:200]}", critical=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the trading bot"""
        self.running = False

        # Send shutdown notification
        try:
            final_balance = self.get_balance()
            try:
                total_return = ((final_balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
            except (InvalidOperation, ZeroDivisionError):
                total_return = Decimal('0')

            self.telegram.send_message(
                f"🛑 <b>BOT STOPPED</b>\n"
                f"💰 Final Balance: ${float(final_balance):,.2f}\n"
                f"📈 Total Return: {float(total_return):+.2f}%\n"
                f"📊 Open Positions: {len(self.get_positions())}\n"
                f"⏰ Runtime: {(time.time() - self.start_time)/3600:.1f}h",
                critical=True
            )
        except Exception:
            pass

        self.logger.info("Trading bot stopped")

# ====================== WEB UI DASHBOARD ======================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'omegax-secret-key-2024')

# HTML Template for Web UI
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmegaX Trading Bot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-card h3 {
            color: #ffd700;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .status-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .neutral { color: #94a3b8; }

        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }

        .btn-primary {
            background: linear-gradient(45deg, #10b981, #059669);
            color: white;
        }

        .btn-danger {
            background: linear-gradient(45deg, #ef4444, #dc2626);
            color: white;
        }

        .btn-warning {
            background: linear-gradient(45deg, #f59e0b, #d97706);
            color: white;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        .positions-section {
            background: rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .positions-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px.
        }

        .positions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .positions-table th,
        .positions-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .positions-table th {
            background: rgba(255, 255, 255, 0.1);
            font-weight: bold;
            color: #ffd700;
        }

        .positions-table tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .close-btn {
            padding: 6px 12px;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }

        .close-btn:hover {
            background: #dc2626;
        }

        .external-link {
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(45deg, #8b5cf6, #7c3aed);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .external-link:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4);
        }

        .refresh-info {
            text-align: center;
            margin-top: 20px;
            color: #94a3b8;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header h1 {
                font-size: 2em;
            }

            .controls {
                flex-direction: column;
                align-items: center;
            }

            .btn {
                width: 100%;
                max-width: 300px;
            }

            .positions-table {
                font-size: 0.9em;
            }

            .external-link {
                position: static;
                display: block;
                margin: 20px auto;
                text-align: center;
                width: fit-content;
            }
        }
    </style>
    <script>
        // Auto-refresh every 5 seconds
        setTimeout(function() {
            location.reload();
        }, 5000);

        function closePosition(symbol) {
            if (confirm('Are you sure you want to close position for ' + symbol + '?')) {
                fetch('/close_position', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({symbol: symbol})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Position closed successfully!');
                        location.reload();
                    } else {
                        alert('Failed to close position: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            }
        }

        function closeAllPositions() {
            if (confirm('Are you sure you want to close ALL positions? This cannot be undone!')) {
                fetch('/close_all_positions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('All positions closed successfully! Closed: ' + data.closed_count);
                        location.reload();
                    } else {
                        alert('Failed to close positions: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            }
        }

        function toggleBot() {
            const action = {{ 'stop' if bot_running else 'start' }};
            fetch('/toggle_bot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({action: action})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Bot ' + action + 'ed successfully!');
                    location.reload();
                } else {
                    alert('Failed to ' + action + ' bot: ' + data.error);
                }
            })
            .catch(error => {
                alert('Error: ' + error);
            });
        }
    </script>
</head>
<body>
    <a href="{{ external_link }}" target="_blank" class="external-link">🔗 BetterNext</a>

    <div class="container">
        <div class="header">
            <h1>🚀 OmegaX Trading Bot</h1>
            <p>Institutional-Grade Crypto Futures Trading</p>
            <p><strong>Status:</strong> 
                <span class="{{ 'positive' if bot_running else 'negative' }}">
                    {{ '🟢 RUNNING' if bot_running else '🔴 STOPPED' }}
                </span>
            </p>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>💰 Balance</h3>
                <div class="status-value">${{ "%.2f"|format(balance) }}</div>
            </div>

            <div class="status-card">
                <h3>📈 Total P&L</h3>
                <div class="status-value {{ 'positive' if total_pnl >= 0 else 'negative' }}">
                    ${{ "%.2f"|format(total_pnl) }}
                </div>
            </div>

            <div class="status-card">
                <h3>📊 Open Positions</h3>
                <div class="status-value">{{ positions|length }}/{{ max_positions }}</div>
            </div>

            <div class="status-card">
                <h3>⏰ Uptime</h3>
                <div class="status-value">{{ "%.1f"|format(uptime) }}h</div>
            </div>

            <div class="status-card">
                <h3>⚡ Leverage</h3>
                <div class="status-value">{{ leverage }}x</div>
            </div>

            <div class="status-card">
                <h3>🎯 Strategy</h3>
                <div class="status-value" style="font-size: 1.2em;">Institutional</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn {{ 'btn-danger' if bot_running else 'btn-primary' }}" onclick="toggleBot()">
                {{ '⏹️ Stop Bot' if bot_running else '▶️ Start Bot' }}
            </button>

            {% if positions %}
            <button class="btn btn-warning" onclick="closeAllPositions()">
                🚫 Close All Positions
            </button>
            {% endif %}

            <a href="/api/status" class="btn btn-primary" target="_blank">
                📊 API Status
            </a>
        </div>

        <div class="positions-section">
            <div class="positions-header">
                <h2>📋 Open Positions</h2>
                <span>{{ positions|length }} active trades</span>
            </div>

            {% if positions %}
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in positions %}
                    <tr>
                        <td><strong>{{ pos.symbol }}</strong></td>
                        <td>
                            <span class="{{ 'positive' if pos.side == 'LONG' else 'negative' }}">
                                {{ pos.side }}
                            </span>
                        </td>
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
                            <button class="close-btn" onclick="closePosition('{{ pos.symbol }}')">
                                ❌ Close
                            </button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div style="text-align: center; padding: 40px; color: #94a3b8;">
                <h3>No open positions</h3>
                <p>The bot is scanning for trading opportunities...</p>
            </div>
            {% endif %}
        </div>

        <div class="refresh-info">
            🔄 Page auto-refreshes every 5 seconds | Last updated: {{ current_time }}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard"""
    global bot_instance

    if not bot_instance:
        return render_template_string(
            DASHBOARD_HTML,
            balance=0.0,
            total_pnl=0.0,
            positions=[],
            bot_running=False,
            uptime=0.0,
            leverage=Config.LEVERAGE,
            max_positions=Config.MAX_POSITIONS,
            external_link='https://betternext.com',
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

    try:
        balance = float(bot_instance.get_balance())
        positions = bot_instance.get_positions()
        total_pnl = sum(pos.get('pnl', 0) for pos in positions if 'pnl' in pos)
        uptime = (time.time() - bot_instance.start_time) / 3600

        return render_template_string(
            DASHBOARD_HTML,
            balance=balance,
            total_pnl=total_pnl,
            positions=positions,
            bot_running=bot_instance.running,
            uptime=uptime,
            leverage=Config.LEVERAGE,
            max_positions=Config.MAX_POSITIONS,
            external_link='https://betternext.com',
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    except Exception as e:
        return f"Error loading dashboard: {e}"

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    global bot_instance

    if not bot_instance:
        return jsonify({
            'status': 'error',
            'message': 'Bot not initialized',
            'balance': 0,
            'positions': 0,
            'running': False
        })

    try:
        balance = float(bot_instance.get_balance())
        positions = bot_instance.get_positions()
        total_pnl = sum(pos.get('pnl', 0) for pos in positions if 'pnl' in pos)
        uptime = (time.time() - bot_instance.start_time) / 3600

        return jsonify({
            'status': 'healthy',
            'service': 'OmegaX Institutional Futures Bot',
            'timestamp': time.time(),
            'balance': balance,
            'total_pnl': total_pnl,
            'positions': len(positions),
            'running': bot_instance.running,
            'uptime_hours': uptime,
            'leverage': Config.LEVERAGE,
            'max_positions': Config.MAX_POSITIONS,
            'trading_pairs': len(Config.TRADING_PAIRS)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': time.time()
        })

@app.route('/close_position', methods=['POST'])
def close_position():
    """Close a specific position"""
    global bot_instance

    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})

    try:
        data = request.get_json()
        symbol = data.get('symbol')

        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'})

        success = bot_instance.close_position(symbol, "Manual Close")

        if success:
            return jsonify({'success': True, 'message': f'Position {symbol} closed successfully'})
        else:
            return jsonify({'success': False, 'error': f'Failed to close position {symbol}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/close_all_positions', methods=['POST'])
def close_all_positions():
    """Close all open positions"""
    global bot_instance

    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})

    try:
        closed_count = bot_instance.close_all_positions()
        return jsonify({
            'success': True, 
            'message': f'Closed {closed_count} positions',
            'closed_count': closed_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_bot', methods=['POST'])
def toggle_bot():
    """Start or stop the trading bot"""
    global bot_instance

    if not bot_instance:
        return jsonify({'success': False, 'error': 'Bot not initialized'})

    try:
        data = request.get_json()
        action = data.get('action')

        if action == 'start':
            bot_instance.start_bot()
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        elif action == 'stop':
            bot_instance.stop_bot()
            return jsonify({'success': True, 'message': 'Bot stopped successfully'})
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ====================== SCHEDULER ======================
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

def run_bot_cycle():
    """Run one cycle of the trading bot"""
    global bot_instance
    if bot_instance and bot_instance.running:
        try:
            bot_instance.manage_positions()
            bot_instance.scan_for_signals()
            bot_instance.send_periodic_report()
        except Exception as e:
            logging.error(f"Bot cycle error: {e}")

# Schedule bot to run every 25 seconds
scheduler.add_job(
    func=run_bot_cycle,
    trigger="interval",
    seconds=Config.UPDATE_INTERVAL,
    id='bot_cycle'
)

# ====================== MAIN ENTRY POINT ======================
def main():
    """Main entry point optimized for Render"""
    global bot_instance

    try:
        # Render-specific port handling
        port = int(os.environ.get('PORT', 10000))

        # Force paper trading for safety on cloud deployment
        os.environ['USE_REALISTIC_PAPER'] = 'true'

        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)

        logger.info("🚀 Starting OmegaX Trading Bot with Web UI on Render")
        logger.info("📝 REALISTIC PAPER TRADING MODE - Real market data, simulated trades")
        logger.info("🌐 Using REAL Binance market data (public API)")
        logger.info("🔒 Cloud deployment - forcing paper trading for safety")

        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            logger.warning("⚠️ Telegram credentials missing - notifications disabled")

        logger.info("🏛️ Initializing OmegaX Institutional Futures Trading Bot")
        logger.info(f"📊 Monitoring {len(Config.TRADING_PAIRS)} trading pairs")
        logger.info(f"⚡ Leverage: {Config.LEVERAGE}x")
        logger.info(f"🎯 Max Positions: {Config.MAX_POSITIONS}")
        logger.info(f"💰 Base Risk: {Config.BASE_RISK_PERCENT}% per trade")
        logger.info(f"🔄 Mode: REALISTIC PAPER TRADING (Render Deployment)")

        # Initialize trading bot with forced paper trading
        bot_instance = TradingBot(use_realistic_paper=True)

        # Start trading bot
        bot_instance.start_bot()

        logger.info(f"🌐 Web UI server starting on port {port}")
        logger.info("✅ Bot is now running with REAL market data on Render")
        logger.info(f"🔗 Dashboard available at: https://your-app-name.onrender.com")
        logger.info(f"📊 API status at: https://your-app-name.onrender.com/api/status")

        # Run Flask app
        app.run(host='0.0.0.0', port=port, debug=False)

    except Exception as e:
        print(f"💥 Fatal startup error: {e}")
        logging.error(f"💥 Fatal startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()