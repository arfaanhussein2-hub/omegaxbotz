#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OmegaX Trading Bot v4.0 - Institutional Algorithms Integration
Multiple institutional-grade futures algorithms in one platform
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
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, InvalidOperation, ROUND_DOWN
from collections import deque, defaultdict
from functools import wraps
import warnings
from typing import List, Dict, Optional, Tuple
import math
from statistics import mean, stdev
from scipy import stats
import pandas as pd
# Add at the top of app.py after imports
import gc
import os

# Memory optimization for Render free tier
class RenderOptimizer:
    @staticmethod
    def optimize_for_free_tier():
        """Optimize settings for Render free tier"""
        # Reduce algorithm count if memory is low
        try:
            import psutil
            memory_mb = psutil.virtual_memory().available / 1024 / 1024
            
            if memory_mb < 400:  # Less than 400MB available
                # Disable memory-intensive algorithms
                os.environ['CITADEL_ENABLED'] = 'false'
                os.environ['TWOSIGMA_ENABLED'] = 'false'
                os.environ['AQR_ENABLED'] = 'false'
                os.environ['MANGROUP_ENABLED'] = 'false'
                
                # Reduce data limits
                os.environ['KLINE_HISTORY_LIMIT'] = '50'
                os.environ['MAX_POSITIONS'] = '8'
                
                print(f"⚠️ Low memory detected ({memory_mb:.0f}MB), optimizing...")
                
        except ImportError:
            # If psutil not available, use conservative settings
            pass
    
    @staticmethod
    def periodic_cleanup():
        """Periodic memory cleanup"""
        gc.collect()
        
        # Clear algorithm caches if they exist
        try:
            if hasattr(bot_instance, 'market_data_cache'):
                # Keep only recent data
                for symbol in list(bot_instance.market_data_cache.keys()):
                    if len(bot_instance.market_data_cache[symbol]) > 50:
                        bot_instance.market_data_cache[symbol] = bot_instance.market_data_cache[symbol][-50:]
        except:
            pass

# Add to main() function before bot initialization
def main():
    global bot_instance
    
    try:
        # Optimize for Render free tier
        RenderOptimizer.optimize_for_free_tier()
        
        port = int(os.environ.get('PORT', 8080))
        # ... rest of main function

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
    print(f"Installing dependencies: {e}")
    os.system("pip install Flask APScheduler numpy scipy pandas")
    from flask import Flask, render_template_string, jsonify, request, redirect, url_for, session
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

getcontext().prec = 32
bot_instance = None

# ====================== ENHANCED CONFIGURATION ======================
class Config:
    """Enhanced configuration for institutional algorithms"""
    
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
    
    # Enhanced Trading Parameters
    INITIAL_BALANCE = Decimal(os.environ.get('INITIAL_BALANCE', '10000.00'))  # Increased for institutional strategies
    BASE_RISK_PERCENT = Decimal(os.environ.get('BASE_RISK_PERCENT', '0.5'))  # More conservative
    MAX_POSITIONS = int(os.environ.get('MAX_POSITIONS', '20'))  # Increased capacity
    LEVERAGE = int(os.environ.get('LEVERAGE', '5'))  # More conservative leverage
    POSITION_TIME_LIMIT = int(os.environ.get('POSITION_TIME_LIMIT', '172800'))  # 48 hours
    SIGNAL_THRESHOLD = Decimal(os.environ.get('SIGNAL_THRESHOLD', '0.65'))
    
    # Risk Management
    MAX_DRAWDOWN = Decimal(os.environ.get('MAX_DRAWDOWN', '0.10'))  # 10%
    STOP_LOSS_PERCENT = Decimal(os.environ.get('STOP_LOSS_PERCENT', '1.5'))
    TAKE_PROFIT_PERCENT = Decimal(os.environ.get('TAKE_PROFIT_PERCENT', '3.0'))
    MIN_POSITION_SIZE_USD = Decimal(os.environ.get('MIN_POSITION_SIZE_USD', '25.00'))
    MAX_POSITION_SIZE_PERCENT = Decimal(os.environ.get('MAX_POSITION_SIZE_PERCENT', '8.0'))
    
    # Institutional Algorithm Controls
    GOLDMAN_ENABLED = os.environ.get('GOLDMAN_ENABLED', 'true').lower() == 'true'
    JPMORGAN_ENABLED = os.environ.get('JPMORGAN_ENABLED', 'true').lower() == 'true'
    CITADEL_ENABLED = os.environ.get('CITADEL_ENABLED', 'true').lower() == 'true'
    RENAISSANCE_ENABLED = os.environ.get('RENAISSANCE_ENABLED', 'true').lower() == 'true'
    TWOSIGMA_ENABLED = os.environ.get('TWOSIGMA_ENABLED', 'true').lower() == 'true'
    DESHAW_ENABLED = os.environ.get('DESHAW_ENABLED', 'true').lower() == 'true'
    BRIDGEWATER_ENABLED = os.environ.get('BRIDGEWATER_ENABLED', 'true').lower() == 'true'
    AQR_ENABLED = os.environ.get('AQR_ENABLED', 'true').lower() == 'true'
    WINTON_ENABLED = os.environ.get('WINTON_ENABLED', 'true').lower() == 'true'
    MANGROUP_ENABLED = os.environ.get('MANGROUP_ENABLED', 'true').lower() == 'true'
    
    # System Settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    UPDATE_INTERVAL = int(os.environ.get('UPDATE_INTERVAL', '45'))  # Slightly slower for complex algos
    REPORT_INTERVAL = int(os.environ.get('REPORT_INTERVAL', '900'))  # 15 minutes
    DATABASE_FILE = os.environ.get('DATABASE_FILE', '/tmp/omegax.db')
    
    # Enhanced settings
    USE_REALISTIC_PAPER = os.environ.get('USE_REALISTIC_PAPER', 'true').lower() == 'true'
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '86400'))
    KLINE_HISTORY_LIMIT = int(os.environ.get('KLINE_HISTORY_LIMIT', '200'))  # Increased for complex algos
    
    # Extended crypto pairs for institutional strategies
    TRADING_PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT',
        'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT', 'ICPUSDT',
        'HBARUSDT', 'APTUSDT', 'NEARUSDT', 'ALGOUSDT', 'FLOWUSDT', 'SANDUSDT',
        'MANAUSDT', 'AXSUSDT', 'CHZUSDT', 'ENJUSDT', 'GALAUSDT', 'THETAUSDT'
    ]

# ====================== ENHANCED MATHEMATICAL FUNCTIONS ======================
def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        result = float(value)
        return result if abs(result) < 1e20 and not math.isnan(result) else default
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

def calculate_returns(prices):
    """Calculate log returns"""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i-1]))
    return returns

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    if len(returns) < window:
        return 0
    return stdev(returns[-window:]) * math.sqrt(252)  # Annualized

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if not returns or len(returns) < 2:
        return 0
    excess_returns = [r - risk_free_rate/252 for r in returns]
    if stdev(excess_returns) == 0:
        return 0
    return mean(excess_returns) / stdev(excess_returns) * math.sqrt(252)

def z_score(value, mean_val, std_val):
    """Calculate Z-score"""
    if std_val == 0:
        return 0
    return (value - mean_val) / std_val

def bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(prices) < window:
        return None, None, None
    
    recent_prices = prices[-window:]
    sma = mean(recent_prices)
    std = stdev(recent_prices)
    
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    
    return upper, sma, lower

def rsi(prices, window=14):
    """Calculate RSI"""
    if len(prices) < window + 1:
        return 50
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, change) for change in changes[-window:]]
    losses = [max(0, -change) for change in changes[-window:]]
    
    avg_gain = mean(gains) if gains else 0
    avg_loss = mean(losses) if losses else 0
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    if len(prices) < slow:
        return 0, 0, 0
    
    # Simple approximation of EMA using SMA
    fast_ma = mean(prices[-fast:])
    slow_ma = mean(prices[-slow:])
    macd_line = fast_ma - slow_ma
    
    # Signal line approximation
    if len(prices) >= slow + signal:
        macd_values = []
        for i in range(signal):
            idx = len(prices) - signal + i
            if idx >= slow:
                f_ma = mean(prices[idx-fast:idx])
                s_ma = mean(prices[idx-slow:idx])
                macd_values.append(f_ma - s_ma)
        signal_line = mean(macd_values) if macd_values else 0
    else:
        signal_line = 0
    
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def stochastic(highs, lows, closes, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    if len(closes) < k_window:
        return 50, 50
    
    recent_highs = highs[-k_window:]
    recent_lows = lows[-k_window:]
    current_close = closes[-1]
    
    highest_high = max(recent_highs)
    lowest_low = min(recent_lows)
    
    if highest_high == lowest_low:
        k_percent = 50
    else:
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # D% is a moving average of K%
    if len(closes) >= k_window + d_window - 1:
        k_values = []
        for i in range(d_window):
            idx = len(closes) - d_window + i
            if idx >= k_window - 1:
                h = max(highs[idx-k_window+1:idx+1])
                l = min(lows[idx-k_window+1:idx+1])
                c = closes[idx]
                if h != l:
                    k_values.append(((c - l) / (h - l)) * 100)
        d_percent = mean(k_values) if k_values else 50
    else:
        d_percent = k_percent
    
    return k_percent, d_percent

# ====================== INSTITUTIONAL ALGORITHMS ======================

class Signal:
    """Enhanced signal class"""
    def __init__(self, symbol, side, confidence, entry_price, stop_loss, take_profit, 
                 reasoning, timestamp, algorithm, additional_data=None):
        self.symbol = symbol
        self.side = side
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.timestamp = timestamp
        self.algorithm = algorithm
        self.additional_data = additional_data or {}

class GoldmanSachsAlgorithm:
    """Goldman Sachs - Momentum & Mean Reversion Hybrid"""
    
    def __init__(self):
        self.name = "Goldman Sachs"
        self.lookback_periods = [5, 10, 20, 50]
        self.momentum_threshold = 0.02
        self.mean_reversion_threshold = 2.0  # Z-score
    
    def generate_signal(self, symbol, klines_data):
        """
        GS Strategy: Combines momentum and mean reversion
        - Momentum: Price breaks above/below multiple moving averages
        - Mean reversion: Price is oversold/overbought (Z-score)
        """
        try:
            if len(klines_data) < 50:
                return None
            
            closes = [float(k[4]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Calculate multiple timeframe momentum
            momentum_scores = []
            for period in self.lookback_periods:
                if len(closes) >= period:
                    old_price = closes[-period]
                    momentum = (current_price - old_price) / old_price
                    momentum_scores.append(momentum)
            
            avg_momentum = mean(momentum_scores) if momentum_scores else 0
            
            # Mean reversion component
            price_sma = mean(closes[-20:]) if len(closes) >= 20 else current_price
            price_std = stdev(closes[-20:]) if len(closes) >= 20 else 0
            z_score_val = z_score(current_price, price_sma, price_std)
            
            # Volume confirmation
            avg_volume = mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Signal generation
            confidence = 0
            side = None
            
            # Strong momentum up + not extremely overbought
            if (avg_momentum > self.momentum_threshold and 
                z_score_val < 1.5 and volume_ratio > 1.2):
                side = 'LONG'
                confidence = min(0.85, 0.65 + abs(avg_momentum) * 10)
            
            # Strong momentum down + not extremely oversold
            elif (avg_momentum < -self.momentum_threshold and 
                  z_score_val > -1.5 and volume_ratio > 1.2):
                side = 'SHORT'
                confidence = min(0.85, 0.65 + abs(avg_momentum) * 10)
            
            # Mean reversion signals (contrarian)
            elif z_score_val > self.mean_reversion_threshold and volume_ratio > 1.1:
                side = 'SHORT'
                confidence = min(0.80, 0.60 + abs(z_score_val) * 0.1)
            
            elif z_score_val < -self.mean_reversion_threshold and volume_ratio > 1.1:
                side = 'LONG'
                confidence = min(0.80, 0.60 + abs(z_score_val) * 0.1)
            
            if not side or confidence < 0.65:
                return None
            
            # Dynamic stops based on volatility
            volatility = calculate_volatility([closes[i]/closes[i-1]-1 for i in range(1, len(closes))])
            vol_multiplier = max(1.0, min(3.0, volatility / 0.3))
            
            stop_distance = current_price * 0.015 * vol_multiplier
            profit_distance = current_price * 0.03 * vol_multiplier
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"GS: Mom={avg_momentum:.3f}, Z={z_score_val:.2f}, Vol={volume_ratio:.2f}",
                timestamp=time.time(),
                algorithm="Goldman Sachs",
                additional_data={
                    'momentum': avg_momentum,
                    'z_score': z_score_val,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility
                }
            )
            
        except Exception as e:
            return None

class JPMorganAlgorithm:
    """JPMorgan - Statistical Arbitrage & Correlation Trading"""
    
    def __init__(self):
        self.name = "JPMorgan"
        self.correlation_window = 30
        self.correlation_threshold = 0.7
        self.spread_zscore_threshold = 1.5
    
    def generate_signal(self, symbol, klines_data, market_data=None):
        """
        JPM Strategy: Statistical arbitrage based on correlations
        - Identifies correlation breakdowns between assets
        - Trades mean reversion of price spreads
        """
        try:
            if len(klines_data) < 50:
                return None
            
            closes = [float(k[4]) for k in klines_data]
            current_price = closes[-1]
            
            # Price momentum and trend
            sma_10 = mean(closes[-10:])
            sma_20 = mean(closes[-20:])
            sma_50 = mean(closes[-50:])
            
            # Relative Strength vs Market (using BTC as proxy)
            if symbol != 'BTCUSDT' and market_data and 'BTCUSDT' in market_data:
                btc_closes = [float(k[4]) for k in market_data['BTCUSDT'][-len(closes):]]
                
                # Calculate relative performance
                asset_returns = [(closes[i]/closes[i-1]-1) for i in range(1, min(len(closes), len(btc_closes)))]
                btc_returns = [(btc_closes[i]/btc_closes[i-1]-1) for i in range(1, len(asset_returns)+1)]
                
                if len(asset_returns) >= 20:
                    # Rolling correlation
                    recent_asset_returns = asset_returns[-20:]
                    recent_btc_returns = btc_returns[-20:]
                    
                    correlation = np.corrcoef(recent_asset_returns, recent_btc_returns)[0,1] if len(recent_asset_returns) == len(recent_btc_returns) else 0
                    
                    # Spread analysis
                    spread_values = [(asset_returns[i] - btc_returns[i]) for i in range(len(asset_returns))]
                    spread_mean = mean(spread_values[-30:]) if len(spread_values) >= 30 else 0
                    spread_std = stdev(spread_values[-30:]) if len(spread_values) >= 30 else 0
                    current_spread = asset_returns[-1] - btc_returns[-1]
                    spread_zscore = z_score(current_spread, spread_mean, spread_std) if spread_std > 0 else 0
                    
                    # Bollinger Band analysis
                    upper_bb, middle_bb, lower_bb = bollinger_bands(closes, 20, 2)
                    
                    confidence = 0
                    side = None
                    
                    # Signals based on statistical relationships
                    if (spread_zscore > self.spread_zscore_threshold and 
                        current_price > upper_bb and sma_10 > sma_20):
                        # Asset overperforming, expect reversion
                        side = 'SHORT'
                        confidence = min(0.82, 0.65 + abs(spread_zscore) * 0.08)
                    
                    elif (spread_zscore < -self.spread_zscore_threshold and 
                          current_price < lower_bb and sma_10 < sma_20):
                        # Asset underperforming, expect catch-up
                        side = 'LONG'
                        confidence = min(0.82, 0.65 + abs(spread_zscore) * 0.08)
                    
                    # Trend following when correlation breaks down
                    elif abs(correlation) < 0.3 and abs(spread_zscore) > 1.0:
                        if sma_10 > sma_20 > sma_50:
                            side = 'LONG'
                            confidence = 0.72
                        elif sma_10 < sma_20 < sma_50:
                            side = 'SHORT'
                            confidence = 0.72
                    
                    if not side or confidence < 0.65:
                        return None
                    
                    # Dynamic position sizing based on correlation
                    vol_adj = max(0.8, min(1.5, 1.0 + abs(correlation)))
                    stop_distance = current_price * 0.02 * vol_adj
                    profit_distance = current_price * 0.035 * vol_adj
                    
                    if side == 'LONG':
                        stop_loss = current_price - stop_distance
                        take_profit = current_price + profit_distance
                    else:
                        stop_loss = current_price + stop_distance
                        take_profit = current_price - profit_distance
                    
                    return Signal(
                        symbol=symbol,
                        side=side,
                        confidence=confidence,
                        entry_price=Decimal(str(current_price)),
                        stop_loss=Decimal(str(stop_loss)),
                        take_profit=Decimal(str(take_profit)),
                        reasoning=f"JPM: Corr={correlation:.2f}, SpreadZ={spread_zscore:.2f}",
                        timestamp=time.time(),
                        algorithm="JPMorgan",
                        additional_data={
                            'correlation': correlation,
                            'spread_zscore': spread_zscore,
                            'relative_performance': current_spread
                        }
                    )
            
            # Fallback to technical analysis if no market data
            rsi_val = rsi(closes, 14)
            macd_line, signal_line, histogram = macd(closes)
            
            if rsi_val < 30 and macd_line > signal_line and sma_10 > sma_20:
                side = 'LONG'
                confidence = 0.70
            elif rsi_val > 70 and macd_line < signal_line and sma_10 < sma_20:
                side = 'SHORT'
                confidence = 0.70
            else:
                return None
            
            stop_distance = current_price * 0.02
            profit_distance = current_price * 0.035
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"JPM: RSI={rsi_val:.1f}, MACD={macd_line:.4f}",
                timestamp=time.time(),
                algorithm="JPMorgan"
            )
            
        except Exception as e:
            return None

class CitadelAlgorithm:
    """Citadel - High-Frequency Market Making & Volatility Trading"""
    
    def __init__(self):
        self.name = "Citadel"
        self.volatility_threshold = 0.25
        self.spread_threshold = 0.001
        self.volume_spike_threshold = 2.0
    
    def generate_signal(self, symbol, klines_data):
        """
        Citadel Strategy: Market making and volatility trading
        - Identifies volatility regimes
        - Trades volatility mean reversion
        - Volume-based momentum
        """
        try:
            if len(klines_data) < 30:
                return None
            
            closes = [float(k[4]) for k in klines_data]
            highs = [float(k[2]) for k in klines_data]
            lows = [float(k[3]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            
            current_price = closes[-1]
            
            # Volatility analysis
            returns = calculate_returns(closes)
            current_vol = calculate_volatility(returns[-20:], 20) if len(returns) >= 20 else 0
            long_vol = calculate_volatility(returns, min(len(returns), 50)) if len(returns) >= 10 else current_vol
            
            # True Range and ATR
            true_ranges = []
            for i in range(1, len(klines_data)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
            
            atr = mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
            atr_ratio = atr / current_price if current_price > 0 else 0
            
            # Volume profile
            avg_volume = mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Stochastic for overbought/oversold
            stoch_k, stoch_d = stochastic(highs, lows, closes)
            
            # Price action patterns
            recent_closes = closes[-5:]
            price_momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] if len(recent_closes) >= 5 else 0
            
            confidence = 0
            side = None
            
            # Volatility breakout strategy
            if (current_vol > long_vol * 1.3 and volume_spike > self.volume_spike_threshold):
                if price_momentum > 0.01 and stoch_k < 80:
                    side = 'LONG'
                    confidence = min(0.88, 0.70 + volume_spike * 0.05)
                elif price_momentum < -0.01 and stoch_k > 20:
                    side = 'SHORT'
                    confidence = min(0.88, 0.70 + volume_spike * 0.05)
            
            # Volatility mean reversion
            elif (current_vol > self.volatility_threshold and current_vol > long_vol * 1.5):
                if stoch_k > 80:
                    side = 'SHORT'
                    confidence = 0.75
                elif stoch_k < 20:
                    side = 'LONG'
                    confidence = 0.75
            
            # Low volatility momentum
            elif current_vol < long_vol * 0.7 and volume_spike > 1.5:
                sma_5 = mean(closes[-5:])
                sma_15 = mean(closes[-15:])
                
                if sma_5 > sma_15 and stoch_k > 50:
                    side = 'LONG'
                    confidence = 0.72
                elif sma_5 < sma_15 and stoch_k < 50:
                    side = 'SHORT'
                    confidence = 0.72
            
            if not side or confidence < 0.65:
                return None
            
            # Adaptive stops based on ATR
            atr_multiplier = max(1.5, min(3.0, current_vol / 0.2))
            stop_distance = atr * atr_multiplier
            profit_distance = atr * atr_multiplier * 2
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"Citadel: Vol={current_vol:.3f}, ATR={atr_ratio:.4f}, VolSpike={volume_spike:.2f}",
                timestamp=time.time(),
                algorithm="Citadel",
                additional_data={
                    'volatility': current_vol,
                    'atr_ratio': atr_ratio,
                    'volume_spike': volume_spike,
                    'stochastic': stoch_k
                }
            )
            
        except Exception as e:
            return None

class RenaissanceAlgorithm:
    """Renaissance Technologies - Mathematical Pattern Recognition"""
    
    def __init__(self):
        self.name = "Renaissance"
        self.pattern_length = 10
        self.similarity_threshold = 0.85
        self.prediction_horizon = 5
    
    def generate_signal(self, symbol, klines_data):
        """
        Renaissance Strategy: Pattern recognition and mathematical models
        - Fractal pattern matching
        - Mathematical sequence recognition
        - Non-linear pattern analysis
        """
        try:
            if len(klines_data) < 60:
                return None
            
            closes = [float(k[4]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Normalize prices for pattern matching
            normalized_prices = []
            window_size = 20
            for i in range(window_size, len(closes)):
                window = closes[i-window_size:i]
                mean_price = mean(window)
                std_price = stdev(window) if len(window) > 1 else 1
                if std_price > 0:
                    normalized_prices.append((closes[i] - mean_price) / std_price)
                else:
                    normalized_prices.append(0)
            
            if len(normalized_prices) < 30:
                return None
            
            # Pattern matching using correlation
            current_pattern = normalized_prices[-self.pattern_length:]
            best_correlation = 0
            best_match_index = 0
            
            # Find similar historical patterns
            for i in range(self.pattern_length, len(normalized_prices) - self.prediction_horizon):
                historical_pattern = normalized_prices[i-self.pattern_length:i]
                
                if len(historical_pattern) == len(current_pattern):
                    correlation = np.corrcoef(current_pattern, historical_pattern)[0,1]
                    if not np.isnan(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_match_index = i
            
            # Fibonacci retracement levels
            recent_high = max(closes[-50:])
            recent_low = min(closes[-50:])
            fib_levels = {
                0.236: recent_low + 0.236 * (recent_high - recent_low),
                0.382: recent_low + 0.382 * (recent_high - recent_low),
                0.618: recent_low + 0.618 * (recent_high - recent_low),
                0.786: recent_low + 0.786 * (recent_high - recent_low)
            }
            
            # Golden ratio analysis
            phi = (1 + math.sqrt(5)) / 2
            price_ratios = []
            for i in range(5, len(closes)):
                if closes[i-5] > 0:
                    ratio = closes[i] / closes[i-5]
                    price_ratios.append(ratio)
            
            # Mathematical sequence detection
            recent_ratios = price_ratios[-10:] if len(price_ratios) >= 10 else price_ratios
            ratio_mean = mean(recent_ratios) if recent_ratios else 1
            
            confidence = 0
            side = None
            
            # Pattern-based prediction
            if best_correlation > self.similarity_threshold:
                # Look at what happened after the best matching pattern
                if best_match_index + self.prediction_horizon < len(closes):
                    future_start = len(closes) - len(normalized_prices) + best_match_index
                    future_end = future_start + self.prediction_horizon
                    
                    if future_end < len(closes):
                        historical_future = closes[future_start:future_end]
                        if len(historical_future) >= 3:
                            future_direction = (historical_future[-1] - historical_future[0]) / historical_future[0]
                            
                            if future_direction > 0.02:
                                side = 'LONG'
                                confidence = min(0.85, 0.65 + best_correlation * 0.2)
                            elif future_direction < -0.02:
                                side = 'SHORT'
                                confidence = min(0.85, 0.65 + best_correlation * 0.2)
            
            # Fibonacci level trading
            if not side:
                for level, price in fib_levels.items():
                    if abs(current_price - price) / current_price < 0.01:  # Near fib level
                        if level in [0.382, 0.618]:  # Key retracement levels
                            trend = 1 if closes[-1] > closes[-10] else -1
                            if trend > 0:
                                side = 'LONG'
                                confidence = 0.70
                            else:
                                side = 'SHORT'
                                confidence = 0.70
                            break
            
            # Golden ratio momentum
            if not side and abs(ratio_mean - phi) < 0.1:
                price_change = (closes[-1] - closes[-5]) / closes[-5]
                if price_change > 0.015:
                    side = 'LONG'
                    confidence = 0.72
                elif price_change < -0.015:
                    side = 'SHORT'
                    confidence = 0.72
            
            if not side or confidence < 0.65:
                return None
            
            # Mathematical stop placement
            volatility = stdev(closes[-20:]) / mean(closes[-20:]) if len(closes) >= 20 else 0.02
            phi_distance = current_price * volatility * phi / 10
            
            if side == 'LONG':
                stop_loss = current_price - phi_distance
                take_profit = current_price + phi_distance * phi
            else:
                stop_loss = current_price + phi_distance
                take_profit = current_price - phi_distance * phi
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"RenTech: Pattern={best_correlation:.3f}, Phi={ratio_mean:.3f}",
                timestamp=time.time(),
                algorithm="Renaissance",
                additional_data={
                    'pattern_correlation': best_correlation,
                    'phi_ratio': ratio_mean,
                    'fib_level': min(fib_levels.values(), key=lambda x: abs(x - current_price))
                }
            )
            
        except Exception as e:
            return None

class TwoSigmaAlgorithm:
    """Two Sigma - Machine Learning Enhanced Signals"""
    
    def __init__(self):
        self.name = "Two Sigma"
        self.feature_window = 30
        self.ensemble_threshold = 0.6
    
    def generate_signal(self, symbol, klines_data):
        """
        Two Sigma Strategy: ML-inspired feature engineering
        - Multiple feature extraction
        - Ensemble decision making
        - Statistical learning approach
        """
        try:
            if len(klines_data) < 50:
                return None
            
            closes = [float(k[4]) for k in klines_data]
            highs = [float(k[2]) for k in klines_data]
            lows = [float(k[3]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Feature Engineering
            features = {}
            
            # Price-based features
            features['sma_5'] = mean(closes[-5:])
            features['sma_10'] = mean(closes[-10:])
            features['sma_20'] = mean(closes[-20:])
            features['sma_50'] = mean(closes[-50:]) if len(closes) >= 50 else features['sma_20']
            
            # Momentum features
            features['roc_5'] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0
            features['roc_10'] = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 else 0
            features['roc_20'] = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 else 0
            
            # Volatility features
            returns = calculate_returns(closes)
            features['volatility_10'] = stdev(returns[-10:]) if len(returns) >= 10 else 0
            features['volatility_20'] = stdev(returns[-20:]) if len(returns) >= 20 else features['volatility_10']
            
            # Volume features
            features['volume_ratio'] = volumes[-1] / mean(volumes[-10:]) if len(volumes) >= 10 else 1
            features['volume_trend'] = (mean(volumes[-5:]) - mean(volumes[-10:])) / mean(volumes[-10:]) if len(volumes) >= 10 else 0
            
            # Technical indicators
            features['rsi'] = rsi(closes, 14)
            macd_line, signal_line, histogram = macd(closes)
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            
            # Bollinger Bands
            upper_bb, middle_bb, lower_bb = bollinger_bands(closes, 20, 2)
            if upper_bb and lower_bb:
                features['bb_position'] = (current_price - lower_bb) / (upper_bb - lower_bb)
                features['bb_width'] = (upper_bb - lower_bb) / middle_bb
            else:
                features['bb_position'] = 0.5
                features['bb_width'] = 0.04
            
            # Stochastic
            stoch_k, stoch_d = stochastic(highs, lows, closes)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            
            # Support/Resistance levels
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            resistance = max(recent_highs)
            support = min(recent_lows)
            features['support_distance'] = (current_price - support) / current_price
            features['resistance_distance'] = (resistance - current_price) / current_price
            
            # Ensemble Models (simplified ML approach)
            models = []
            
            # Model 1: Trend Following
            trend_score = 0
            if features['sma_5'] > features['sma_10'] > features['sma_20']:
                trend_score += 1
            if features['roc_10'] > 0.01:
                trend_score += 1
            if features['volume_ratio'] > 1.2:
                trend_score += 1
            if features['macd'] > features['macd_signal']:
                trend_score += 1
            models.append(('LONG' if trend_score >= 3 else 'SHORT' if trend_score <= 1 else 'NEUTRAL', trend_score / 4))
            
            # Model 2: Mean Reversion
            reversion_score = 0
            if features['rsi'] < 30:
                reversion_score += 1
            if features['bb_position'] < 0.2:
                reversion_score += 1
            if features['stoch_k'] < 20:
                reversion_score += 1
            if features['support_distance'] < 0.02:
                reversion_score += 1
            models.append(('LONG' if reversion_score >= 3 else 'SHORT' if features['rsi'] > 70 and features['bb_position'] > 0.8 else 'NEUTRAL', reversion_score / 4))
            
            # Model 3: Momentum
            momentum_score = 0
            if features['roc_5'] > 0.005 and features['roc_10'] > 0.01:
                momentum_score += 1
            if features['volume_trend'] > 0.1:
                momentum_score += 1
            if features['volatility_10'] > features['volatility_20'] * 1.2:
                momentum_score += 1
            if features['macd_histogram'] > 0:
                momentum_score += 1
            models.append(('LONG' if momentum_score >= 3 else 'SHORT' if momentum_score <= 1 else 'NEUTRAL', momentum_score / 4))
            
            # Model 4: Statistical
            stat_score = 0
            z_price = z_score(current_price, features['sma_20'], features['volatility_20'] * features['sma_20'])
            if -1 < z_price < 1:  # Normal range
                if features['roc_10'] > 0:
                    stat_score += 1
            elif z_price < -1.5:  # Oversold
                stat_score += 1
            elif z_price > 1.5:  # Overbought
                stat_score -= 1
            
            if features['bb_width'] > 0.06:  # High volatility
                if abs(features['roc_5']) > 0.01:
                    stat_score += 1
            
            models.append(('LONG' if stat_score >= 1 else 'SHORT' if stat_score <= -1 else 'NEUTRAL', abs(stat_score) / 2))
            
            # Ensemble Decision
            long_votes = sum(1 for model, conf in models if model == 'LONG')
            short_votes = sum(1 for model, conf in models if model == 'SHORT')
            avg_confidence = mean([conf for model, conf in models])
            
            if long_votes >= 3:
                side = 'LONG'
                confidence = min(0.85, 0.65 + avg_confidence * 0.25)
            elif short_votes >= 3:
                side = 'SHORT'
                confidence = min(0.85, 0.65 + avg_confidence * 0.25)
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # Dynamic position sizing based on model agreement
            model_agreement = max(long_votes, short_votes) / len(models)
            vol_adj = max(0.8, min(1.5, model_agreement))
            
            stop_distance = current_price * 0.018 * vol_adj
            profit_distance = current_price * 0.032 * vol_adj
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"TwoSigma: L={long_votes}, S={short_votes}, Conf={avg_confidence:.2f}",
                timestamp=time.time(),
                algorithm="Two Sigma",
                additional_data={
                    'long_votes': long_votes,
                    'short_votes': short_votes,
                    'model_agreement': model_agreement,
                    'features': features
                }
            )
            
        except Exception as e:
            return None

# Continue with remaining algorithms...
class DEShawAlgorithm:
    """D.E. Shaw - Quantitative Factor Models"""
    
    def __init__(self):
        self.name = "D.E. Shaw"
        self.factor_lookback = 40
        
    def generate_signal(self, symbol, klines_data):
        try:
            if len(klines_data) < 50:
                return None
                
            closes = [float(k[4]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Factor 1: Value (Price vs Moving Average)
            sma_20 = mean(closes[-20:])
            value_factor = (sma_20 - current_price) / current_price
            
            # Factor 2: Quality (Volatility-adjusted returns)
            returns = calculate_returns(closes)
            avg_return = mean(returns[-20:]) if len(returns) >= 20 else 0
            vol_return = stdev(returns[-20:]) if len(returns) >= 20 else 0.01
            quality_factor = avg_return / vol_return if vol_return > 0 else 0
            
            # Factor 3: Momentum (Multi-timeframe)
            mom_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0
            mom_20 = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 else 0
            momentum_factor = (mom_5 + mom_20) / 2
            
            # Factor 4: Low Volatility
            vol_factor = -vol_return  # Negative because we want low vol
            
            # Factor 5: Profitability (Volume-Price trend)
            price_volume_corr = 0
            if len(closes) >= 20 and len(volumes) >= 20:
                recent_prices = closes[-20:]
                recent_volumes = volumes[-20:]
                price_volume_corr = np.corrcoef(recent_prices, recent_volumes)[0,1]
                if np.isnan(price_volume_corr):
                    price_volume_corr = 0
            
            # Combine factors with weights
            factor_score = (
                0.2 * value_factor +
                0.25 * quality_factor +
                0.3 * momentum_factor +
                0.15 * vol_factor +
                0.1 * price_volume_corr
            )
            
            # Generate signal based on factor score
            if factor_score > 0.02:
                side = 'LONG'
                confidence = min(0.80, 0.65 + abs(factor_score) * 10)
            elif factor_score < -0.02:
                side = 'SHORT'
                confidence = min(0.80, 0.65 + abs(factor_score) * 10)
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # Risk-adjusted stops
            stop_distance = current_price * max(0.015, vol_return * 100)
            profit_distance = stop_distance * 2
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"DEShaw: FactorScore={factor_score:.4f}",
                timestamp=time.time(),
                algorithm="D.E. Shaw",
                additional_data={'factor_score': factor_score}
            )
            
        except Exception as e:
            return None

class BridgewaterAlgorithm:
    """Bridgewater - All Weather/Risk Parity"""
    
    def __init__(self):
        self.name = "Bridgewater"
        
    def generate_signal(self, symbol, klines_data):
        try:
            if len(klines_data) < 50:
                return None
                
            closes = [float(k[4]) for k in klines_data]
            current_price = closes[-1]
            
            # Economic regime detection (simplified)
            returns = calculate_returns(closes)
            
            # Growth proxy (momentum)
            growth_signal = mean(returns[-10:]) if len(returns) >= 10 else 0
            
            # Inflation proxy (volatility)
            inflation_signal = stdev(returns[-20:]) if len(returns) >= 20 else 0
            
            # Trend strength
            sma_10 = mean(closes[-10:])
            sma_50 = mean(closes[-50:]) if len(closes) >= 50 else sma_10
            trend_strength = (sma_10 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Risk parity signal
            if growth_signal > 0.001 and inflation_signal < 0.03:
                # Growth environment
                side = 'LONG'
                confidence = 0.72
            elif growth_signal < -0.001 and inflation_signal > 0.02:
                # Stagflation environment
                side = 'SHORT'
                confidence = 0.72
            elif abs(trend_strength) > 0.05:
                # Strong trend
                side = 'LONG' if trend_strength > 0 else 'SHORT'
                confidence = min(0.75, 0.65 + abs(trend_strength) * 2)
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # All-weather position sizing
            vol_target = 0.02  # 2% volatility target
            position_vol = inflation_signal if inflation_signal > 0 else 0.02
            vol_scalar = vol_target / position_vol
            
            stop_distance = current_price * 0.025 / vol_scalar
            profit_distance = current_price * 0.04 / vol_scalar
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"Bridgewater: Growth={growth_signal:.4f}, Trend={trend_strength:.3f}",
                timestamp=time.time(),
                algorithm="Bridgewater"
            )
            
        except Exception as e:
            return None

class AQRAlgorithm:
    """AQR - Factor Investing & Style Premia"""
    
    def __init__(self):
        self.name = "AQR"
        
    def generate_signal(self, symbol, klines_data):
        try:
            if len(klines_data) < 50:
                return None
                
            closes = [float(k[4]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Momentum Style
            mom_1m = (closes[-1] - closes[-22]) / closes[-22] if len(closes) > 22 else 0
            mom_3m = (closes[-1] - closes[-66]) / closes[-66] if len(closes) > 66 else mom_1m
            momentum_score = (mom_1m + mom_3m) / 2
            
            # Value Style (Price vs fundamentals proxy)
            sma_100 = mean(closes[-100:]) if len(closes) >= 100 else mean(closes)
            value_score = (sma_100 - current_price) / current_price
            
            # Quality Style (Earnings quality proxy using volume)
            avg_volume = mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_quality = volumes[-1] / avg_volume if avg_volume > 0 else 1
            quality_score = min(2.0, volume_quality) - 1  # Center around 0
            
            # Low Beta Style
            returns = calculate_returns(closes)
            return_vol = stdev(returns[-30:]) if len(returns) >= 30 else 0.02
            low_beta_score = -return_vol  # Negative vol is good
            
            # Combine style factors
            style_score = (
                0.4 * momentum_score +
                0.2 * value_score +
                0.2 * quality_score +
                0.2 * low_beta_score
            )
            
            if style_score > 0.015:
                side = 'LONG'
                confidence = min(0.78, 0.65 + abs(style_score) * 8)
            elif style_score < -0.015:
                side = 'SHORT'
                confidence = min(0.78, 0.65 + abs(style_score) * 8)
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # Style-based risk management
            stop_distance = current_price * max(0.02, return_vol * 50)
            profit_distance = stop_distance * 1.8
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"AQR: Style={style_score:.4f}, Mom={momentum_score:.3f}",
                timestamp=time.time(),
                algorithm="AQR"
            )
            
        except Exception as e:
            return None

class WintonAlgorithm:
    """Winton - Systematic CTA Strategies"""
    
    def __init__(self):
        self.name = "Winton"
        
    def generate_signal(self, symbol, klines_data):
        try:
            if len(klines_data) < 50:
                return None
                
            closes = [float(k[4]) for k in klines_data]
            current_price = closes[-1]
            
            # Multiple timeframe trend following
            signals = []
            timeframes = [5, 10, 20, 50]
            
            for tf in timeframes:
                if len(closes) >= tf:
                    old_price = closes[-tf]
                    trend = (current_price - old_price) / old_price
                    signals.append(1 if trend > 0 else -1)
                    
            # Breakout detection
            recent_high = max(closes[-20:]) if len(closes) >= 20 else current_price
            recent_low = min(closes[-20:]) if len(closes) >= 20 else current_price
            
            breakout_signal = 0
            if current_price > recent_high * 0.995:  # Near recent high
                breakout_signal = 1
            elif current_price < recent_low * 1.005:  # Near recent low
                breakout_signal = -1
            
            # Volatility filter
            returns = calculate_returns(closes)
            vol = stdev(returns[-20:]) if len(returns) >= 20 else 0.02
            vol_percentile = vol / 0.05  # Normalize to typical crypto vol
            
            # Combine signals
            trend_signal = mean(signals) if signals else 0
            
            # Only trade in medium volatility environments
            if 0.3 < vol_percentile < 2.0:
                if trend_signal > 0.5 and breakout_signal >= 0:
                    side = 'LONG'
                    confidence = min(0.76, 0.65 + abs(trend_signal) * 0.15)
                elif trend_signal < -0.5 and breakout_signal <= 0:
                    side = 'SHORT'
                    confidence = min(0.76, 0.65 + abs(trend_signal) * 0.15)
                else:
                    return None
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # CTA-style risk management
            atr_approx = (recent_high - recent_low) / 20
            stop_distance = atr_approx * 2
            profit_distance = atr_approx * 3
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"Winton: Trend={trend_signal:.2f}, Vol={vol_percentile:.2f}",
                timestamp=time.time(),
                algorithm="Winton"
            )
            
        except Exception as e:
            return None

class ManGroupAlgorithm:
    """Man Group - Multi-Strategy Quantitative"""
    
    def __init__(self):
        self.name = "Man Group"
        
    def generate_signal(self, symbol, klines_data):
        try:
            if len(klines_data) < 50:
                return None
                
            closes = [float(k[4]) for k in klines_data]
            volumes = [float(k[5]) for k in klines_data]
            current_price = closes[-1]
            
            # Strategy 1: Trend Following
            sma_20 = mean(closes[-20:])
            sma_50 = mean(closes[-50:]) if len(closes) >= 50 else sma_20
            trend_score = 1 if sma_20 > sma_50 else -1
            
            # Strategy 2: Mean Reversion
            upper_bb, middle_bb, lower_bb = bollinger_bands(closes, 20, 2)
            if upper_bb and lower_bb:
                bb_position = (current_price - lower_bb) / (upper_bb - lower_bb)
                if bb_position > 0.8:
                    reversion_score = -1  # Sell high
                elif bb_position < 0.2:
                    reversion_score = 1   # Buy low
                else:
                    reversion_score = 0
            else:
                reversion_score = 0
            
            # Strategy 3: Momentum
            rsi_val = rsi(closes, 14)
            momentum_score = 0
            if rsi_val > 60 and rsi_val < 80:
                momentum_score = 1
            elif rsi_val > 20 and rsi_val < 40:
                momentum_score = -1
            
            # Strategy 4: Volume Analysis
            avg_volume = mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            volume_score = 1 if volumes[-1] > avg_volume * 1.5 else 0
            
            # Multi-strategy combination
            strategies = [trend_score, reversion_score, momentum_score]
            strategy_agreement = sum(1 for s in strategies if s > 0) - sum(1 for s in strategies if s < 0)
            
            # Final signal with volume confirmation
            if strategy_agreement >= 2 and volume_score > 0:
                side = 'LONG'
                confidence = min(0.80, 0.65 + abs(strategy_agreement) * 0.08)
            elif strategy_agreement <= -2 and volume_score > 0:
                side = 'SHORT'
                confidence = min(0.80, 0.65 + abs(strategy_agreement) * 0.08)
            else:
                return None
            
            if confidence < 0.65:
                return None
            
            # Multi-strategy risk management
            returns = calculate_returns(closes)
            portfolio_vol = stdev(returns[-20:]) if len(returns) >= 20 else 0.02
            
            stop_distance = current_price * max(0.02, portfolio_vol * 30)
            profit_distance = stop_distance * 2
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=Decimal(str(current_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                reasoning=f"ManGroup: Agreement={strategy_agreement}, Vol={volume_score}",
                timestamp=time.time(),
                algorithm="Man Group"
            )
            
        except Exception as e:
            return None

# ====================== ENHANCED TRADING BOT ======================
# [Rest of the original bot code remains the same, just need to integrate the algorithm classes]

class InstitutionalTradingBot:
    """Enhanced trading bot with institutional algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize algorithms
        self.algorithms = {}
        if Config.GOLDMAN_ENABLED:
            self.algorithms['Goldman'] = GoldmanSachsAlgorithm()
        if Config.JPMORGAN_ENABLED:
            self.algorithms['JPMorgan'] = JPMorganAlgorithm()
        if Config.CITADEL_ENABLED:
            self.algorithms['Citadel'] = CitadelAlgorithm()
        if Config.RENAISSANCE_ENABLED:
            self.algorithms['Renaissance'] = RenaissanceAlgorithm()
        if Config.TWOSIGMA_ENABLED:
            self.algorithms['TwoSigma'] = TwoSigmaAlgorithm()
        if Config.DESHAW_ENABLED:
            self.algorithms['DEShaw'] = DEShawAlgorithm()
        if Config.BRIDGEWATER_ENABLED:
            self.algorithms['Bridgewater'] = BridgewaterAlgorithm()
        if Config.AQR_ENABLED:
            self.algorithms['AQR'] = AQRAlgorithm()
        if Config.WINTON_ENABLED:
            self.algorithms['Winton'] = WintonAlgorithm()
        if Config.MANGROUP_ENABLED:
            self.algorithms['ManGroup'] = ManGroupAlgorithm()
        
        self.logger.info(f"✅ Loaded {len(self.algorithms)} institutional algorithms")
        
        # [Rest of initialization same as original bot]
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
        self.market_data_cache = {}
        
        self._init_database()

    def generate_institutional_signals(self, symbol):
        """Generate signals from all enabled institutional algorithms"""
        try:
            # Get market data
            klines = self.binance.get_klines(symbol, '5m', Config.KLINE_HISTORY_LIMIT)
            if len(klines) < 30:
                return []
            
            signals = []
            
            # Run each algorithm
            for algo_name, algorithm in self.algorithms.items():
                try:
                    if algo_name == 'JPMorgan':
                        # JPMorgan needs market data for correlation analysis
                        signal = algorithm.generate_signal(symbol, klines, self.market_data_cache)
                    else:
                        signal = algorithm.generate_signal(symbol, klines)
                    
                    if signal and signal.confidence >= float(Config.SIGNAL_THRESHOLD):
                        signals.append(signal)
                        
                except Exception as e:
                    self.logger.debug(f"{algo_name} failed for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
            return []

    def aggregate_signals(self, signals):
        """Aggregate multiple institutional signals into one decision"""
        if not signals:
            return None
        
        # Group by side
        long_signals = [s for s in signals if s.side == 'LONG']
        short_signals = [s for s in signals if s.side == 'SHORT']
        
        # Weighted voting by confidence
        long_weight = sum(s.confidence for s in long_signals)
        short_weight = sum(s.confidence for s in short_signals)
        
        # Require majority consensus
        if long_weight > short_weight and len(long_signals) >= 2:
            # Use highest confidence signal as base
            best_signal = max(long_signals, key=lambda x: x.confidence)
            best_signal.confidence = min(0.90, long_weight / len(signals))
            best_signal.reasoning = f"Consensus: {len(long_signals)}L vs {len(short_signals)}S"
            return best_signal
            
        elif short_weight > long_weight and len(short_signals) >= 2:
            best_signal = max(short_signals, key=lambda x: x.confidence)
            best_signal.confidence = min(0.90, short_weight / len(signals))
            best_signal.reasoning = f"Consensus: {len(long_signals)}L vs {len(short_signals)}S"
            return best_signal
        
        return None

    def scan_for_signals(self):
        """Enhanced signal scanning with institutional algorithms"""
        try:
            # Update market data cache
            for symbol in Config.TRADING_PAIRS:
                try:
                    self.market_data_cache[symbol] = self.binance.get_klines(symbol, '5m', 50)
                except Exception:
                    continue
            
            max_scans = max(3, Config.MAX_POSITIONS - len(self.positions))
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
                    
                    # Generate signals from all algorithms
                    signals = self.generate_institutional_signals(symbol)
                    
                    # Aggregate signals
                    final_signal = self.aggregate_signals(signals)
                    
                    if final_signal:
                        self.logger.info(f"Signal: {symbol} {final_signal.side} from {len(signals)} algos")
                        self.open_position(final_signal)
                    
                    time.sleep(1.0)  # Rate limiting
                    
                except Exception as e:
                    self.logger.debug(f"Scan failed {symbol}: {e}")
                    continue
                
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Scanning failed: {e}")

    # [Rest of the methods remain the same as original bot - get_balance, get_positions, etc.]

# ====================== [KEEP ALL ORIGINAL FLASK AND MAIN CODE] ======================
# The rest of the code (Flask app, templates, main function) remains exactly the same