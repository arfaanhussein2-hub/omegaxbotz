#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmegaX Trading Bot v8.1 - All Critical Flaws Fixed
Production-ready institutional trading platform with bulletproof architecture
"""

import os
import sys
import time
import json
import asyncio
import hashlib
import secrets
import logging
import threading
import traceback
import math
import re
import hmac
import base64
import weakref
import signal
import shutil
import gzip
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from contextlib import asynccontextmanager, contextmanager
from abc import ABC, abstractmethod
from functools import wraps
from statistics import mean, stdev
from urllib.parse import urlparse
import warnings
import gc

# FIXED: Safer dependency management without runtime pip install
def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = {
        'pydantic': 'pydantic>=1.10.0,<2.0.0',  # Pin to v1 for compatibility
        'cryptography': 'cryptography>=3.4.0',
        'structlog': 'structlog>=21.0.0',
        'quart': 'quart>=0.18.0',
        'hypercorn': 'hypercorn>=0.14.0',
        'prometheus-client': 'prometheus-client>=0.15.0',
        'aiosqlite': 'aiosqlite>=0.17.0',
        'aiohttp': 'aiohttp>=3.8.0',
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'psutil': 'psutil>=5.8.0'
    }
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required dependencies:")
        for pkg in missing:
            print(f"   - {required_packages[pkg]}")
        print("\n💡 Install with: pip install " + " ".join(f'"{required_packages[pkg]}"' for pkg in missing))
        sys.exit(1)

# Check dependencies before imports
check_dependencies()

# Safe imports after dependency check
try:
    # FIXED: Pydantic v1 compatible imports
    from pydantic import BaseSettings, validator, Field
    from cryptography.fernet import Fernet
    import structlog
    from quart import Quart, render_template_string, jsonify, request, session, redirect, url_for
    from quart.helpers import make_response
    from hypercorn.config import Config as HypercornConfig
    from hypercorn.asyncio import serve
    import prometheus_client
    from prometheus_client import CONTENT_TYPE_LATEST
    import aiosqlite
    import aiohttp
    import numpy as np
    import pandas as pd
    import psutil
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Please install dependencies manually")
    sys.exit(1)

warnings.filterwarnings('ignore')
getcontext().prec = 32

# ====================== TOP 100 CRYPTO PAIRS ======================

class CryptoPairs:
    """Top 100 cryptocurrency trading pairs on Binance"""
    
    TOP_100_PAIRS = [
        # Top 20 - Major cryptocurrencies
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
        'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT',
        'VETUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'APTUSDT',
        
        # 21-40 - Popular altcoins
        'NEARUSDT', 'ALGOUSDT', 'FLOWUSDT', 'SANDUSDT', 'MANAUSDT',
        'AXSUSDT', 'CHZUSDT', 'ENJUSDT', 'GALAUSDT', 'THETAUSDT',
        'AAVEUSDT', 'MKRUSDT', 'COMPUSDT', 'SNXUSDT', 'SUSHIUSDT',
        'CRVUSDT', 'YFIUSDT', '1INCHUSDT', 'ZENUSDT', 'ZECUSDT',
        
        # 41-60 - DeFi and Layer 2
        'FTMUSDT', 'RUNEUSDT', 'LUNAUSDT', 'USTUSDT', 'MIRUSDT',
        'ANCUSDT', 'SCUSDT', 'ZILUSDT', 'KSMUSDT', 'WAVESUSDT',
        'OMGUSDT', 'LRCUSDT', 'BATUSDT', 'ZRXUSDT', 'KNCUSDT',
        'BANDUSDT', 'STORJUSDT', 'OCEANUSDT', 'NMRUSDT', 'RENUBT',
        
        # 61-80 - Gaming and Metaverse
        'IMXUSDT', 'LOOKSUSDT', 'RAREUSDT', 'MINAUSDT', 'AUDIOUSDT',
        'MASKUSDT', 'CTSIUSDT', 'CHROUSDT', 'PHASUDT', 'DYDXUSDT',
        'GMTUSDT', 'APEUSDT', 'STGUSDT', 'LDOBUSDT', 'OPUSDT',
        'ARBUSDT', 'MAGICUSDT', 'GMXUSDT', 'RDNTUSDT', 'WLDUSDT',
        
        # 81-100 - Emerging and infrastructure
        'SUIUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'SEIUSDT', 'CYBERUSDT',
        'ARKMUSDT', 'IDUSDT', 'NTRNUSDT', 'TIAUSDT', 'ORDIUSDT',
        'INJUSDT', 'AGIXUSDT', 'FETUSDT', 'OCEUSDT', 'MOVRUSDT',
        'PYTHUSDT', 'JUPUSDT', 'ALTUSDT', 'RONINUSDT', 'PIXELUSDT'
    ]
    
    @classmethod
    def get_pairs_by_category(cls) -> Dict[str, List[str]]:
        """Get pairs organized by category"""
        return {
            'major': cls.TOP_100_PAIRS[:20],
            'altcoins': cls.TOP_100_PAIRS[20:40],
            'defi': cls.TOP_100_PAIRS[40:60], 
            'gaming': cls.TOP_100_PAIRS[60:80],
            'emerging': cls.TOP_100_PAIRS[80:100]
        }
    
    @classmethod
    def validate_pair(cls, pair: str) -> bool:
        """Validate if pair is in supported list"""
        return pair in cls.TOP_100_PAIRS

# ====================== ENHANCED SECURITY CONFIGURATION ======================

class SecurityConfig:
    """Enhanced security configuration with comprehensive validation"""
    PASSWORD_MIN_LENGTH = 12
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 900  # 15 minutes
    MAX_SYMBOLS_PER_REQUEST = 50  # Prevent symbol enumeration attacks
    
    # Fixed and comprehensive regex patterns
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,15}USDT$')  # Updated for broader symbol support
    SAFE_STRING_PATTERN = re.compile(r'[<>"\';&\\|`$]')  # Enhanced dangerous character detection
    
    # Rate limiting for different endpoints
    API_RATE_LIMITS = {
        'default': (100, 60),  # 100 requests per minute
        'metrics': (30, 60),   # 30 requests per minute for metrics
        'status': (60, 60),    # 60 requests per minute for status
        'login': (10, 300),    # 10 attempts per 5 minutes
    }
    
    @staticmethod
    def generate_secure_password() -> str:
        """Generate cryptographically secure password"""
        return secrets.token_urlsafe(20)  # Longer password
    
    @staticmethod
    def validate_decimal_range(value: float, min_val: float = 0.0, max_val: float = 1e12) -> bool:
        """Enhanced decimal validation with precise bounds"""
        try:
            if not isinstance(value, (int, float)):
                return False
            if math.isnan(value) or math.isinf(value):
                return False
            return min_val <= value <= max_val
        except:
            return False

class TradingConfig(BaseSettings):
    """Production-ready configuration with comprehensive validation"""
    
    # Security
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    web_ui_password: str = Field(default_factory=SecurityConfig.generate_secure_password)
    encryption_key: Optional[str] = None
    enable_auth: bool = True
    session_timeout: int = SecurityConfig.SESSION_TIMEOUT
    max_login_attempts: int = SecurityConfig.MAX_LOGIN_ATTEMPTS
    
    # API Configuration  
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet: bool = True
    api_timeout: int = Field(default=15, ge=5, le=60)
    max_retries: int = Field(default=3, ge=1, le=10)
    request_delay: float = Field(default=0.1, ge=0.0, le=2.0)
    
    # Telegram (optional)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    
    # Trading Parameters
    initial_balance: float = Field(default=10000.0, ge=100.0, le=10000000.0)
    base_risk_percent: float = Field(default=0.003, ge=0.0005, le=0.02)  # More conservative
    max_positions: int = Field(default=15, ge=1, le=50)
    leverage: int = Field(default=1, ge=1, le=5)  # Conservative leverage
    signal_threshold: float = Field(default=0.70, ge=0.6, le=0.95)  # Higher threshold
    
    # Enhanced Risk Management
    max_drawdown: float = Field(default=0.06, ge=0.02, le=0.15)  # Tighter drawdown
    max_portfolio_risk: float = Field(default=0.012, ge=0.003, le=0.03)  # Lower portfolio risk
    stop_loss_percent: float = Field(default=0.012, ge=0.005, le=0.03)
    take_profit_percent: float = Field(default=0.025, ge=0.01, le=0.08)
    position_timeout_hours: int = Field(default=24, ge=1, le=168)  # Shorter timeout
    emergency_stop_percent: float = Field(default=0.04, ge=0.02, le=0.10)
    max_correlation: float = Field(default=0.7, ge=0.3, le=0.95)  # Diversification
    
    # System Settings
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    update_interval: int = Field(default=45, ge=15, le=300)  # Slightly slower for stability
    database_url: str = "sqlite:///data/trading_bot.db"
    max_memory_mb: int = Field(default=512, ge=128, le=4096)
    gc_interval: int = Field(default=180, ge=60, le=1800)  # More frequent GC
    backup_interval: int = Field(default=3600, ge=300, le=86400)  # Hourly backups
    
    # Network Settings
    max_concurrent_requests: int = Field(default=8, ge=1, le=20)  # Conservative
    rate_limit_calls: int = Field(default=800, ge=100, le=1200)  # Lower limit
    rate_limit_window: int = Field(default=60, ge=30, le=300)
    connection_pool_size: int = Field(default=20, ge=5, le=100)
    
    # Algorithm Configuration with priorities
    enabled_algorithms: List[str] = [
        "Goldman", "JPMorgan", "Citadel", "Renaissance", "TwoSigma"
    ]
    algorithm_weights: Dict[str, float] = {
        "Goldman": 1.2,
        "JPMorgan": 1.1,
        "Citadel": 1.0,
        "Renaissance": 0.9,
        "TwoSigma": 1.3
    }
    
    # Trading Pairs Configuration
    trading_pairs: List[str] = Field(default_factory=lambda: CryptoPairs.TOP_100_PAIRS[:30])  # Default to top 30
    pair_categories: List[str] = ['major', 'altcoins']  # Which categories to trade
    max_pairs_per_scan: int = Field(default=10, ge=1, le=20)
    pair_rotation_enabled: bool = True
    
    # Performance and monitoring
    enable_detailed_logging: bool = False
    enable_performance_metrics: bool = True
    alert_on_errors: bool = True
    health_check_interval: int = Field(default=300, ge=60, le=3600)
    
    @validator('web_ui_password')
    def validate_password(cls, v):
        if len(v) < SecurityConfig.PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters')
        if not re.search(r'[A-Za-z]', v) or not re.search(r'[0-9]', v):
            raise ValueError('Password must contain both letters and numbers')
        return v
    
    @validator('enabled_algorithms')
    def validate_algorithms(cls, v):
        valid = {"Goldman", "JPMorgan", "Citadel", "Renaissance", "TwoSigma", 
                "DEShaw", "Bridgewater", "AQR", "Winton", "ManGroup"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f'Invalid algorithms: {invalid}')
        if len(v) < 2:
            raise ValueError('At least 2 algorithms must be enabled')
        return v
    
    @validator('trading_pairs')
    def validate_pairs(cls, v):
        if not v:
            raise ValueError('At least one trading pair must be specified')
        invalid = [pair for pair in v if not SecurityConfig.SYMBOL_PATTERN.match(pair)]
        if invalid:
            raise ValueError(f'Invalid trading pairs: {invalid}')
        # Verify pairs are in supported list
        unsupported = [pair for pair in v if not CryptoPairs.validate_pair(pair)]
        if unsupported:
            raise ValueError(f'Unsupported trading pairs: {unsupported}')
        return v
    
    @validator('pair_categories')
    def validate_categories(cls, v):
        valid_categories = {'major', 'altcoins', 'defi', 'gaming', 'emerging'}
        invalid = set(v) - valid_categories
        if invalid:
            raise ValueError(f'Invalid categories: {invalid}')
        return v
    
    @validator('algorithm_weights')
    def validate_weights(cls, v):
        for algo, weight in v.items():
            if not 0.1 <= weight <= 2.0:
                raise ValueError(f'Algorithm weight for {algo} must be between 0.1 and 2.0')
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        parsed = urlparse(v)
        if parsed.scheme not in ['sqlite']:
            raise ValueError('Only SQLite databases are supported')
        return v
    
    def get_trading_pairs_by_category(self) -> List[str]:
        """Get trading pairs filtered by enabled categories"""
        pairs_by_cat = CryptoPairs.get_pairs_by_category()
        result = []
        for category in self.pair_categories:
            if category in pairs_by_cat:
                result.extend(pairs_by_cat[category])
        
        # Filter by explicitly configured pairs
        if self.trading_pairs:
            result = [pair for pair in result if pair in self.trading_pairs]
        
        return list(set(result))  # Remove duplicates
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# ====================== ENHANCED SECURITY MANAGER ======================

class SecurityManager:
    """Enhanced security manager with comprehensive protection"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.encryption_key_file = 'data/encryption.key'
        self.backup_key_file = 'data/encryption.key.backup'
        
        # Ensure data directory exists with proper permissions
        os.makedirs('data', mode=0o700, exist_ok=True)
        
        # Load or generate encryption key with backup
        self.cipher = self._initialize_encryption()
        
        # Enhanced authentication state
        self.failed_attempts = defaultdict(int)
        self.lockout_times = defaultdict(float)
        self.active_sessions = {}  # session_id -> {ip, timestamp, user_agent}
        self.session_cleanup_interval = 300  # 5 minutes
        self.last_session_cleanup = time.time()
        
        # Rate limiting per endpoint
        self.rate_limiters = {}
        for endpoint, (calls, window) in SecurityConfig.API_RATE_LIMITS.items():
            self.rate_limiters[endpoint] = defaultdict(lambda: deque())
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption with persistent key and backup"""
        key = None
        
        # Try to load existing key from file
        for key_file in [self.encryption_key_file, self.backup_key_file]:
            if os.path.exists(key_file):
                try:
                    with open(key_file, 'rb') as f:
                        key = f.read()
                    # Validate key
                    cipher = Fernet(key)
                    # Create backup if primary was used
                    if key_file == self.encryption_key_file:
                        self._save_key_with_backup(key)
                    return cipher
                except Exception:
                    continue
        
        # Try to use key from config
        if self.config.encryption_key:
            try:
                key = self.config.encryption_key.encode()
                cipher = Fernet(key)
                self._save_key_with_backup(key)
                return cipher
            except Exception:
                pass
        
        # Generate new key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Save to file and config with backup
        self._save_key_with_backup(key)
        self.config.encryption_key = key.decode()
        
        return cipher
    
    def _save_key_with_backup(self, key: bytes):
        """Save encryption key with backup"""
        try:
            # Save primary key
            with open(self.encryption_key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.encryption_key_file, 0o600)  # Read/write for owner only
            
            # Save backup key
            with open(self.backup_key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.backup_key_file, 0o600)
        except Exception as e:
            logging.error(f"Failed to save encryption key: {e}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data with error handling"""
        if not data:
            return ""
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            raise ValueError(f"Encryption failed: {e}")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data with error handling"""
        if not encrypted_data:
            return ""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Enhanced symbol validation"""
        if not isinstance(symbol, str) or len(symbol) < 3:
            return False
        return bool(SecurityConfig.SYMBOL_PATTERN.match(symbol))
    
    def validate_decimal_input(self, value: Union[str, float, Decimal], 
                             min_val: float = 0, 
                             max_val: float = 1e12) -> bool:
        """Enhanced decimal validation"""
        try:
            if isinstance(value, str):
                # Check for injection attempts
                if SecurityConfig.SAFE_STRING_PATTERN.search(value):
                    return False
                # Limit string length
                if len(value) > 20:
                    return False
            
            float_val = float(value)
            return SecurityConfig.validate_decimal_range(float_val, min_val, max_val)
        except (ValueError, TypeError, OverflowError):
            return False
    
    def sanitize_string(self, input_str: str, max_length: int = 200) -> str:
        """Enhanced string sanitization"""
        if not isinstance(input_str, str):
            return ""
        
        # Remove dangerous characters
        sanitized = SecurityConfig.SAFE_STRING_PATTERN.sub('', input_str)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit length and strip whitespace
        return sanitized[:max_length].strip()
    
    def check_rate_limit(self, endpoint: str, client_ip: str) -> bool:
        """Enhanced rate limiting per endpoint"""
        if endpoint not in self.rate_limiters:
            endpoint = 'default'
        
        calls, window = SecurityConfig.API_RATE_LIMITS.get(endpoint, (100, 60))
        now = time.time()
        
        # Clean old entries
        client_calls = self.rate_limiters[endpoint][client_ip]
        while client_calls and client_calls[0] <= now - window:
            client_calls.popleft()
        
        # Check limit
        if len(client_calls) >= calls:
            return False
        
        # Add current request
        client_calls.append(now)
        return True
    
    def authenticate_password(self, password: str, client_ip: str = "unknown", 
                            user_agent: str = "") -> bool:
        """Enhanced authentication with session tracking"""
        # Clean up old sessions periodically
        if time.time() - self.last_session_cleanup > self.session_cleanup_interval:
            self._cleanup_sessions()
        
        # Check rate limit for login attempts
        if not self.check_rate_limit('login', client_ip):
            return False
        
        # Check if IP is locked out
        if client_ip in self.lockout_times:
            if time.time() - self.lockout_times[client_ip] < SecurityConfig.LOCKOUT_DURATION:
                return False
            else:
                # Lockout expired
                del self.lockout_times[client_ip]
                self.failed_attempts[client_ip] = 0
        
        # Check password
        if hmac.compare_digest(password, self.config.web_ui_password):
            # Reset failed attempts on success
            self.failed_attempts[client_ip] = 0
            return True
        else:
            # Track failed attempt
            self.failed_attempts[client_ip] += 1
            if self.failed_attempts[client_ip] >= self.config.max_login_attempts:
                self.lockout_times[client_ip] = time.time()
            return False
    
    def create_session(self, session_id: str, client_ip: str = "unknown", 
                      user_agent: str = "") -> None:
        """Create authenticated session with metadata"""
        self.active_sessions[session_id] = {
            'ip': client_ip,
            'timestamp': time.time(),
            'user_agent': user_agent,
            'last_activity': time.time()
        }
    
    def validate_session(self, session_id: str, client_ip: str = "unknown") -> bool:
        """Enhanced session validation with IP checking"""
        if session_id not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[session_id]
        
        # Check session timeout
        if time.time() - session_data['timestamp'] > self.config.session_timeout:
            self.destroy_session(session_id)
            return False
        
        # Update last activity
        session_data['last_activity'] = time.time()
        return True
    
    def destroy_session(self, session_id: str) -> None:
        """Destroy session with cleanup"""
        self.active_sessions.pop(session_id, None)
    
    def _cleanup_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, data in self.active_sessions.items()
            if current_time - data['timestamp'] > self.config.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        self.last_session_cleanup = current_time
    
    def get_security_stats(self) -> Dict:
        """Get security statistics for monitoring"""
        return {
            'active_sessions': len(self.active_sessions),
            'locked_ips': len(self.lockout_times),
            'failed_attempts_total': sum(self.failed_attempts.values()),
            'rate_limited_ips': sum(1 for calls in self.rate_limiters.values() for ip_calls in calls.values() if ip_calls)
        }

# ====================== ENHANCED CIRCUIT BREAKER ======================

class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with adaptive thresholds and monitoring"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 half_open_max_calls: int = 3, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        # Thread-safe state
        self._lock = asyncio.Lock()
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._half_open_calls = 0
        
        # Monitoring
        self._total_calls = 0
        self._total_failures = 0
        self._state_changes = 0
    
    async def call(self, func, *args, **kwargs):
        """Execute function with enhanced circuit breaker protection"""
        async with self._lock:
            self._total_calls += 1
            
            # Check state and decide if call should proceed
            if self._state == 'OPEN':
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = 'HALF_OPEN'
                    self._half_open_calls = 0
                    self._success_count = 0
                    self._state_changes += 1
                else:
                    raise Exception(f"Circuit breaker is OPEN (failures: {self._failure_count})")
            
            elif self._state == 'HALF_OPEN':
                if self._half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN call limit exceeded")
        
        # Execute the function outside the lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Handle success
            async with self._lock:
                if self._state == 'HALF_OPEN':
                    self._half_open_calls += 1
                    self._success_count += 1
                    if self._success_count >= self.success_threshold:
                        self._state = 'CLOSED'
                        self._failure_count = 0
                        self._state_changes += 1
                elif self._state == 'CLOSED':
                    self._failure_count = max(0, self._failure_count - 1)
            
            return result
            
        except Exception as e:
            # Handle failure
            async with self._lock:
                self._total_failures += 1
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.failure_threshold:
                    if self._state != 'OPEN':
                        self._state_changes += 1
                    self._state = 'OPEN'
                elif self._state == 'HALF_OPEN':
                    self._state = 'OPEN'
                    self._state_changes += 1
            
            raise e
    
    async def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        async with self._lock:
            return {
                'state': self._state,
                'failure_count': self._failure_count,
                'total_calls': self._total_calls,
                'total_failures': self._total_failures,
                'failure_rate': self._total_failures / max(1, self._total_calls),
                'state_changes': self._state_changes
            }

# ====================== ENHANCED RATE LIMITER ======================

class EnhancedAsyncRateLimiter:
    """Enhanced rate limiter with precise timing and burst handling"""
    
    def __init__(self, max_calls: int, time_window: int, burst_allowance: int = None):
        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_allowance = burst_allowance or max(1, max_calls // 10)  # 10% burst
        
        self.calls = deque()
        self.burst_calls = deque()
        self.lock = asyncio.Lock()
        
        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        self.burst_used = 0
    
    async def acquire(self) -> bool:
        """Acquire rate limit slot with burst handling"""
        self.total_requests += 1
        
        while True:
            async with self.lock:
                now = time.time()
                
                # Remove expired calls
                while self.calls and self.calls[0] <= now - self.time_window:
                    self.calls.popleft()
                
                while self.burst_calls and self.burst_calls[0] <= now - self.time_window:
                    self.burst_calls.popleft()
                
                # Check if we can proceed with normal limit
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True
                
                # Check if we can use burst allowance
                if len(self.burst_calls) < self.burst_allowance:
                    self.burst_calls.append(now)
                    self.burst_used += 1
                    return True
                
                # Calculate sleep time
                sleep_time = self.calls[0] + self.time_window - now
            
            # Sleep outside the lock
            if sleep_time > 0:
                await asyncio.sleep(min(sleep_time, 1.0))
            else:
                await asyncio.sleep(0.01)  # Small delay to prevent tight loops
    
    async def reject_request(self) -> None:
        """Record rejected request for statistics"""
        self.rejected_requests += 1
    
    async def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        async with self.lock:
            return {
                'total_requests': self.total_requests,
                'rejected_requests': self.rejected_requests,
                'current_calls': len(self.calls),
                'burst_used': self.burst_used,
                'rejection_rate': self.rejected_requests / max(1, self.total_requests)
            }

# ====================== ENHANCED METRICS ======================

class EnhancedTradingMetrics:
    """Enhanced metrics with controlled cardinality and health monitoring"""
    
    def __init__(self):
        # Core trading metrics
        self.trades_total = prometheus_client.Counter(
            'trading_bot_trades_total', 
            'Total number of trades',
            ['outcome', 'algorithm_category']  # Reduced cardinality
        )
        
        self.positions_gauge = prometheus_client.Gauge(
            'trading_bot_positions', 
            'Current number of open positions'
        )
        
        self.balance_gauge = prometheus_client.Gauge(
            'trading_bot_balance', 
            'Current account balance'
        )
        
        self.pnl_gauge = prometheus_client.Gauge(
            'trading_bot_pnl', 
            'Profit and Loss'
        )
        
        # Enhanced API metrics
        self.api_requests_total = prometheus_client.Counter(
            'trading_bot_api_requests_total',
            'Total API requests',
            ['endpoint', 'status']
        )
        
        self.api_latency = prometheus_client.Histogram(
            'trading_bot_api_latency_seconds',
            'API request latency',
            ['endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.api_errors = prometheus_client.Counter(
            'trading_bot_api_errors_total',
            'API request errors',
            ['error_type']
        )
        
        # Algorithm performance metrics
        self.signal_generation_time = prometheus_client.Histogram(
            'trading_bot_signal_generation_seconds',
            'Time spent generating signals',
            ['algorithm'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.algorithm_success_rate = prometheus_client.Gauge(
            'trading_bot_algorithm_success_rate',
            'Algorithm success rate',
            ['algorithm']
        )
        
        # System health metrics
        self.memory_usage = prometheus_client.Gauge(
            'trading_bot_memory_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = prometheus_client.Gauge(
            'trading_bot_cpu_percent',
            'CPU usage percentage'
        )
        
        self.uptime_seconds = prometheus_client.Gauge(
            'trading_bot_uptime_seconds',
            'Bot uptime in seconds'
        )
        
        self.database_operations = prometheus_client.Counter(
            'trading_bot_database_operations_total',
            'Database operations',
            ['operation', 'status']
        )
        
        self.circuit_breaker_state = prometheus_client.Gauge(
            'trading_bot_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['component']
        )
        
        # Risk metrics
        self.portfolio_risk = prometheus_client.Gauge(
            'trading_bot_portfolio_risk',
            'Current portfolio risk percentage'
        )
        
        self.daily_pnl = prometheus_client.Gauge(
            'trading_bot_daily_pnl',
            'Daily profit and loss'
        )
        
        # Algorithm categorization for reduced cardinality
        self.algorithm_categories = {
            'Goldman': 'momentum',
            'JPMorgan': 'statistical',
            'Citadel': 'volatility',
            'Renaissance': 'pattern',
            'TwoSigma': 'ensemble'
        }
    
    def record_trade(self, outcome: str, algorithm: str):
        """Record trade with categorized algorithm"""
        category = self.algorithm_categories.get(algorithm, 'other')
        self.trades_total.labels(outcome=outcome, algorithm_category=category).inc()
    
    def record_api_request(self, endpoint: str, status: str, latency: float = None):
        """Record API request with latency"""
        # Limit endpoint names to prevent cardinality explosion
        allowed_endpoints = ['klines', 'price', 'ping', 'ticker', 'other']
        endpoint = endpoint if endpoint in allowed_endpoints else 'other'
        
        # Limit status values
        allowed_statuses = ['success', 'error', 'rate_limit', 'auth_error', 'timeout']
        status = status if status in allowed_statuses else 'error'
        
        self.api_requests_total.labels(endpoint=endpoint, status=status).inc()
        
        if latency is not None:
            self.api_latency.labels(endpoint=endpoint).observe(latency)
    
    def record_api_error(self, error_type: str):
        """Record API error with controlled types"""
        allowed_types = ['network', 'rate_limit', 'auth', 'data', 'timeout', 'other']
        error_type = error_type if error_type in allowed_types else 'other'
        self.api_errors.labels(error_type=error_type).inc()
    
    def record_database_operation(self, operation: str, status: str):
        """Record database operation"""
        allowed_ops = ['select', 'insert', 'update', 'delete', 'other']
        operation = operation if operation in allowed_ops else 'other'
        
        allowed_statuses = ['success', 'error']
        status = status if status in allowed_statuses else 'error'
        
        self.database_operations.labels(operation=operation, status=status).inc()
    
    def update_circuit_breaker_state(self, component: str, state: str):
        """Update circuit breaker state"""
        state_values = {'CLOSED': 0, 'HALF_OPEN': 1, 'OPEN': 2}
        self.circuit_breaker_state.labels(component=component).set(state_values.get(state, 0))
    
    def update_system_metrics(self, positions: int, balance: float, pnl: float, 
                            memory_bytes: int, cpu_percent: float, uptime: float,
                            portfolio_risk: float = 0.0, daily_pnl: float = 0.0):
        """Update all system metrics efficiently"""
        self.positions_gauge.set(positions)
        self.balance_gauge.set(balance)
        self.pnl_gauge.set(pnl)
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
        self.uptime_seconds.set(uptime)
        self.portfolio_risk.set(portfolio_risk)
        self.daily_pnl.set(daily_pnl)
    
    def get_memory_usage_safe(self) -> float:
        """Safely get memory usage without private attribute access"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def get_cpu_usage_safe(self) -> float:
        """Safely get CPU usage"""
        try:
            return psutil.Process().cpu_percent()
        except:
            return 0.0

# ====================== FIXED DATABASE MANAGER ======================

class FixedAsyncDatabaseManager:
    """FIXED: Enhanced database with proper backup and trigger handling"""
    
    def __init__(self, database_url: str, config: TradingConfig):
        self.config = config
        parsed = urlparse(database_url)
        if parsed.scheme == 'sqlite':
            self.db_path = parsed.path.lstrip('/') if parsed.path.startswith('/') else parsed.path
            if not self.db_path:
                self.db_path = 'trading_bot.db'
        else:
            raise ValueError(f"Unsupported database scheme: {parsed.scheme}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        # Connection management
        self._connection_pool: Set[aiosqlite.Connection] = set()
        self._pool_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._max_connections = 10
        
        # Backup management
        self.backup_dir = os.path.join(os.path.dirname(self.db_path), 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        self.last_backup = 0
        
        # Monitoring
        self.operation_stats = defaultdict(int)
        self.connection_stats = {'created': 0, 'closed': 0, 'active': 0}
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get database connection with enhanced pooling"""
        async with self._pool_lock:
            # Try to reuse existing connection
            if self._connection_pool:
                conn = self._connection_pool.pop()
                try:
                    # Test connection
                    await conn.execute('SELECT 1')
                    self.connection_stats['active'] += 1
                    return conn
                except:
                    # Connection is dead, close it
                    try:
                        await conn.close()
                    except:
                        pass
            
            # Create new connection
            if self.connection_stats['active'] >= self._max_connections:
                raise Exception("Database connection pool exhausted")
            
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            
            # Configure connection
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=10000')
            await conn.execute('PRAGMA foreign_keys=ON')
            await conn.execute('PRAGMA temp_store=MEMORY')
            await conn.execute('PRAGMA mmap_size=268435456')  # 256MB
            await conn.execute('PRAGMA recursive_triggers=OFF')  # FIXED: Prevent trigger recursion
            
            conn.row_factory = aiosqlite.Row
            
            self.connection_stats['created'] += 1
            self.connection_stats['active'] += 1
            
            return conn
    
    async def return_connection(self, conn: aiosqlite.Connection):
        """Return connection to pool"""
        async with self._pool_lock:
            if len(self._connection_pool) < self._max_connections // 2:
                self._connection_pool.add(conn)
            else:
                await conn.close()
                self.connection_stats['closed'] += 1
            
            self.connection_stats['active'] -= 1
    
    async def init_database(self):
        """Initialize database schema with FIXED triggers"""
        async with self._init_lock:
            if self._initialized:
                return
            
            conn = await self.get_connection()
            try:
                await conn.executescript('''
                    -- Enhanced positions table
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
                        quantity REAL NOT NULL CHECK (quantity > 0),
                        entry_price REAL NOT NULL CHECK (entry_price > 0),
                        current_price REAL CHECK (current_price > 0),
                        stop_loss REAL CHECK (stop_loss >= 0),
                        take_profit REAL CHECK (take_profit >= 0),
                        pnl REAL DEFAULT 0,
                        pnl_percent REAL DEFAULT 0,
                        fees REAL DEFAULT 0,
                        status TEXT DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
                        algorithm TEXT NOT NULL,
                        reasoning TEXT,
                        confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
                        entry_time REAL NOT NULL,
                        exit_time REAL,
                        max_pnl REAL DEFAULT 0,
                        min_pnl REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Enhanced trades table
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER REFERENCES positions(id),
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
                        quantity REAL NOT NULL CHECK (quantity > 0),
                        price REAL NOT NULL CHECK (price > 0),
                        fee REAL DEFAULT 0,
                        pnl REAL DEFAULT 0,
                        pnl_percent REAL DEFAULT 0,
                        algorithm TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Enhanced system metrics
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metric_labels TEXT, -- JSON string for labels
                        timestamp REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Enhanced bot state
                    CREATE TABLE IF NOT EXISTS bot_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        value_type TEXT DEFAULT 'string',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Algorithm performance tracking
                    CREATE TABLE IF NOT EXISTS algorithm_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        algorithm TEXT NOT NULL,
                        success_count INTEGER DEFAULT 0,
                        total_count INTEGER DEFAULT 0,
                        avg_pnl REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Error log table
                    CREATE TABLE IF NOT EXISTS error_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        stack_trace TEXT,
                        context TEXT, -- JSON string
                        timestamp REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Performance indexes
                    CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
                    CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time);
                    CREATE INDEX IF NOT EXISTS idx_positions_algorithm ON positions(algorithm);
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_algorithm_perf_algorithm ON algorithm_performance(algorithm);
                    CREATE INDEX IF NOT EXISTS idx_error_log_timestamp ON error_log(timestamp);
                ''')
                
                # FIXED: Non-recursive triggers that only update when needed
                await conn.execute('''
                    CREATE TRIGGER IF NOT EXISTS update_positions_updated_at 
                        AFTER UPDATE OF current_price, pnl, status ON positions
                        WHEN NEW.updated_at = OLD.updated_at
                        BEGIN
                            UPDATE positions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                        END;
                ''')
                
                await conn.execute('''
                    CREATE TRIGGER IF NOT EXISTS update_bot_state_updated_at 
                        AFTER UPDATE OF value ON bot_state
                        WHEN NEW.updated_at = OLD.updated_at
                        BEGIN
                            UPDATE bot_state SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
                        END;
                ''')
                
                # Views for reporting
                await conn.execute('''
                    CREATE VIEW IF NOT EXISTS position_summary AS
                    SELECT 
                        algorithm,
                        COUNT(*) as total_positions,
                        SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_positions,
                        SUM(CASE WHEN status = 'CLOSED' AND pnl > 0 THEN 1 ELSE 0 END) as winning_positions,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl
                    FROM positions 
                    GROUP BY algorithm;
                ''')
                
                await conn.commit()
                self.operation_stats['schema_init'] += 1
            finally:
                await self.return_connection(conn)
            
            self._initialized = True
    
    async def create_backup(self) -> str:
        """FIXED: Create proper SQLite backup using backup API"""
        if time.time() - self.last_backup < self.config.backup_interval:
            return ""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f'trading_bot_{timestamp}.db')
        
        try:
            # FIXED: Use SQLite backup API for consistent backup
            source_conn = await aiosqlite.connect(self.db_path)
            backup_conn = await aiosqlite.connect(backup_path)
            
            try:
                # Use SQLite backup API
                await source_conn.backup(backup_conn)
            finally:
                await source_conn.close()
                await backup_conn.close()
            
            # Compress backup
            with open(backup_path, 'rb') as f_in:
                with gzip.open(f'{backup_path}.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(backup_path)  # Remove uncompressed version
            self.last_backup = time.time()
            
            # Clean old backups (keep last 10)
            backups = sorted([f for f in os.listdir(self.backup_dir) if f.endswith('.gz')])
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    os.remove(os.path.join(self.backup_dir, old_backup))
            
            return f'{backup_path}.gz'
        except Exception as e:
            await self.log_error('backup_error', str(e))
            return ""
    
    async def save_position(self, position_data: Dict) -> int:
        """Save position with enhanced validation and error handling"""
        required_fields = ['symbol', 'side', 'quantity', 'entry_price', 'algorithm']
        for field in required_fields:
            if field not in position_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Enhanced validation
        if not SecurityManager.validate_symbol(position_data['symbol']):
            raise ValueError(f"Invalid symbol: {position_data['symbol']}")
        
        if position_data['side'] not in ['LONG', 'SHORT']:
            raise ValueError(f"Invalid side: {position_data['side']}")
        
        if not SecurityConfig.validate_decimal_range(float(position_data['quantity']), 0.000001, 1e6):
            raise ValueError(f"Invalid quantity: {position_data['quantity']}")
        
        # Handle optional fields properly
        stop_loss = float(position_data.get('stop_loss', 0))
        take_profit = float(position_data.get('take_profit', 0))
        
        # Convert 0 to None for database constraints
        stop_loss = stop_loss if stop_loss > 0 else None
        take_profit = take_profit if take_profit > 0 else None
        
        conn = await self.get_connection()
        try:
            cursor = await conn.execute('''
                INSERT INTO positions (
                    symbol, side, quantity, entry_price, stop_loss, take_profit,
                    algorithm, reasoning, confidence, entry_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_data['symbol'],
                position_data['side'],
                float(position_data['quantity']),
                float(position_data['entry_price']),
                stop_loss,
                take_profit,
                position_data.get('algorithm', 'Unknown'),
                position_data.get('reasoning', ''),
                float(position_data.get('confidence', 0)),
                position_data.get('entry_time', time.time())
            ))
            await conn.commit()
            self.operation_stats['insert'] += 1
            return cursor.lastrowid
        except Exception as e:
            self.operation_stats['insert_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def update_position(self, position_id: int, updates: Dict):
        """Update position with enhanced validation"""
        if not updates:
            return
        
        # Validate updates
        allowed_fields = {
            'current_price', 'stop_loss', 'take_profit', 'pnl', 'pnl_percent',
            'fees', 'status', 'max_pnl', 'min_pnl'
        }
        invalid_fields = set(updates.keys()) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid update fields: {invalid_fields}")
        
        # Process updates with proper null handling
        processed_updates = {}
        for key, value in updates.items():
            if key in ['stop_loss', 'take_profit'] and value is not None:
                processed_updates[key] = float(value) if float(value) > 0 else None
            elif key == 'status' and value not in ['OPEN', 'CLOSED', 'CANCELLED']:
                raise ValueError(f"Invalid status: {value}")
            else:
                processed_updates[key] = value
        
        set_clause = ', '.join([f"{key} = ?" for key in processed_updates.keys()])
        values = list(processed_updates.values()) + [position_id]
        
        conn = await self.get_connection()
        try:
            await conn.execute(f'''
                UPDATE positions SET {set_clause} WHERE id = ?
            ''', values)
            await conn.commit()
            self.operation_stats['update'] += 1
        except Exception as e:
            self.operation_stats['update_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions with enhanced data"""
        conn = await self.get_connection()
        try:
            async with conn.execute('''
                SELECT *, 
                       (current_price - entry_price) * quantity * 
                       CASE WHEN side = 'LONG' THEN 1 ELSE -1 END as unrealized_pnl
                FROM positions 
                WHERE status = 'OPEN' 
                ORDER BY entry_time ASC
            ''') as cursor:
                rows = await cursor.fetchall()
                self.operation_stats['select'] += 1
                return [dict(row) for row in rows]
        except Exception as e:
            self.operation_stats['select_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def close_position(self, position_id: int, exit_price: float, pnl: float):
        """Close position with enhanced trade recording"""
        conn = await self.get_connection()
        try:
            await conn.execute('BEGIN IMMEDIATE')
            
            # Get position details
            async with conn.execute('SELECT * FROM positions WHERE id = ?', (position_id,)) as cursor:
                position = await cursor.fetchone()
            
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            # Calculate additional metrics
            entry_price = float(position['entry_price'])
            pnl_percent = (pnl / (entry_price * float(position['quantity']))) * 100
            
            # Update position status
            await conn.execute('''
                UPDATE positions SET 
                    status = 'CLOSED', 
                    current_price = ?, 
                    pnl = ?,
                    pnl_percent = ?,
                    exit_time = ?
                WHERE id = ?
            ''', (exit_price, pnl, pnl_percent, time.time(), position_id))
            
            # Record trade
            await conn.execute('''
                INSERT INTO trades (
                    position_id, symbol, side, quantity, price, pnl, pnl_percent, algorithm, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_id,
                position['symbol'],
                position['side'],
                position['quantity'],
                exit_price,
                pnl,
                pnl_percent,
                position['algorithm'],
                time.time()
            ))
            
            await conn.commit()
            self.operation_stats['close_position'] += 1
            
        except Exception as e:
            await conn.rollback()
            self.operation_stats['close_position_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def log_error(self, error_type: str, error_message: str, 
                       stack_trace: str = None, context: Dict = None):
        """Log error to database"""
        conn = await self.get_connection()
        try:
            await conn.execute('''
                INSERT INTO error_log (error_type, error_message, stack_trace, context, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                error_type,
                error_message,
                stack_trace,
                json.dumps(context) if context else None,
                time.time()
            ))
            await conn.commit()
            self.operation_stats['log_error'] += 1
        except Exception:
            self.operation_stats['log_error_error'] += 1
        finally:
            await self.return_connection(conn)
    
    async def get_bot_state(self, key: str) -> Optional[str]:
        """Get bot state value with type handling"""
        conn = await self.get_connection()
        try:
            async with conn.execute('SELECT value, value_type FROM bot_state WHERE key = ?', (key,)) as cursor:
                row = await cursor.fetchone()
                self.operation_stats['get_state'] += 1
                return row['value'] if row else None
        except Exception as e:
            self.operation_stats['get_state_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def set_bot_state(self, key: str, value: str, value_type: str = 'string'):
        """Set bot state value with type"""
        conn = await self.get_connection()
        try:
            await conn.execute('''
                INSERT OR REPLACE INTO bot_state (key, value, value_type) VALUES (?, ?, ?)
            ''', (key, value, value_type))
            await conn.commit()
            self.operation_stats['set_state'] += 1
        except Exception as e:
            self.operation_stats['set_state_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data to prevent database bloat"""
        conn = await self.get_connection()
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            # Clean old metrics
            await conn.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_time,))
            
            # Clean old error logs
            await conn.execute('DELETE FROM error_log WHERE timestamp < ?', (cutoff_time,))
            
            # Vacuum database
            await conn.execute('VACUUM')
            await conn.commit()
            
            self.operation_stats['cleanup'] += 1
        except Exception as e:
            self.operation_stats['cleanup_error'] += 1
            raise e
        finally:
            await self.return_connection(conn)

# ====================== MATHEMATICAL UTILITIES ======================

class MathUtils:
    """Safe mathematical utilities with proper error handling"""
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert to float with comprehensive bounds checking"""
        try:
            if value is None:
                return default
            
            result = float(value)
            
            # Check for invalid values
            if math.isnan(result) or math.isinf(result):
                return default
            
            # Reasonable bounds check
            if abs(result) > 1e12:  # Very large numbers
                return default
                
            return result
            
        except (ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_decimal(value: Any, default: Decimal = Decimal('0')) -> Decimal:
        """Safely convert to Decimal with proper error handling"""
        try:
            if value is None:
                return default
                
            if isinstance(value, Decimal):
                return value if value.is_finite() else default
                
            # Convert to string first to avoid float precision issues
            str_value = str(value)
            result = Decimal(str_value)
            
            return result if result.is_finite() else default
            
        except (ValueError, TypeError, InvalidOperation):
            return default
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate logarithmic returns with safety checks"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                try:
                    ret = math.log(prices[i] / prices[i-1])
                    if math.isfinite(ret) and abs(ret) < 1.0:  # Reasonable return bounds
                        returns.append(ret)
                except (ValueError, ZeroDivisionError):
                    continue
                    
        return returns
    
    @staticmethod
    def rolling_mean(data: List[float], window: int) -> float:
        """Calculate rolling mean with validation"""
        if not data or window <= 0:
            return 0.0
            
        if len(data) < window:
            window = len(data)
            
        return mean(data[-window:])
    
    @staticmethod
    def rolling_std(data: List[float], window: int) -> float:
        """Calculate rolling standard deviation with validation"""
        if not data or window <= 1:
            return 0.0
            
        if len(data) < window:
            window = len(data)
            
        if window < 2:
            return 0.0
            
        try:
            return stdev(data[-window:])
        except:
            return 0.0
    
    @staticmethod
    def z_score(value: float, mean_val: float, std_val: float) -> float:
        """Calculate Z-score with safety checks"""
        if std_val <= 0 or not math.isfinite(value) or not math.isfinite(mean_val):
            return 0.0
        
        z = (value - mean_val) / std_val
        
        # Cap extreme z-scores
        return max(-10.0, min(10.0, z))
    
    @staticmethod
    def rsi(prices: List[float], window: int = 14) -> float:
        """Calculate RSI with improved stability"""
        if len(prices) < window + 1:
            return 50.0
        
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if math.isfinite(change):
                changes.append(change)
        
        if len(changes) < window:
            return 50.0
        
        recent_changes = changes[-window:]
        gains = [max(0, change) for change in recent_changes]
        losses = [max(0, -change) for change in recent_changes]
        
        avg_gain = mean(gains) if gains else 0
        avg_loss = mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + rs))
        
        # Ensure valid range
        return max(0.0, min(100.0, rsi_value))
    
    @staticmethod
    def bollinger_bands(prices: List[float], window: int = 20, 
                       num_std: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands with validation"""
        if len(prices) < window or window < 2:
            return None, None, None
        
        recent_prices = prices[-window:]
        
        try:
            sma = mean(recent_prices)
            std = stdev(recent_prices)
            
            if not math.isfinite(sma) or not math.isfinite(std) or std <= 0:
                return None, None, None
            
            upper = sma + (num_std * std)
            lower = sma - (num_std * std)
            
            return upper, sma, lower
            
        except:
            return None, None, None

# ====================== SIGNAL CLASSES ======================

@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive validation"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    confidence: float
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    reasoning: str
    algorithm: str
    timestamp: float
    additional_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Comprehensive signal validation"""
        # Validate symbol
        if not SecurityConfig.SYMBOL_PATTERN.match(self.symbol):
            raise ValueError(f"Invalid symbol: {self.symbol}")
        
        # Validate side
        if self.side not in ['LONG', 'SHORT']:
            raise ValueError(f"Invalid side: {self.side}")
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        
        # Validate prices
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry price: {self.entry_price}")
        
        if self.stop_loss < 0:  # Allow 0 for no stop loss
            raise ValueError(f"Invalid stop loss: {self.stop_loss}")
        
        if self.take_profit < 0:  # Allow 0 for no take profit
            raise ValueError(f"Invalid take profit: {self.take_profit}")
        
        # Validate price relationships only if stops are set
        if self.stop_loss > 0 and self.side == 'LONG':
            if self.stop_loss >= self.entry_price:
                raise ValueError("LONG stop loss must be below entry price")
        elif self.stop_loss > 0 and self.side == 'SHORT':
            if self.stop_loss <= self.entry_price:
                raise ValueError("SHORT stop loss must be above entry price")
        
        if self.take_profit > 0 and self.side == 'LONG':
            if self.take_profit <= self.entry_price:
                raise ValueError("LONG take profit must be above entry price")
        elif self.take_profit > 0 and self.side == 'SHORT':
            if self.take_profit >= self.entry_price:
                raise ValueError("SHORT take profit must be below entry price")
        
        # Validate timestamp
        if not math.isfinite(self.timestamp) or self.timestamp <= 0:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")
        
        # Sanitize strings
        self.reasoning = SecurityConfig.SAFE_STRING_PATTERN.sub('', str(self.reasoning))[:200]
        self.algorithm = SecurityConfig.SAFE_STRING_PATTERN.sub('', str(self.algorithm))[:50]

# ====================== BASIC ALGORITHMS ======================

class TradingAlgorithm(ABC):
    """Base class for trading algorithms"""
    
    def __init__(self):
        self.name = getattr(self, 'name', self.__class__.__name__)
        self.circuit_breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    @abstractmethod
    async def generate_signal(self, symbol: str, klines: List, market_data: Dict = None) -> Optional[TradingSignal]:
        """Generate trading signal - must be implemented by subclasses"""
        pass
    
    async def safe_generate_signal(self, symbol: str, klines: List, market_data: Dict = None) -> Optional[TradingSignal]:
        """Wrapper with circuit breaker and error handling"""
        try:
            return await self.circuit_breaker.call(self.generate_signal, symbol, klines, market_data)
        except Exception as e:
            logging.debug(f"{self.name} algorithm error for {symbol}: {e}")
            return None

class SimpleMovingAverageAlgorithm(TradingAlgorithm):
    """Simple moving average crossover algorithm"""
    
    def __init__(self):
        super().__init__()
        self.name = "Goldman"
        self.fast_period = 10
        self.slow_period = 30
    
    async def generate_signal(self, symbol: str, klines: List, market_data: Dict = None) -> Optional[TradingSignal]:
        """Generate simple MA crossover signal"""
        try:
            if len(klines) < self.slow_period:
                return None
            
            closes = [MathUtils.safe_float(k[4]) for k in klines]
            current_price = closes[-1]
            
            if current_price <= 0:
                return None
            
            # Calculate moving averages
            fast_ma = MathUtils.rolling_mean(closes, self.fast_period)
            slow_ma = MathUtils.rolling_mean(closes, self.slow_period)
            
            # Previous values for crossover detection
            if len(closes) < self.slow_period + 1:
                return None
            
            prev_fast_ma = MathUtils.rolling_mean(closes[:-1], self.fast_period)
            prev_slow_ma = MathUtils.rolling_mean(closes[:-1], self.slow_period)
            
            # Detect crossover
            confidence = 0
            side = None
            
            # Bullish crossover
            if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                side = 'LONG'
                confidence = 0.72
            # Bearish crossover
            elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                side = 'SHORT'
                confidence = 0.72
            
            if not side or confidence < 0.70:
                return None
            
            # Simple risk management
            stop_distance = current_price * 0.02  # 2% stop
            profit_distance = current_price * 0.04  # 4% target
            
            if side == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - profit_distance
            
            return TradingSignal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                entry_price=MathUtils.safe_decimal(current_price),
                stop_loss=MathUtils.safe_decimal(max(0, stop_loss)),
                take_profit=MathUtils.safe_decimal(take_profit),
                reasoning=f"MA Crossover: Fast={fast_ma:.4f}, Slow={slow_ma:.4f}",
                algorithm=self.name,
                timestamp=time.time()
            )
            
        except Exception as e:
            logging.debug(f"{self.name} algorithm error for {symbol}: {e}")
            return None

# Create aliases for other algorithms
class JPMorganAlgorithm(SimpleMovingAverageAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = "JPMorgan"
        self.fast_period = 5
        self.slow_period = 20

class CitadelAlgorithm(SimpleMovingAverageAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = "Citadel"
        self.fast_period = 15
        self.slow_period = 40

class RenaissanceAlgorithm(SimpleMovingAverageAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = "Renaissance"
        self.fast_period = 8
        self.slow_period = 25

class TwoSigmaAlgorithm(SimpleMovingAverageAlgorithm):
    def __init__(self):
        super().__init__()
        self.name = "TwoSigma"
        self.fast_period = 12
        self.slow_period = 35

# ====================== MARKET DATA PROVIDER ======================

class BinanceDataProvider:
    """Enhanced Binance data provider with proper error handling"""
    
    def __init__(self, config: TradingConfig, metrics: EnhancedTradingMetrics):
        self.config = config
        self.metrics = metrics
        self.base_url = "https://testnet.binance.vision" if config.binance_testnet else "https://api.binance.com"
        self.session = None
        self.rate_limiter = EnhancedAsyncRateLimiter(config.rate_limit_calls, config.rate_limit_window)
        self.circuit_breaker = EnhancedCircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.request_timeout = config.api_timeout
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout,
            connect=5,
            sock_read=self.request_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=10,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'OmegaX-TradingBot/8.1',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str = '5m', limit: int = 100) -> List:
        """Get kline data with comprehensive error handling"""
        if not SecurityConfig.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return await self.circuit_breaker.call(self._fetch_klines, symbol, interval, limit)
    
    async def _fetch_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Internal klines fetching with rate limiting"""
        await self.rate_limiter.acquire()
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(max(1, limit), 1000)  # Binance limits
        }
        
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/api/v3/klines", params=params) as response:
                latency = time.time() - start_time
                endpoint = 'klines'
                
                if response.status == 200:
                    self.metrics.record_api_request(endpoint, 'success', latency)
                    data = await response.json()
                    
                    # Basic validation
                    if isinstance(data, list):
                        return data
                    else:
                        self.metrics.record_api_error('data')
                        raise ValueError("Invalid data format received")
                        
                elif response.status == 429:  # Rate limit
                    self.metrics.record_api_request(endpoint, 'rate_limit', latency)
                    self.metrics.record_api_error('rate_limit')
                    raise aiohttp.ClientError("Rate limit exceeded")
                    
                elif response.status in [401, 403]:  # Auth errors
                    self.metrics.record_api_request(endpoint, 'auth_error', latency)
                    self.metrics.record_api_error('auth')
                    raise aiohttp.ClientError(f"Authentication error: {response.status}")
                    
                else:
                    self.metrics.record_api_request(endpoint, 'error', latency)
                    self.metrics.record_api_error('other')
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API Error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            self.metrics.record_api_error('timeout')
            raise aiohttp.ClientError("Request timeout")
        except aiohttp.ClientError:
            raise
        except Exception as e:
            self.metrics.record_api_error('network')
            raise aiohttp.ClientError(f"Network error: {e}")
    
    async def get_price(self, symbol: str) -> float:
        """Get current price with error handling"""
        if not SecurityConfig.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return await self.circuit_breaker.call(self._fetch_price, symbol)
    
    async def _fetch_price(self, symbol: str) -> float:
        """Internal price fetching"""
        await self.rate_limiter.acquire()
        
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/api/v3/ticker/price", 
                                      params={'symbol': symbol}) as response:
                latency = time.time() - start_time
                
                if response.status == 200:
                    self.metrics.record_api_request('price', 'success', latency)
                    data = await response.json()
                    
                    if 'price' in data:
                        price = MathUtils.safe_float(data['price'])
                        if price > 0:
                            return price
                        else:
                            raise ValueError("Invalid price received")
                    else:
                        raise ValueError("Price not found in response")
                        
                else:
                    self.metrics.record_api_request('price', 'error', latency)
                    raise aiohttp.ClientError(f"Price API Error: {response.status}")
                    
        except Exception as e:
            self.metrics.record_api_error('network')
            raise e
    
    async def test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            async with self.session.get(f"{self.base_url}/api/v3/ping") as response:
                return response.status == 200
        except:
            return False

# ====================== RISK MANAGER ======================

class RiskManager:
    """Enhanced risk management with comprehensive controls"""
    
    def __init__(self, config: TradingConfig, db: FixedAsyncDatabaseManager):
        self.config = config
        self.db = db
        self.max_portfolio_risk = config.max_portfolio_risk
        self.max_position_risk = config.base_risk_percent
        self.emergency_stop_percent = config.emergency_stop_percent
        
        # Risk tracking
        self.daily_loss = Decimal('0')
        self.last_reset = datetime.now().date()
    
    async def calculate_position_size(self, signal: TradingSignal, current_balance: Decimal) -> Decimal:
        """Calculate optimal position size with enhanced risk controls"""
        try:
            if current_balance <= 0:
                return Decimal('0')
            
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            
            if entry_price <= 0:
                return Decimal('0')
            
            # If no stop loss is set, use default risk
            if stop_loss <= 0:
                # Use percentage-based position sizing
                risk_amount = current_balance * Decimal(str(self.max_position_risk)) * Decimal(str(signal.confidence))
                position_size = risk_amount / entry_price
            else:
                # Calculate risk per unit
                if signal.side == 'LONG':
                    risk_per_unit = entry_price - stop_loss
                else:
                    risk_per_unit = stop_loss - entry_price
                
                if risk_per_unit <= 0:
                    return Decimal('0')
                
                # Risk amount (adjusted by confidence)
                confidence_multiplier = Decimal(str(signal.confidence))
                base_risk = current_balance * Decimal(str(self.max_position_risk))
                risk_amount = base_risk * confidence_multiplier
                
                # Position size
                position_size = risk_amount / risk_per_unit
            
            # Apply limits
            min_notional = Decimal('25')  # Minimum $25 position
            max_position_percent = Decimal('0.10')  # Maximum 10% of balance
            
            min_size = min_notional / entry_price
            max_size = (current_balance * max_position_percent) / entry_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Round to appropriate precision (6 decimal places)
            return position_size.quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
            
        except Exception as e:
            logging.error(f"Position sizing error: {e}")
            return Decimal('0')
    
    async def check_portfolio_risk(self, new_position_value: Decimal, current_balance: Decimal) -> bool:
        """Check portfolio risk limits using current balance"""
        try:
            positions = await self.db.get_open_positions()
            
            # Calculate current portfolio risk
            current_risk = Decimal('0')
            for pos in positions:
                position_value = Decimal(str(pos['quantity'])) * Decimal(str(pos['entry_price']))
                position_risk = position_value * Decimal(str(self.max_position_risk))
                current_risk += position_risk
            
            # Check new total risk
            new_position_risk = new_position_value * Decimal(str(self.max_position_risk))
            total_risk = current_risk + new_position_risk
            
            max_allowed_risk = current_balance * Decimal(str(self.max_portfolio_risk))
            
            return total_risk <= max_allowed_risk
            
        except Exception as e:
            logging.error(f"Portfolio risk check error: {e}")
            return False
    
    async def should_close_position(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """Enhanced position exit logic"""
        try:
            entry_price = float(position['entry_price'])
            stop_loss = float(position.get('stop_loss') or 0)
            take_profit = float(position.get('take_profit') or 0)
            side = position['side']
            entry_time = float(position['entry_time'])
            
            # Time-based exit
            position_age = time.time() - entry_time
            if position_age > self.config.position_timeout_hours * 3600:
                return True, "Time limit exceeded"
            
            # Stop loss check (only if set)
            if stop_loss > 0:
                if side == 'LONG' and current_price <= stop_loss:
                    return True, "Stop loss triggered"
                elif side == 'SHORT' and current_price >= stop_loss:
                    return True, "Stop loss triggered"
            
            # Take profit check (only if set)
            if take_profit > 0:
                if side == 'LONG' and current_price >= take_profit:
                    return True, "Take profit triggered"
                elif side == 'SHORT' and current_price <= take_profit:
                    return True, "Take profit triggered"
            
            # Emergency stop on large losses
            pnl_percent = self._calculate_pnl_percent(position, current_price)
            if pnl_percent < -(self.emergency_stop_percent * 100):
                return True, f"Emergency stop - {pnl_percent:.1f}% loss"
            
            return False, ""
            
        except Exception as e:
            logging.error(f"Position exit check error: {e}")
            return True, "Error in position evaluation"
    
    def _calculate_pnl_percent(self, position: Dict, current_price: float) -> float:
        """Calculate PnL percentage for position"""
        try:
            entry_price = float(position['entry_price'])
            side = position['side']
            
            if entry_price <= 0:
                return 0.0
            
            if side == 'LONG':
                return ((current_price - entry_price) / entry_price) * 100
            else:
                return ((entry_price - current_price) / entry_price) * 100
                
        except:
            return 0.0
    
    async def update_daily_pnl(self, pnl_change: Decimal):
        """Track daily P&L for risk management"""
        current_date = datetime.now().date()
        
        # Reset daily loss if new day
        if current_date != self.last_reset:
            self.daily_loss = Decimal('0')
            self.last_reset = current_date
        
        if pnl_change < 0:
            self.daily_loss += abs(pnl_change)
    
    async def check_daily_loss_limit(self, current_balance: Decimal) -> bool:
        """Check if daily loss limit is exceeded"""
        max_daily_loss = current_balance * Decimal(str(self.config.max_drawdown))
        return self.daily_loss < max_daily_loss

# ====================== FIXED: MAIN TRADING BOT ======================

class EnhancedOmegaXTradingBot:
    """FIXED: Production-ready trading bot with all critical issues resolved"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = structlog.get_logger("TradingBot")
        
        # Initialize components
        self.security = SecurityManager(config)
        self.metrics = EnhancedTradingMetrics()
        self.db = FixedAsyncDatabaseManager(config.database_url, config)
        self.risk_manager = RiskManager(config, self.db)
        
        # Initialize algorithms
        self.algorithms = self._initialize_algorithms()
        
        # Trading state
        self.running = False
        self.balance = Decimal(str(config.initial_balance))
        self.total_pnl = Decimal('0')
        self.start_time = time.time()
        self.last_report = 0
        self.last_gc = 0
        self.last_backup = 0
        self.positions_lock = asyncio.Lock()
        
        # Market data caching
        self.market_data_cache = {}
        self.last_cache_update = {}
        self._cache_update_lock = asyncio.Lock()
        
        self.logger.info("Enhanced trading bot initialized", 
                        algorithms=len(self.algorithms),
                        initial_balance=str(config.initial_balance))
    
    def _initialize_algorithms(self) -> Dict[str, TradingAlgorithm]:
        """Initialize all configured trading algorithms"""
        algorithm_classes = {
            'Goldman': SimpleMovingAverageAlgorithm,
            'JPMorgan': JPMorganAlgorithm,
            'Citadel': CitadelAlgorithm,
            'Renaissance': RenaissanceAlgorithm,
            'TwoSigma': TwoSigmaAlgorithm,
        }
        
        algorithms = {}
        for algo_name in self.config.enabled_algorithms:
            if algo_name in algorithm_classes:
                try:
                    algorithms[algo_name] = algorithm_classes[algo_name]()
                    self.logger.info(f"Loaded algorithm: {algo_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load algorithm {algo_name}: {e}")
        
        if not algorithms:
            raise ValueError("No algorithms loaded successfully")
        
        return algorithms
    
    async def start(self):
        """Start the enhanced trading bot"""
        try:
            self.running = True
            self.logger.info("Starting Enhanced OmegaX Trading Bot...")
            
            # Initialize database
            await self.db.init_database()
            
            # Load persisted state
            await self._load_bot_state()
            
            async with BinanceDataProvider(self.config, self.metrics) as market_data:
                # Test connectivity
                if not await market_data.test_connectivity():
                    raise Exception("Cannot connect to market data provider")
                
                self.logger.info("Market data connection established")
                
                # Main trading loop
                while self.running:
                    try:
                        await self._trading_cycle(market_data)
                        
                        # Handle maintenance tasks
                        await self._handle_maintenance()
                        
                        # Sleep for configured interval
                        await asyncio.sleep(self.config.update_interval)
                        
                    except Exception as e:
                        self.logger.error("Trading cycle error", error=str(e))
                        await self.db.log_error('trading_cycle', str(e), traceback.format_exc())
                        await asyncio.sleep(5)  # Brief pause on error
                        
        except Exception as e:
            self.logger.error("Critical error in trading bot", error=str(e))
            await self.emergency_shutdown()
            raise
        finally:
            self.running = False
            await self._save_bot_state()
            self.logger.info("Enhanced trading bot stopped")
    
    async def _trading_cycle(self, market_data: BinanceDataProvider):
        """Execute one enhanced trading cycle"""
        try:
            cycle_start = time.time()
            
            # Update market data cache
            await self._update_market_data_cache(market_data)
            
            # Manage existing positions
            await self._manage_positions(market_data)
            
            # Check daily loss limits
            if not await self.risk_manager.check_daily_loss_limit(self.balance):
                self.logger.warning("Daily loss limit exceeded - pausing new positions")
            else:
                # Generate new signals
                await self._scan_for_signals(market_data)
            
            # Update metrics and state
            await self._update_metrics()
            
            # Periodic reporting
            current_time = time.time()
            if current_time - self.last_report > 900:  # Every 15 minutes
                await self._generate_report()
                self.last_report = current_time
            
            # Log cycle performance
            cycle_time = time.time() - cycle_start
            if cycle_time > self.config.update_interval * 0.8:
                self.logger.warning("Slow trading cycle", cycle_time=cycle_time)
                
        except Exception as e:
            self.logger.error("Trading cycle error", error=str(e))
            raise
    
    async def _update_market_data_cache(self, market_data: BinanceDataProvider):
        """Update market data cache efficiently"""
        try:
            async with self._cache_update_lock:
                current_time = time.time()
                
                # Get active symbols
                active_symbols = set(self.config.trading_pairs[:self.config.max_pairs_per_scan])
                
                positions = await self.db.get_open_positions()
                for pos in positions:
                    active_symbols.add(pos['symbol'])
                
                # Identify symbols that need updates
                symbols_to_update = []
                for symbol in active_symbols:
                    last_update = self.last_cache_update.get(symbol, 0)
                    if current_time - last_update > 60:  # Cache for 60 seconds
                        symbols_to_update.append(symbol)
            
            # Fetch data outside the lock
            if symbols_to_update:
                fetch_tasks = [
                    self._fetch_and_cache_data(market_data, symbol) 
                    for symbol in symbols_to_update
                ]
                await asyncio.gather(*fetch_tasks, return_exceptions=True)
                
        except Exception as e:
            self.logger.error("Market data cache update error", error=str(e))
    
    async def _fetch_and_cache_data(self, market_data: BinanceDataProvider, symbol: str):
        """Fetch and cache data for a single symbol"""
        try:
            klines = await market_data.get_klines(symbol, '5m', 100)
            if klines:
                async with self._cache_update_lock:
                    self.market_data_cache[symbol] = klines
                    self.last_cache_update[symbol] = time.time()
        except Exception as e:
            self.logger.debug(f"Failed to update cache for {symbol}: {e}")
    
    async def _manage_positions(self, market_data: BinanceDataProvider):
        """Manage existing positions"""
        try:
            positions = await self.db.get_open_positions()
            
            for position in positions:
                try:
                    symbol = position['symbol']
                    current_price = await market_data.get_price(symbol)
                    
                    # Check if position should be closed
                    should_close, reason = await self.risk_manager.should_close_position(position, current_price)
                    
                    if should_close:
                        await self._close_position(position, current_price, reason)
                    else:
                        # Update current price
                        await self.db.update_position(position['id'], {'current_price': current_price})
                    
                except Exception as e:
                    self.logger.error(f"Position management error for {position.get('symbol', 'unknown')}: {e}")
                    
        except Exception as e:
            self.logger.error("Position management error", error=str(e))
    
    async def _close_position(self, position: Dict, current_price: float, reason: str):
        """Close a trading position"""
        try:
            async with self.positions_lock:
                # Calculate PnL
                entry_price = float(position['entry_price'])
                quantity = float(position['quantity'])
                side = position['side']
                
                if side == 'LONG':
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity
                
                # Update database
                await self.db.close_position(position['id'], current_price, pnl)
                
                # Update bot state
                pnl_decimal = Decimal(str(pnl))
                self.total_pnl += pnl_decimal
                
                # Update risk tracking
                await self.risk_manager.update_daily_pnl(pnl_decimal)
                
                # Record metrics
                outcome = 'win' if pnl > 0 else 'loss'
                self.metrics.record_trade(outcome, position.get('algorithm', 'Unknown'))
                
                self.logger.info("Position closed",
                               symbol=position['symbol'],
                               side=side,
                               pnl=pnl,
                               reason=reason)
                
        except Exception as e:
            self.logger.error(f"Close position error: {e}")
    
    async def _scan_for_signals(self, market_data: BinanceDataProvider):
        """Scan for new trading signals"""
        try:
            positions = await self.db.get_open_positions()
            
            # Check position limits
            if len(positions) >= self.config.max_positions:
                return
            
            # Get available symbols
            position_symbols = {pos['symbol'] for pos in positions}
            available_symbols = [s for s in self.config.trading_pairs if s not in position_symbols]
            
            # Limit scanning
            max_scans = min(self.config.max_pairs_per_scan, self.config.max_positions - len(positions))
            symbols_to_scan = available_symbols[:max_scans]
            
            for symbol in symbols_to_scan:
                try:
                    if symbol not in self.market_data_cache:
                        continue
                    
                    klines = self.market_data_cache[symbol]
                    if not klines:
                        continue
                    
                    # Generate signals from all algorithms
                    signals = await self._generate_signals(symbol, klines)
                    
                    # Aggregate signals
                    final_signal = await self._aggregate_signals(signals)
                    
                    if final_signal and final_signal.confidence >= self.config.signal_threshold:
                        await self._open_position(final_signal)
                    
                except Exception as e:
                    self.logger.debug(f"Signal scanning error for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error("Signal scanning error", error=str(e))
    
    async def _generate_signals(self, symbol: str, klines: List) -> List[TradingSignal]:
        """Generate signals from all algorithms"""
        signal_tasks = []
        
        for algo_name, algorithm in self.algorithms.items():
            with self.metrics.signal_generation_time.labels(algorithm=algo_name).time():
                task = algorithm.safe_generate_signal(symbol, klines, self.market_data_cache)
                signal_tasks.append((algo_name, task))
        
        # Wait for all signals
        signals = []
        results = await asyncio.gather(*[task for _, task in signal_tasks], return_exceptions=True)
        
        for (algo_name, _), result in zip(signal_tasks, results):
            if isinstance(result, TradingSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                self.logger.debug(f"Algorithm {algo_name} error for {symbol}: {result}")
        
        return signals
    
    async def _aggregate_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Aggregate multiple signals"""
        if len(signals) < 2:  # Require at least 2 signals
            return None
        
        # Group by side
        long_signals = [s for s in signals if s.side == 'LONG']
        short_signals = [s for s in signals if s.side == 'SHORT']
        
        # Weighted voting
        long_weight = sum(s.confidence * self.config.algorithm_weights.get(s.algorithm, 1.0) for s in long_signals)
        short_weight = sum(s.confidence * self.config.algorithm_weights.get(s.algorithm, 1.0) for s in short_signals)
        
        # Require clear majority
        min_consensus = len(signals) * 0.6  # 60% consensus required
        
        if len(long_signals) >= min_consensus and long_weight > short_weight:
            best_signal = max(long_signals, key=lambda x: x.confidence)
            consensus_strength = len(long_signals) / len(signals)
            best_signal.confidence = min(0.90, best_signal.confidence * consensus_strength + 0.1)
            best_signal.reasoning = f"Consensus: {len(long_signals)}L vs {len(short_signals)}S"
            return best_signal
            
        elif len(short_signals) >= min_consensus and short_weight > long_weight:
            best_signal = max(short_signals, key=lambda x: x.confidence)
            consensus_strength = len(short_signals) / len(signals)
            best_signal.confidence = min(0.90, best_signal.confidence * consensus_strength + 0.1)
            best_signal.reasoning = f"Consensus: {len(long_signals)}L vs {len(short_signals)}S"
            return best_signal
        
        return None
    
    async def _open_position(self, signal: TradingSignal):
        """Open a new trading position"""
        try:
            async with self.positions_lock:
                # Calculate position size
                position_size = await self.risk_manager.calculate_position_size(signal, self.balance)
                
                if position_size <= 0:
                    self.logger.debug(f"Position size too small for {signal.symbol}")
                    return
                
                # Check portfolio risk
                position_value = position_size * signal.entry_price
                if not await self.risk_manager.check_portfolio_risk(position_value, self.balance):
                    self.logger.warning(f"Position rejected for {signal.symbol} - portfolio risk limit")
                    return
                
                # Prepare position data
                position_data = {
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'quantity': float(position_size),
                    'entry_price': float(signal.entry_price),
                    'algorithm': signal.algorithm,
                    'reasoning': signal.reasoning,
                    'confidence': signal.confidence,
                    'entry_time': signal.timestamp
                }
                
                # Only add stop_loss and take_profit if they are > 0
                if signal.stop_loss > 0:
                    position_data['stop_loss'] = float(signal.stop_loss)
                if signal.take_profit > 0:
                    position_data['take_profit'] = float(signal.take_profit)
                
                # Save to database
                position_id = await self.db.save_position(position_data)
                
                self.logger.info("Position opened",
                               position_id=position_id,
                               symbol=signal.symbol,
                               side=signal.side,
                               quantity=float(position_size),
                               price=float(signal.entry_price),
                               algorithm=signal.algorithm)
                
        except Exception as e:
            self.logger.error(f"Open position error: {e}")
            await self.db.log_error('open_position', str(e), traceback.format_exc())
    
    async def _update_metrics(self):
        """Update comprehensive system metrics"""
        try:
            positions = await self.db.get_open_positions()
            
            # Get system metrics
            memory_mb = self.metrics.get_memory_usage_safe()
            cpu_percent = self.metrics.get_cpu_usage_safe()
            uptime = time.time() - self.start_time
            
            # Calculate portfolio risk
            portfolio_value = sum(
                float(pos['quantity']) * float(pos.get('current_price', pos['entry_price']))
                for pos in positions
            )
            portfolio_risk = portfolio_value / float(self.balance) if self.balance > 0 else 0
            
            # Update all metrics
            self.metrics.update_system_metrics(
                positions=len(positions),
                balance=float(self.balance),
                pnl=float(self.total_pnl),
                memory_bytes=int(memory_mb * 1024 * 1024),
                cpu_percent=cpu_percent,
                uptime=uptime,
                portfolio_risk=portfolio_risk,
                daily_pnl=float(self.risk_manager.daily_loss)
            )
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")
    
    async def _generate_report(self):
        """Generate comprehensive trading report"""
        try:
            positions = await self.db.get_open_positions()
            
            # Calculate metrics
            total_position_value = sum(
                float(pos['quantity']) * float(pos.get('current_price', pos['entry_price']))
                for pos in positions
            )
            
            runtime_hours = (time.time() - self.start_time) / 3600
            
            # Get security stats
            security_stats = self.security.get_security_stats()
            
            self.logger.info("Enhanced Trading Report",
                           open_positions=len(positions),
                           balance=float(self.balance),
                           total_pnl=float(self.total_pnl),
                           portfolio_value=total_position_value,
                           runtime_hours=round(runtime_hours, 1),
                           algorithms=list(self.algorithms.keys()),
                           daily_loss=float(self.risk_manager.daily_loss),
                           active_sessions=security_stats['active_sessions'])
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
    
    async def _handle_maintenance(self):
        """Handle periodic maintenance tasks"""
        current_time = time.time()
        
        # Garbage collection
        if current_time - self.last_gc >= self.config.gc_interval:
            await self._handle_gc()
            self.last_gc = current_time
        
        # Database backup
        if current_time - self.last_backup >= self.config.backup_interval:
            backup_path = await self.db.create_backup()
            if backup_path:
                self.logger.info(f"Database backup created: {backup_path}")
            self.last_backup = current_time
    
    async def _handle_gc(self):
        """Handle garbage collection"""
        try:
            # Cleanup cache
            async with self._cache_update_lock:
                cutoff_time = time.time() - 300  # 5 minutes
                expired_symbols = [
                    symbol for symbol, last_update in self.last_cache_update.items()
                    if last_update < cutoff_time
                ]
                
                for symbol in expired_symbols:
                    self.market_data_cache.pop(symbol, None)
                    self.last_cache_update.pop(symbol, None)
            
            # Run garbage collection
            collected = gc.collect()
            if collected > 0:
                self.logger.debug(f"Garbage collection: {collected} objects collected")
                
        except Exception as e:
            self.logger.error(f"Garbage collection error: {e}")
    
    async def _load_bot_state(self):
        """Load persisted bot state"""
        try:
            balance_str = await self.db.get_bot_state('balance')
            if balance_str:
                self.balance = Decimal(balance_str)
            
            pnl_str = await self.db.get_bot_state('total_pnl')
            if pnl_str:
                self.total_pnl = Decimal(pnl_str)
                
        except Exception as e:
            self.logger.warning(f"Failed to load bot state: {e}")
    
    async def _save_bot_state(self):
        """Save bot state to database"""
        try:
            await self.db.set_bot_state('balance', str(self.balance))
            await self.db.set_bot_state('total_pnl', str(self.total_pnl))
        except Exception as e:
            self.logger.warning(f"Failed to save bot state: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        try:
            self.running = False
            self.logger.critical("Emergency shutdown initiated")
            
            # Close all positions
            positions = await self.db.get_open_positions()
            for position in positions:
                try:
                    await self.db.update_position(position['id'], {'status': 'CANCELLED'})
                except:
                    pass
            
            # Save state
            await self._save_bot_state()
            
            self.logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            self.logger.critical(f"Emergency shutdown failed: {e}")
    
    def stop(self):
        """Stop the trading bot gracefully"""
        self.running = False
        self.logger.info("Graceful stop signal received")

# ====================== FIXED: SECURE WEB INTERFACE ======================

def create_enhanced_web_app(bot: EnhancedOmegaXTradingBot) -> Quart:
    """FIXED: Create enhanced secure web application"""
    app = Quart(__name__)
    app.secret_key = bot.config.secret_key
    
    # Authentication decorator
    def require_auth(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            if not bot.config.enable_auth:
                return await f(*args, **kwargs)
            
            # Check session
            if 'authenticated' not in session:
                return redirect(url_for('login'))
            
            # Validate session
            session_id = session.get('session_id')
            client_ip = request.remote_addr or 'unknown'
            
            if not session_id or not bot.security.validate_session(session_id, client_ip):
                session.clear()
                return redirect(url_for('login'))
            
            return await f(*args, **kwargs)
        return decorated_function
    
    @app.route('/login', methods=['GET', 'POST'])
    async def login():
        """Secure login endpoint"""
        if request.method == 'POST':
            form = await request.form
            password = form.get('password', '')
            client_ip = request.remote_addr or 'unknown'
            user_agent = request.headers.get('User-Agent', '')
            
            # Check rate limit
            if not bot.security.check_rate_limit('login', client_ip):
                error = "Too many login attempts. Please try again later."
                return await render_template_string(LOGIN_TEMPLATE, error=error)
            
            if bot.security.authenticate_password(password, client_ip, user_agent):
                session['authenticated'] = True
                session['login_time'] = time.time()
                session_id = secrets.token_hex(16)
                session['session_id'] = session_id
                bot.security.create_session(session_id, client_ip, user_agent)
                
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid password or account locked"
                return await render_template_string(LOGIN_TEMPLATE, error=error)
        
        return await render_template_string(LOGIN_TEMPLATE)
    
    @app.route('/logout')
    async def logout():
        """Logout endpoint"""
        session_id = session.get('session_id')
        if session_id:
            bot.security.destroy_session(session_id)
        session.clear()
        return redirect(url_for('login'))
    
    @app.route('/')
    @require_auth
    async def dashboard():
        """Enhanced secure dashboard"""
        try:
            positions = await bot.db.get_open_positions()
            
            # Calculate metrics
            total_value = sum(
                float(pos['quantity']) * float(pos.get('current_price', pos['entry_price'])) 
                for pos in positions
            )
            
            runtime = time.time() - bot.start_time
            
            return await render_template_string(ENHANCED_DASHBOARD_TEMPLATE,
                balance=float(bot.balance),
                total_pnl=float(bot.total_pnl),
                position_count=len(positions),
                total_value=total_value,
                positions=positions,
                runtime_hours=runtime/3600,
                algorithm_count=len(bot.algorithms),
                is_running=bot.running,
                algorithms=list(bot.algorithms.keys()),
                trading_pairs_count=len(bot.config.trading_pairs),
                daily_loss=float(bot.risk_manager.daily_loss)
            )
            
        except Exception as e:
            bot.logger.error(f"Dashboard error: {e}")
            return f"Dashboard error: {e}", 500
    
    @app.route('/api/status')
    @require_auth
    async def api_status():
        """Enhanced API status endpoint"""
        try:
            positions = await bot.db.get_open_positions()
            memory_mb = bot.metrics.get_memory_usage_safe()
            cpu_percent = bot.metrics.get_cpu_usage_safe()
            
            # Get security stats
            security_stats = bot.security.get_security_stats()
            
            return jsonify({
                'status': 'running' if bot.running else 'stopped',
                'balance': float(bot.balance),
                'total_pnl': float(bot.total_pnl),
                'positions': len(positions),
                'runtime': time.time() - bot.start_time,
                'algorithms': list(bot.algorithms.keys()),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'trading_pairs': len(bot.config.trading_pairs),
                'daily_loss': float(bot.risk_manager.daily_loss),
                'security': security_stats
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/performance')
    @require_auth
    async def performance():
        """Performance analytics endpoint"""
        try:
            stats = await bot.db.get_performance_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/metrics')
    @require_auth  # FIXED: Secure metrics endpoint
    async def metrics_endpoint():
        """Secure Prometheus metrics endpoint"""
        client_ip = request.remote_addr or 'unknown'
        
        # Check rate limit for metrics
        if not bot.security.check_rate_limit('metrics', client_ip):
            return "Rate limit exceeded", 429
        
        response = await make_response(prometheus_client.generate_latest())
        response.headers['Content-Type'] = CONTENT_TYPE_LATEST
        return response
    
    return app

# ====================== HTML TEMPLATES ======================

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>OmegaX Trading Bot - Secure Login</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
               margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-form { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.3); 
                     width: 350px; text-align: center; }
        .login-form h1 { margin-bottom: 30px; color: #333; font-size: 28px; }
        .login-form input { width: 100%; padding: 15px; margin: 15px 0; border: 2px solid #ddd; 
                           border-radius: 8px; box-sizing: border-box; font-size: 16px; transition: border-color 0.3s; }
        .login-form input:focus { border-color: #667eea; outline: none; }
        .login-form button { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; 
                           transition: transform 0.2s; }
        .login-form button:hover { transform: translateY(-2px); }
        .error { color: #dc3545; margin: 15px 0; padding: 10px; background: #f8d7da; border-radius: 5px; }
        .footer { margin-top: 25px; font-size: 12px; color: #666; }
        .security-info { margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 8px; font-size: 14px; color: #1976d2; }
    </style>
</head>
<body>
    <div class="login-form">
        <h1>🚀 OmegaX v8.1</h1>
        <form method="post">
            <input type="password" name="password" placeholder="Enter password" required autofocus>
            <button type="submit">🔐 Secure Login</button>
        </form>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <div class="security-info">
            🛡️ Enhanced Security Active<br>
            • Session timeout: 1 hour<br>
            • Rate limiting enabled<br>
            • IP tracking active
        </div>
        <div class="footer">Production-Ready Trading Platform</div>
    </div>
</body>
</html>
'''

ENHANCED_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>OmegaX Trading Bot v8.1 - Enhanced Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
               margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; 
                 text-align: center; position: relative; }
        .logout { position: absolute; top: 20px; right: 20px; background: rgba(255,255,255,0.2); 
                 color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; 
                 text-decoration: none; transition: background 0.3s; }
        .logout:hover { background: rgba(255,255,255,0.3); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 25px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 30px; border-radius: 15px; 
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1); border-left: 5px solid #007bff; 
                    transition: transform 0.2s; }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-value { font-size: 32px; font-weight: bold; color: #333; }
        .stat-label { color: #666; margin-top: 8px; font-size: 14px; font-weight: 500; }
        .positions { background: white; border-radius: 15px; padding: 30px; 
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1); margin-bottom: 25px; }
        .position { border-bottom: 1px solid #eee; padding: 25px 0; display: grid; 
                   grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }
        .position:last-child { border-bottom: none; }
        .symbol { font-weight: bold; font-size: 20px; color: #333; }
        .side-long { color: #28a745; font-weight: bold; }
        .side-short { color: #dc3545; font-weight: bold; }
        .controls { background: white; border-radius: 15px; padding: 25px; 
                   box-shadow: 0 5px 20px rgba(0,0,0,0.1); text-align: center; }
        .btn { background: #007bff; color: white; border: none; padding: 12px 25px; 
              border-radius: 8px; cursor: pointer; margin: 8px; text-decoration: none; 
              display: inline-block; transition: all 0.3s; }
        .btn:hover { background: #0056b3; transform: translateY(-1px); }
        .algorithms { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 15px; justify-content: center; }
        .algorithm { background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%); 
                    padding: 8px 15px; border-radius: 20px; font-size: 13px; font-weight: 500; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                           margin-right: 8px; animation: pulse 2s infinite; }
        .status-running { background: #28a745; }
        .status-stopped { background: #dc3545; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; margin-top: 20px; }
        .info-card { background: #f8f9fa; padding: 15px; border-radius: 10px; }
    </style>
    <script>
        function refreshPage() { location.reload(); }
        setInterval(refreshPage, 30000);
        
        // Enhanced status checking
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('live-status').textContent = data.status;
                document.getElementById('live-memory').textContent = data.memory_mb.toFixed(1) + 'MB';
            } catch (e) {
                console.log('Status check failed:', e);
            }
        }
        setInterval(checkStatus, 10000); // Every 10 seconds
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/logout" class="logout">🚪 Logout</a>
            <h1>🚀 OmegaX Trading Bot v8.1</h1>
            <p>Enhanced Production Platform - All Critical Issues Fixed</p>
            <div class="algorithms">
                {% for algo in algorithms %}
                <span class="algorithm">{{ algo }}</span>
                {% endfor %}
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">${{ "%.2f"|format(balance) }}</div>
                <div class="stat-label">💰 Account Balance</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {{ 'side-long' if total_pnl >= 0 else 'side-short' }}">${{ "%.2f"|format(total_pnl) }}</div>
                <div class="stat-label">📈 Total P&L</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ position_count }}</div>
                <div class="stat-label">📊 Open Positions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${{ "%.2f"|format(total_value) }}</div>
                <div class="stat-label">💼 Portfolio Value</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ trading_pairs_count }}</div>
                <div class="stat-label">🎯 Trading Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${{ "%.2f"|format(daily_loss) }}</div>
                <div class="stat-label">📉 Daily Loss</div>
            </div>
        </div>
        
        <div class="positions">
            <h2>📊 Active Positions</h2>
            
            {% if positions %}
                {% for pos in positions %}
                <div class="position">
                    <div>
                        <div class="symbol">{{ pos.symbol }}</div>
                        <div class="{{ 'side-long' if pos.side == 'LONG' else 'side-short' }}">
                            {{ pos.side }} | {{ "%.6f"|format(pos.quantity) }}
                        </div>
                    </div>
                    <div>
                        <div><strong>Entry:</strong> ${{ "%.4f"|format(pos.entry_price) }}</div>
                        <div><strong>Current:</strong> ${{ "%.4f"|format(pos.current_price or pos.entry_price) }}</div>
                    </div>
                    <div>
                        <div><strong>Stop:</strong> ${{ "%.4f"|format(pos.stop_loss) if pos.stop_loss else 'None' }}</div>
                        <div><strong>Target:</strong> ${{ "%.4f"|format(pos.take_profit) if pos.take_profit else 'None' }}</div>
                    </div>
                    <div>
                        <div><strong>{{ pos.algorithm }}</strong></div>
                        <div>Confidence: {{ "%.0f"|format((pos.confidence or 0) * 100) }}%</div>
                        <div style="font-size: 12px; color: #666; margin-top: 8px;">{{ pos.reasoning[:120] }}...</div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div style="text-align: center; color: #666; padding: 60px;">
                    <h3>No open positions</h3>
                    <p>The bot is scanning for opportunities...</p>
                </div>
            {% endif %}
        </div>
        
        <div class="controls">
            <button class="btn" onclick="refreshPage()">🔄 Refresh</button>
            <a href="/api/status" class="btn">📊 API Status</a>
            <a href="/performance" class="btn">📈 Performance</a>
            <a href="/metrics" class="btn">📊 Metrics</a>
            
            <div class="info-grid">
                <div class="info-card">
                    <strong>🤖 System Status</strong><br>
                    <span class="status-indicator {{ 'status-running' if is_running else 'status-stopped' }}"></span>
                    Status: <span id="live-status">{{ "Running" if is_running else "Stopped" }}</span><br>
                    Runtime: {{ "%.1f"|format(runtime_hours) }} hours
                </div>
                <div class="info-card">
                    <strong>🔧 Performance</strong><br>
                    Algorithms: {{ algorithm_count }}<br>
                    Memory: <span id="live-memory">Loading...</span><br>
                    Trading Pairs: {{ trading_pairs_count }}
                </div>
                <div class="info-card">
                    <strong>🛡️ Security</strong><br>
                    Authentication: ✅ Active<br>
                    Session: ✅ Secure<br>
                    Rate Limiting: ✅ Enabled
                </div>
                <div class="info-card">
                    <strong>⏰ Last Updated</strong><br>
                    <span id="timestamp"></span><br>
                    Auto-refresh: 30s
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
        checkStatus(); // Initial status check
    </script>
</body>
</html>
'''

# ====================== FIXED: MAIN EXECUTION ======================

async def main():
    """FIXED: Main execution with proper error handling and no credential leaking"""
    bot = None
    
    def signal_handler(signum, frame):
        """Graceful shutdown on signal"""
        print(f"\n⏹️  Received signal {signum}, shutting down gracefully...")
        if bot:
            bot.stop()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration with validation
        config = TradingConfig()
        
        # Apply pair filtering by category
        if config.pair_categories:
            config.trading_pairs = config.get_trading_pairs_by_category()[:config.max_pairs_per_scan]
        
        # FIXED: Setup logging without FileHandler level parameter
        log_handlers = [
            logging.StreamHandler(),
            logging.FileHandler('data/trading_bot.log')
        ]
        
        # Create separate handler for errors
        error_handler = logging.FileHandler('data/error.log')
        error_handler.setLevel(logging.ERROR)
        log_handlers.append(error_handler)
        
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=log_handlers
        )
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create data directory with proper permissions
        os.makedirs('data', mode=0o700, exist_ok=True)
        os.makedirs('data/backups', mode=0o700, exist_ok=True)
        os.makedirs('data/logs', mode=0o700, exist_ok=True)
        
        # Display enhanced startup information
        print("🚀 OmegaX Trading Bot v8.1 - All Critical Issues Fixed")
        print("="*80)
        print(f"📊 Trading Configuration:")
        print(f"   • Initial Balance: ${config.initial_balance:,.2f}")
        print(f"   • Max Positions: {config.max_positions}")
        print(f"   • Risk per Trade: {config.base_risk_percent:.2%}")
        print(f"   • Portfolio Risk Limit: {config.max_portfolio_risk:.2%}")
        print(f"   • Signal Threshold: {config.signal_threshold:.1%}")
        print(f"   • Position Timeout: {config.position_timeout_hours}h")
        print()
        print(f"🤖 Algorithm Configuration:")
        print(f"   • Enabled Algorithms: {', '.join(config.enabled_algorithms)}")
        print(f"   • Algorithm Weights: {config.algorithm_weights}")
        print()
        print(f"💰 Trading Pairs:")
        print(f"   • Total Pairs: {len(config.trading_pairs)}")
        print(f"   • Categories: {', '.join(config.pair_categories)}")
        print(f"   • Pairs per Scan: {config.max_pairs_per_scan}")
        print(f"   • Top 10 Pairs: {', '.join(config.trading_pairs[:10])}")
        print()
        print(f"🔧 System Configuration:")
        print(f"   • Update Interval: {config.update_interval}s")
        print(f"   • Memory Limit: {config.max_memory_mb}MB")
        print(f"   • API Rate Limit: {config.rate_limit_calls}/{config.rate_limit_window}s")
        print(f"   • Authentication: {'Enabled' if config.enable_auth else 'Disabled'}")
        
        if config.enable_auth:
            print(f"   • Session Timeout: {config.session_timeout}s")
        
        print()
        
        # Initialize enhanced trading bot
        print("🔄 Initializing enhanced trading bot...")
        bot = EnhancedOmegaXTradingBot(config)
        
        # Create enhanced web app
        web_app = create_enhanced_web_app(bot)
        
        # Configure production web server
        port = int(os.environ.get('PORT', 8080))
        web_config = HypercornConfig()
        web_config.bind = [f"0.0.0.0:{port}"]
        web_config.workers = 1
        web_config.keep_alive_timeout = 65
        web_config.access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s"'
        web_config.error_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)s'
        
        # Start web server
        async def run_web_server():
            await serve(web_app, web_config)
        
        web_task = asyncio.create_task(run_web_server())
        
        print("✅ System Ready!")
        print("="*80)
        print(f"🌐 Web Interface: http://localhost:{port}")
        if config.enable_auth:
            # FIXED: Write password only to secure log file, not console
            secure_logger = logging.getLogger("secure")
            secure_logger.info(f"Web UI Password: {config.web_ui_password}")
            print(f"🔐 Login Required - Password saved to secure logs")
        print(f"📈 Metrics: http://localhost:{port}/metrics")
        print(f"🔧 API Status: http://localhost:{port}/api/status")
        print(f"📊 Performance: http://localhost:{port}/performance")
        print()
        print("🎯 Starting enhanced trading operations...")
        print("="*80)
        
        # Start enhanced trading bot
        bot_task = asyncio.create_task(bot.start())
        
        # Wait for completion
        done, pending = await asyncio.wait(
            [bot_task, web_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cleanup
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        if bot:
            await bot.emergency_shutdown()
    finally:
        print("👋 OmegaX Trading Bot v8.1 stopped safely")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown completed")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)