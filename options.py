import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator
from datetime import datetime, timedelta
from scipy.stats import norm
import pytz
import warnings
warnings.filterwarnings('ignore')

# Ultra-Enhanced CSS for Intraday
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 900;
    }
    .live-badge {
        display: inline-block;
        padding: 8px 20px;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 12px rgba(255,65,108,0.4);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .signal-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: glow 2s infinite;
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 8px 16px rgba(0,0,0,0.2); }
        50% { box-shadow: 0 12px 24px rgba(0,0,0,0.3); }
    }
    .buy-call {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        border: 4px solid #00ff88;
    }
    .buy-put {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: 4px solid #ff6666;
    }
    .wait {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: 4px solid #ffcc55;
    }
    .intraday-action-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease;
    }
    .intraday-action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .buy-action {
        border-left-color: #00c851;
        background: linear-gradient(135deg, #f0fff4 0%, #c8f7dc 100%);
    }
    .sell-action {
        border-left-color: #ff4444;
        background: linear-gradient(135deg, #fff0f0 0%, #ffc9c9 100%);
    }
    .wait-action {
        border-left-color: #ff8800;
        background: linear-gradient(135deg, #fff8f0 0%, #ffe8cc 100%);
    }
    .intraday-timer {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 15px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .next-hour-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .scalping-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .probability-bar {
        background: #e0e0e0;
        height: 40px;
        border-radius: 20px;
        overflow: hidden;
        margin: 15px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .probability-fill {
        background: linear-gradient(90deg, #00c851 0%, #96c93d 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

def get_market_hours():
    """Determine if market is open and time until close"""
    try:
        # US Market hours (9:30 AM - 4:00 PM ET)
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = market_open <= now_et <= market_close and now_et.weekday() < 5
        
        if is_open:
            time_to_close = market_close - now_et
            hours = int(time_to_close.seconds // 3600)
            minutes = int((time_to_close.seconds % 3600) // 60)
            return True, f"{hours}h {minutes}m"
        else:
            return False, "Market Closed"
    except:
        return None, "Unknown"

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes for intraday (hours to expiry)"""
    try:
        S = max(float(S), 0.01)
        K = max(float(K), 0.01)
        T = max(float(T), 1/365/24)  # Minimum 1 hour
        sigma = max(float(sigma), 0.01)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.01)
    except:
        return 0.01

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Greeks for intraday trading"""
    try:
        S = max(float(S), 0.01)
        K = max(float(K), 0.01)
        T = max(float(T), 1/365/24)
        sigma = max(float(sigma), 0.01)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        if option_type == 'call':
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'Delta': round(delta, 4),
            'Gamma': round(gamma, 4),
            'Theta': round(theta, 4),
            'Vega': round(vega, 4),
            'Rho': round(rho, 4)
        }
    except:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}

@st.cache_data(ttl=60)  # Cache for only 1 minute for real-time
def fetch_realtime_data(symbol, interval="5m"):
    """Fetch real-time intraday data"""
    try:
        ticker = yf.Ticker(symbol)
        # Get today's data only
        data = ticker.history(period="1d", interval=interval)
        
        if data.empty:
            # Fallback to 5 days if today's data not available
            data = ticker.history(period="5d", interval=interval)
        
        if data.empty:
            return None
        return data.dropna()
    except:
        return None

def calculate_intraday_indicators(data):
    """Calculate indicators optimized for same-day trading"""
    df = data.copy()
    
    if len(df) < 20:
        return df
    
    try:
        # Price metrics
        df['Returns'] = df['Close'].pct_change()
        df['Intraday_High'] = df['High'].expanding().max()
        df['Intraday_Low'] = df['Low'].expanding().min()
        df['Intraday_Range'] = ((df['Intraday_High'] - df['Intraday_Low']) / df['Close']) * 100
        
        # Intraday Volatility (last hour)
        df['Intraday_Vol'] = df['Returns'].rolling(window=12).std() * np.sqrt(252) * 100  # 12 periods = 1 hour at 5m
        
        # Fast EMAs for scalping
        for period in [3, 5, 8, 13, 21]:
            if len(df) >= period:
                df[f'EMA_{period}'] = EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # Momentum (very responsive)
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['RSI_Fast'] = RSIIndicator(df['Close'], window=6).rsi()
        
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # MACD for momentum shifts
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands (tight for intraday)
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR for stop losses
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
        
        # VWAP (critical for intraday)
        df['VWAP'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df['Price_vs_VWAP'] = ((df['Close'] - df['VWAP']) / df['VWAP']) * 100
        
        # Volume
        df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Momentum_3'] = df['Close'].pct_change(3) * 100
        df['Momentum_5'] = df['Close'].pct_change(5) * 100
        df['Momentum_10'] = df['Close'].pct_change(10) * 100
        
        return df.ffill().bfill().fillna(0)
    
    except Exception as e:
        st.warning(f"Indicator error: {str(e)}")
        return data.ffill().bfill().fillna(0)

def generate_intraday_signal(data, current_price):
    """Generate same-day intraday trading signals"""
    
    if len(data) < 20:
        return {
            'action': 'WAIT',
            'direction': 'NEUTRAL',
            'confidence': 'LOW',
            'score': 0,
            'entry_price': current_price,
            'target_price': current_price,
            'stop_loss': current_price,
            'reasons': ['Need more data for same-day analysis'],
            'risk_level': 'UNKNOWN',
            'timeframe': 'Wait',
            'win_probability': 50,
            'strength': 0,
            'strategy': 'Waiting for clear signal',
            'exit_time': 'N/A'
        }
    
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # 1. FAST EMA SCALPING (35 points)
    if all(col in data.columns for col in ['EMA_3', 'EMA_8', 'EMA_21']):
        ema3 = data['EMA_3'].iloc[-1]
        ema8 = data['EMA_8'].iloc[-1]
        ema21 = data['EMA_21'].iloc[-1]
        ema3_prev = data['EMA_3'].iloc[-2]
        ema8_prev = data['EMA_8'].iloc[-2]
        
        # Perfect scalping setup
        if ema3 > ema8 > ema21:
            buy_score += 30
            reasons.append("🔥 SCALPING SETUP: Perfect bullish EMA stack (3>8>21)")
        elif ema3 < ema8 < ema21:
            sell_score += 30
            reasons.append("🔥 SCALPING SETUP: Perfect bearish EMA stack (3<8<21)")
        # Crossovers (immediate action)
        elif ema3 > ema8 and ema3_prev <= ema8_prev:
            buy_score += 25
            reasons.append("⚡ FRESH CROSSOVER: EMA3 just crossed above EMA8")
        elif ema3 < ema8 and ema3_prev >= ema8_prev:
            sell_score += 25
            reasons.append("⚡ FRESH CROSSOVER: EMA3 just crossed below EMA8")
        elif ema3 > ema8:
            buy_score += 15
            reasons.append("📈 Short-term bullish (EMA3 > EMA8)")
        else:
            sell_score += 15
            reasons.append("📉 Short-term bearish (EMA3 < EMA8)")
    
    # 2. RSI MOMENTUM (25 points)
    if 'RSI_Fast' in data.columns:
        rsi_fast = data['RSI_Fast'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # Extreme oversold/overbought for quick reversals
        if rsi_fast < 20:
            buy_score += 20
            reasons.append(f"💪 EXTREME OVERSOLD: Fast RSI at {rsi_fast:.1f} - Quick bounce expected")
        elif rsi_fast < 30:
            buy_score += 12
            reasons.append(f"📊 Oversold: Fast RSI at {rsi_fast:.1f}")
        elif rsi_fast > 80:
            sell_score += 20
            reasons.append(f"⚠️ EXTREME OVERBOUGHT: Fast RSI at {rsi_fast:.1f} - Quick drop expected")
        elif rsi_fast > 70:
            sell_score += 12
            reasons.append(f"📊 Overbought: Fast RSI at {rsi_fast:.1f}")
        
        # Momentum building
        if 40 < rsi < 60 and rsi_fast > rsi:
            buy_score += 5
            reasons.append("⚡ Momentum building (Fast RSI > Slow RSI)")
        elif 40 < rsi < 60 and rsi_fast < rsi:
            sell_score += 5
            reasons.append("⚡ Momentum fading (Fast RSI < Slow RSI)")
    
    # 3. VWAP POSITION (20 points)
    if 'Price_vs_VWAP' in data.columns:
        vwap_dist = data['Price_vs_VWAP'].iloc[-1]
        vwap_dist_prev = data['Price_vs_VWAP'].iloc[-2]
        
        # VWAP is the intraday anchor
        if vwap_dist > 0.3 and vwap_dist_prev <= 0:
            buy_score += 15
            reasons.append(f"✅ VWAP BREAKOUT: Price just broke above VWAP (+{vwap_dist:.2f}%)")
        elif vwap_dist < -0.3 and vwap_dist_prev >= 0:
            sell_score += 15
            reasons.append(f"❌ VWAP BREAKDOWN: Price just broke below VWAP ({vwap_dist:.2f}%)")
        elif vwap_dist > 0.5:
            buy_score += 10
            reasons.append(f"💪 Strong above VWAP (+{vwap_dist:.2f}%)")
        elif vwap_dist < -0.5:
            sell_score += 10
            reasons.append(f"💪 Strong below VWAP ({vwap_dist:.2f}%)")
    
    # 4. VOLUME SPIKE (15 points)
    if 'Volume_Ratio' in data.columns:
        vol_ratio = data['Volume_Ratio'].iloc[-1]
        
        if vol_ratio > 3:
            if buy_score > sell_score:
                buy_score += 15
                reasons.append(f"🔥 MASSIVE VOLUME SPIKE: {vol_ratio:.1f}x average - Confirming bullish move")
            elif sell_score > buy_score:
                sell_score += 15
                reasons.append(f"🔥 MASSIVE VOLUME SPIKE: {vol_ratio:.1f}x average - Confirming bearish move")
        elif vol_ratio > 1.8:
            if buy_score > sell_score:
                buy_score += 8
                reasons.append(f"📊 High volume: {vol_ratio:.1f}x average")
            elif sell_score > buy_score:
                sell_score += 8
                reasons.append(f"📊 High volume: {vol_ratio:.1f}x average")
    
    # 5. MOMENTUM SHIFTS (5 points)
    if 'Momentum_3' in data.columns:
        mom3 = data['Momentum_3'].iloc[-1]
        
        if mom3 > 0.3:
            buy_score += 5
            reasons.append(f"⚡ Strong upward momentum: +{mom3:.2f}% in 15min")
        elif mom3 < -0.3:
            sell_score += 5
            reasons.append(f"⚡ Strong downward momentum: {mom3:.2f}% in 15min")
    
    # Calculate metrics
    net_score = buy_score - sell_score
    total_score = buy_score + sell_score
    
    win_probability = 50 + (net_score / 100 * 50) if total_score > 0 else 50
    win_probability = max(30, min(90, win_probability))
    
    # Confidence
    if abs(net_score) >= 45:
        confidence = 'VERY HIGH'
        strength = 10
    elif abs(net_score) >= 30:
        confidence = 'HIGH'
        strength = 8
    elif abs(net_score) >= 18:
        confidence = 'MEDIUM'
        strength = 6
    else:
        confidence = 'LOW'
        strength = 3
    
    # Risk level
    risk_level = 'LOW' if confidence in ['VERY HIGH', 'HIGH'] else 'MEDIUM' if confidence == 'MEDIUM' else 'HIGH'
    
    # Calculate prices (tighter for intraday)
    atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.01
    intraday_vol = data['Intraday_Vol'].iloc[-1] if 'Intraday_Vol' in data.columns else 20
    
    # Tighter multipliers for same-day trading
    vol_mult = max(intraday_vol / 25, 0.8)
    
    if net_score >= 30:
        action = "BUY CALL"
        direction = "BULLISH"
        entry_price = current_price
        target_price = current_price * (1 + (0.008 * vol_mult * (strength / 10)))  # 0.8% target
        stop_loss = current_price * (1 - (0.005 * vol_mult))  # 0.5% stop
        timeframe = "30 min - 2 hours"
        exit_time = "Before 3:30 PM ET"
        strategy = "Quick scalp: ATM Call, exit at target or 3:30 PM"
        
    elif net_score >= 18:
        action = "BUY CALL"
        direction = "BULLISH"
        entry_price = current_price
        target_price = current_price * (1 + (0.006 * vol_mult * (strength / 10)))
        stop_loss = current_price * (1 - (0.004 * vol_mult))
        timeframe = "20 min - 1 hour"
        exit_time = "Before 3:30 PM ET"
        strategy = "Moderate scalp: ATM Call"
        
    elif net_score <= -30:
        action = "BUY PUT"
        direction = "BEARISH"
        entry_price = current_price
        target_price = current_price * (1 - (0.008 * vol_mult * (strength / 10)))
        stop_loss = current_price * (1 + (0.005 * vol_mult))
        timeframe = "30 min - 2 hours"
        exit_time = "Before 3:30 PM ET"
        strategy = "Quick scalp: ATM Put, exit at target or 3:30 PM"
        
    elif net_score <= -18:
        action = "BUY PUT"
        direction = "BEARISH"
        entry_price = current_price
        target_price = current_price * (1 - (0.006 * vol_mult * (strength / 10)))
        stop_loss = current_price * (1 + (0.004 * vol_mult))
        timeframe = "20 min - 1 hour"
        exit_time = "Before 3:30 PM ET"
        strategy = "Moderate scalp: ATM Put"
        
    else:
        action = "WAIT"
        direction = "NEUTRAL"
        entry_price = current_price
        target_price = current_price
        stop_loss = current_price
        timeframe = "Wait for setup"
        exit_time = "N/A"
        strategy = "No clear intraday setup - preserve capital"
        reasons.append("⚖️ Choppy market - wait for clearer direction")
    
    return {
        'action': action,
        'direction': direction,
        'confidence': confidence,
        'score': net_score,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'reasons': reasons,
        'risk_level': risk_level,
        'timeframe': timeframe,
        'exit_time': exit_time,
        'win_probability': round(win_probability, 1),
        'strength': strength,
        'strategy': strategy
    }

def calculate_intraday_option_metrics(current_price, strike_price, hours_to_close, volatility, option_type='call', risk_free_rate=0.05):
    """Calculate option metrics for same-day expiry"""
    
    # Convert hours to years for Black-Scholes
    T = max(hours_to_close / (365 * 24), 1/(365*24))
    sigma = max(volatility / 100.0, 0.01)
    
    option_price = black_scholes(current_price, strike_price, T, risk_free_rate, sigma, option_type)
    greeks = calculate_greeks(current_price, strike_price, T, risk_free_rate, sigma, option_type)
    
    if option_type == 'call':
        intrinsic_value = max(current_price - strike_price, 0)
        breakeven = strike_price + option_price
    else:
        intrinsic_value = max(strike_price - current_price, 0)
        breakeven = strike_price - option_price
    
    time_value = option_price - intrinsic_value
    breakeven_move = ((breakeven - current_price) / current_price) * 100
    max_loss = option_price
    
    # Quick profit scenarios
    price_changes = [-2, -1, -0.5, 0, 0.5, 1, 2]
    profit_scenarios = []
    
    for change in price_changes:
        new_price = current_price * (1 + change / 100)
        new_option_price = black_scholes(new_price, strike_price, T, risk_free_rate, sigma, option_type)
        profit = (new_option_price - option_price) * 100
        profit_pct = ((new_option_price - option_price) / option_price) * 100 if option_price > 0 else 0
        
        profit_scenarios.append({
            'price_change': change,
            'new_price': new_price,
            'option_value': new_option_price,
            'profit_loss': profit,
            'profit_pct': profit_pct
        })
    
    return {
        'option_price': option_price,
        'greeks': greeks,
        'intrinsic_value': intrinsic_value,
        'time_value': time_value,
        'breakeven': breakeven,
        'breakeven_move_pct': breakeven_move,
        'max_loss': max_loss,
        'max_profit': "Unlimited" if option_type == 'call' else strike_price - option_price,
        'profit_scenarios': profit_scenarios,
        'contract_cost': option_price * 100,
        'moneyness': 'ITM' if intrinsic_value > 0 else 'ATM' if abs(current_price - strike_price) / current_price < 0.01 else 'OTM'
    }

# Streamlit App
st.set_page_config(page_title="Same-Day Options Trading AI", layout="wide")

st.markdown("<h1 class='main-header'>⚡ SAME-DAY INTRADAY OPTIONS TRADING AI v6.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #667eea; font-size: 1.3rem; font-weight: bold;'>Real-Time Scalping Signals • Exit Before Close • 0DTE Options</p>", unsafe_allow_html=True)

# Market Status
market_open, time_left = get_market_hours()

if market_open:
    st.markdown(f"""
    <div class='live-badge' style='text-align: center; margin: 15px auto; display: block; width: fit-content;'>
        🔴 LIVE MARKET • TIME LEFT: {time_left}
    </div>
    """, unsafe_allow_html=True)
elif market_open is False:
    st.markdown("""
    <div style='background: #ff4444; padding: 15px; border-radius: 12px; color: white; text-align: center; margin: 15px 0;'>
        ⏸️ MARKET CLOSED • Signals for reference only
    </div>
    """, unsafe_allow_html=True)

# Input Section
st.markdown("---")
col_i1, col_i2, col_i3, col_i4 = st.columns(4)

with col_i1:
    symbol = st.text_input("📈 Stock Symbol", value="^NSEBANK", help="Popular: SPY, QQQ, AAPL, TSLA").upper()

with col_i2:
    interval = st.selectbox("⏱️ Interval", ["1m", "2m", "5m"], index=2, help="5m recommended for scalping")

with col_i3:
    option_type_display = st.radio("🔄 Option", ['📈 Call', '📉 Put'])
    option_type = 'call' if '📈' in option_type_display else 'put'

with col_i4:
    # Hours until market close for 0DTE
    if market_open:
        hours_left = float(time_left.split('h')[0]) if 'h' in time_left else 0.5
        hours_to_expiry = st.slider("⏰ Hours to Exit", 0.25, 6.5, min(hours_left, 3.0), 0.25)
    else:
        hours_to_expiry = st.slider("⏰ Hours to Exit", 0.25, 6.5, 2.0, 0.25)

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5) / 100
        auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=False)
    with col_adv2:
        show_charts = st.checkbox("📊 Show Technical Charts", value=True)

if auto_refresh and market_open:
    import time
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 60:
        st.rerun()
        st.session_state.last_refresh = time.time()

# Fetch and analyze
with st.spinner('⚡ Fetching real-time data & analyzing...'):
    data = fetch_realtime_data(symbol, interval)

if data is not None and not data.empty:
    data_with_indicators = calculate_intraday_indicators(data)
    
    current_price = data['Close'].iloc[-1]
    current_time = data.index[-1].strftime('%H:%M:%S')
    
    # Intraday volatility
    if 'Intraday_Vol' in data_with_indicators.columns:
        intraday_vol = max(data_with_indicators['Intraday_Vol'].iloc[-1], 15.0)
    else:
        intraday_vol = 25.0
    
    # Generate signals
    signals = generate_intraday_signal(data_with_indicators, current_price)
    
    # Calculate option metrics
    strike_price = round(current_price)
    option_metrics = calculate_intraday_option_metrics(
        current_price, strike_price, hours_to_expiry, intraday_vol, option_type, risk_free_rate
    )
    
    # Market Overview
    open_price = data['Open'].iloc[0]
    day_change = ((current_price - open_price) / open_price) * 100
    day_high = data['High'].max()
    day_low = data['Low'].min()
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>SYMBOL</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{symbol}</div>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>CURRENT</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>${current_price:.2f}</div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>{day_change:+.2f}% today</div>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>DAY RANGE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>${day_low:.2f} - ${day_high:.2f}</div>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>INTRADAY VOL</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{intraday_vol:.1f}%</div>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>LAST UPDATE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Intraday Timer
    if market_open:
        st.markdown(f"""
        <div class='intraday-timer'>
            ⏰ INTRADAY TRADING • EXIT ALL POSITIONS BY {signals['exit_time']} • TIME REMAINING: {time_left}
        </div>
        """, unsafe_allow_html=True)
    
    # Main Signal
    st.markdown("---")
    st.subheader("⚡ REAL-TIME INTRADAY SIGNAL")
    
    if signals['action'] == 'BUY CALL':
        signal_class = 'buy-call'
        action_emoji = '🚀'
        action_text = 'BUY CALL - SCALP LONG'
    elif signals['action'] == 'BUY PUT':
        signal_class = 'buy-put'
        action_emoji = '📉'
        action_text = 'BUY PUT - SCALP SHORT'
    else:
        signal_class = 'wait'
        action_emoji = '⏸️'
        action_text = 'WAIT - NO SETUP'
    
    st.markdown(f"""
    <div class='signal-box {signal_class}'>
        <div style='font-size: 2rem; margin-bottom: 15px;'>{action_emoji} {action_text} {action_emoji}</div>
        <div style='font-size: 1.3rem; margin-bottom: 12px;'>{signals['confidence']} Confidence • {signals['strength']}/10 Strength</div>
        <div style='font-size: 1.1rem;'>
            📊 <strong>Score:</strong> {signals['score']:+d}/100 | 
            ⏰ <strong>Hold:</strong> {signals['timeframe']} | 
            🎯 <strong>Win Rate:</strong> {signals['win_probability']}%
        </div>
        <div style='font-size: 1rem; margin-top: 8px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 8px;'>
            💡 <strong>Strategy:</strong> {signals['strategy']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Win probability bar
    prob_width = signals['win_probability']
    st.markdown(f"""
    <div style='margin: 20px 0;'>
        <div style='text-align: center; margin-bottom: 10px; font-size: 1.1rem; font-weight: bold; color: #667eea;'>
            Success Probability for This Scalp
        </div>
        <div class='probability-bar'>
            <div class='probability-fill' style='width: {prob_width}%;'>
                {signals['win_probability']}% WIN RATE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trade Execution
    if signals['action'] != 'WAIT':
        st.markdown("---")
        st.subheader("⚡ INTRADAY SCALPING PLAN")
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        profit_potential = ((signals['target_price'] - signals['entry_price']) / signals['entry_price']) * 100
        loss_potential = ((signals['stop_loss'] - signals['entry_price']) / signals['entry_price']) * 100
        
        with col_ex1:
            st.markdown(f"""
            <div class='intraday-action-card buy-action'>
                <h3 style='color: #00c851; margin-bottom: 15px;'>🎯 ENTRY NOW</h3>
                <h1 style='color: #007e33; margin: 15px 0;'>${signals['entry_price']:.2f}</h1>
                <p style='color: #666; margin: 10px 0;'>Enter immediately at market</p>
                <div style='background: rgba(0,200,81,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;'>
                    <strong style='font-size: 0.9rem;'>{signals['strategy']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ex2:
            st.markdown(f"""
            <div class='intraday-action-card buy-action'>
                <h3 style='color: #00c851; margin-bottom: 15px;'>💰 TARGET</h3>
                <h1 style='color: #007e33; margin: 15px 0;'>${signals['target_price']:.2f}</h1>
                <p style='color: #666; margin: 10px 0;'>Take profit or exit by {signals['exit_time']}</p>
                <div style='background: rgba(0,200,81,0.2); padding: 10px; border-radius: 8px; margin-top: 10px;'>
                    <strong style='color: #00c851; font-size: 1.3rem;'>+{profit_potential:.2f}%</strong><br>
                    <span style='font-size: 0.9rem;'>Quick Gain Target</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ex3:
            st.markdown(f"""
            <div class='intraday-action-card sell-action'>
                <h3 style='color: #ff4444; margin-bottom: 15px;'>🛑 STOP LOSS</h3>
                <h1 style='color: #cc0000; margin: 15px 0;'>${signals['stop_loss']:.2f}</h1>
                <p style='color: #666; margin: 10px 0;'>Cut loss immediately if hit</p>
                <div style='background: rgba(255,68,68,0.2); padding: 10px; border-radius: 8px; margin-top: 10px;'>
                    <strong style='color: #ff4444; font-size: 1.3rem;'>{loss_potential:.2f}%</strong><br>
                    <span style='font-size: 0.9rem;'>Max Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Option Details
        st.markdown("---")
        st.subheader("💎 0DTE OPTION PRICING")
        
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        
        with col_o1:
            st.markdown(f"""
            <div class='scalping-box'>
                <h4 style='margin: 0 0 12px 0;'>💰 Premium</h4>
                <h2 style='margin: 15px 0;'>${option_metrics['option_price']:.2f}</h2>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Per share</p>
                <p style='margin: 5px 0 0 0; font-weight: bold;'>${option_metrics['contract_cost']:.0f}</p>
                <p style='margin: 0; font-size: 0.8rem; opacity: 0.8;'>Per contract</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_o2:
            st.markdown(f"""
            <div class='scalping-box'>
                <h4 style='margin: 0 0 12px 0;'>⏰ Time Value</h4>
                <h2 style='margin: 15px 0;'>${option_metrics['time_value']:.2f}</h2>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Decaying fast!</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.8;'>{hours_to_expiry:.1f}h left</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_o3:
            st.markdown(f"""
            <div class='scalping-box'>
                <h4 style='margin: 0 0 12px 0;'>⚠️ Max Risk</h4>
                <h2 style='margin: 15px 0;'>${option_metrics['max_loss']:.2f}</h2>
                <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Per share</p>
                <p style='margin: 5px 0 0 0; font-weight: bold;'>${option_metrics['max_loss'] * 100:.0f}</p>
                <p style='margin: 0; font-size: 0.8rem; opacity: 0.8;'>Per contract</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_o4:
            st.markdown(f"""
            <div class='scalping-box'>
                <h4 style='margin: 0 0 12px 0;'>🔖 Status</h4>
                <h2 style='margin: 15px 0;'>{option_metrics['moneyness']}</h2>
                <p style='margin: 0; font-size: 0.85rem; opacity: 0.9;'>
                    {'In The Money' if option_metrics['moneyness'] == 'ITM' else 'At The Money' if option_metrics['moneyness'] == 'ATM' else 'Out of The Money'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Greeks
        st.markdown("### 🔬 Critical Greeks for Scalping")
        col_g1, col_g2, col_g3 = st.columns(3)
        
        greeks = option_metrics['greeks']
        
        with col_g1:
            st.markdown(f"""
            <div class='next-hour-prediction'>
                <h4 style='margin: 0 0 8px 0;'>Δ Delta: {greeks['Delta']:.3f}</h4>
                <p style='margin: 0; font-size: 0.95rem;'>For every $1 move in {symbol}, option moves ${abs(greeks['Delta']):.2f}</p>
                <p style='margin: 10px 0 0 0; font-size: 0.85rem;'>Example: {symbol} +$1 = Option +${abs(greeks['Delta']):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_g2:
            theta_hourly = greeks['Theta'] / 24
            st.markdown(f"""
            <div class='next-hour-prediction'>
                <h4 style='margin: 0 0 8px 0;'>Θ Theta: ${greeks['Theta']:.3f}/day</h4>
                <p style='margin: 0; font-size: 0.95rem;'>Time decay per hour: ${abs(theta_hourly):.3f}</p>
                <p style='margin: 10px 0 0 0; font-size: 0.85rem; color: #ff4444;'>⚠️ Losing ${abs(theta_hourly):.3f} per hour!</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_g3:
            st.markdown(f"""
            <div class='next-hour-prediction'>
                <h4 style='margin: 0 0 8px 0;'>Γ Gamma: {greeks['Gamma']:.4f}</h4>
                <p style='margin: 0; font-size: 0.95rem;'>Delta changes by {greeks['Gamma']:.4f} per $1 move</p>
                <p style='margin: 10px 0 0 0; font-size: 0.85rem;'>{'High gamma = Fast profits/losses' if greeks['Gamma'] > 0.01 else 'Low gamma = Slower movement'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick P/L scenarios
        st.markdown("---")
        st.subheader("💰 QUICK PROFIT/LOSS SCENARIOS")
        
        fig_pl = go.Figure()
        
        fig_pl.add_trace(go.Bar(
            x=[f"{s['price_change']:+.1f}%" for s in option_metrics['profit_scenarios']],
            y=[s['profit_loss'] for s in option_metrics['profit_scenarios']],
            marker=dict(
                color=[s['profit_loss'] for s in option_metrics['profit_scenarios']],
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f"${s['profit_loss']:.0f}" for s in option_metrics['profit_scenarios']],
            textposition='outside',
            hovertemplate='<b>Stock Move:</b> %{x}<br><b>P/L:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig_pl.update_layout(
            title=f'Profit/Loss per Contract (Current Hour)',
            xaxis_title='Stock Price Change',
            yaxis_title='Profit/Loss ($)',
            height=350,
            template='plotly_white',
            showlegend=False
        )
        
        fig_pl.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2)
        
        st.plotly_chart(fig_pl, use_container_width=True)
    
    else:
        # Wait state
        st.markdown("---")
        st.markdown(f"""
        <div class='intraday-action-card wait-action' style='padding: 40px; text-align: center; min-height: 250px;'>
            <h2 style='color: #ff8800; margin-bottom: 20px;'>⏸️ NO CLEAR INTRADAY SETUP</h2>
            <p style='font-size: 1.2rem; margin: 15px 0; color: #333;'>Market Status: <strong>{signals['direction']}</strong></p>
            <div style='background: rgba(255,136,0,0.1); padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <p style='margin: 10px 0; color: #555; font-size: 1.1rem;'>
                    ⏰ Wait for a clearer setup. In intraday trading, patience is key.
                    Don't force trades in choppy conditions.
                </p>
                <p style='margin: 15px 0 0 0; color: #333; font-weight: bold;'>
                    ✅ Check back in 15-30 minutes or when you see strong momentum
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis Details
    st.markdown("---")
    st.subheader("🔍 SIGNAL ANALYSIS")
    
    col_an1, col_an2 = st.columns(2)
    
    with col_an1:
        st.markdown("#### 📊 Technical Reasons")
        for i, reason in enumerate(signals['reasons'], 1):
            st.markdown(f"""
            <div style='background: white; padding: 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
                <p style='margin: 0; color: #333;'><strong>{i}.</strong> {reason}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_an2:
        st.markdown("#### 📈 Score Breakdown")
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 12px; box-shadow: 0 3px 8px rgba(0,0,0,0.1);'>
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span style='font-weight: bold; color: #00c851;'>🟢 Bullish Signals</span>
                    <span style='font-weight: bold; color: #00c851;'>{signals['buy_score']}/100</span>
                </div>
                <div style='background: #e0e0e0; height: 25px; border-radius: 12px; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, #00c851 0%, #96c93d 100%); height: 100%; width: {signals['buy_score']}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                        {signals['buy_score']}%
                    </div>
                </div>
            </div>
            
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span style='font-weight: bold; color: #ff4444;'>🔴 Bearish Signals</span>
                    <span style='font-weight: bold; color: #ff4444;'>{signals['sell_score']}/100</span>
                </div>
                <div style='background: #e0e0e0; height: 25px; border-radius: 12px; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, #ff4444 0%, #ff4b2b 100%); height: 100%; width: {signals['sell_score']}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                        {signals['sell_score']}%
                    </div>
                </div>
            </div>
            
            <div style='margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; text-align: center;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>NET SCORE</div>
                <div style='font-size: 2rem; font-weight: bold; margin: 5px 0;'>{signals['score']:+d}/100</div>
                <div style='font-size: 0.85rem; opacity: 0.9;'>{signals['direction']} BIAS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Charts
    if show_charts:
        st.markdown("---")
        st.subheader("📊 INTRADAY TECHNICAL CHARTS")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Price Action with Fast EMAs', 'Volume', 'RSI Fast'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data_with_indicators.index,
            open=data_with_indicators['Open'],
            high=data_with_indicators['High'],
            low=data_with_indicators['Low'],
            close=data_with_indicators['Close'],
            name='Price',
            showlegend=False
        ), row=1, col=1)
        
        # Fast EMAs
        ema_colors = {'EMA_3': 'orange', 'EMA_8': 'blue', 'EMA_21': 'purple'}
        for ema, color in ema_colors.items():
            if ema in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators[ema],
                    name=ema,
                    line=dict(color=color, width=2)
                ), row=1, col=1)
        
        # VWAP
        if 'VWAP' in data_with_indicators.columns:
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['VWAP'],
                name='VWAP',
                line=dict(color='brown', width=2.5, dash='dash')
            ), row=1, col=1)
        
        # Add levels
        if signals['action'] != 'WAIT':
            fig.add_hline(y=signals['entry_price'], line_dash="solid", line_color="blue",
                         annotation_text=f"Entry", row=1, col=1)
            fig.add_hline(y=signals['target_price'], line_dash="dash", line_color="green",
                         annotation_text=f"Target", row=1, col=1)
            fig.add_hline(y=signals['stop_loss'], line_dash="dash", line_color="red",
                         annotation_text=f"Stop", row=1, col=1)
        
        # Volume
        colors = ['red' if data_with_indicators['Close'].iloc[i] < data_with_indicators['Open'].iloc[i]
                 else 'green' for i in range(len(data_with_indicators))]
        
        fig.add_trace(go.Bar(
            x=data_with_indicators.index,
            y=data_with_indicators['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # RSI Fast
        if 'RSI_Fast' in data_with_indicators.columns:
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['RSI_Fast'],
                name='RSI Fast',
                line=dict(color='purple', width=2)
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=800,
            title=f"{symbol} - Real-Time Intraday Analysis",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Intraday Rules
    st.markdown("---")
    st.subheader("🛡️ INTRADAY TRADING RULES")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 20px; border-radius: 15px; color: white;'>
            <h4 style='margin: 0 0 15px 0;'>⏰ TIME RULES</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li style='margin: 10px 0;'><strong>Exit by 3:30 PM ET</strong> - No exceptions!</li>
                <li style='margin: 10px 0;'>Never hold 0DTE options overnight</li>
                <li style='margin: 10px 0;'>Best times: 9:30-11 AM, 2-3:30 PM</li>
                <li style='margin: 10px 0;'>Avoid lunch hour (12-1 PM)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white;'>
            <h4 style='margin: 0 0 15px 0;'>💰 POSITION SIZING</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li style='margin: 10px 0;'>Risk only <strong>0.5-1%</strong> per scalp</li>
                <li style='margin: 10px 0;'>Max 1-2 contracts for beginners</li>
                <li style='margin: 10px 0;'>Don't average down losers</li>
                <li style='margin: 10px 0;'>Take profits at 30-50% gains</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 15px; color: white;'>
            <h4 style='margin: 0 0 15px 0;'>🎯 TRADE MANAGEMENT</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li style='margin: 10px 0;'>Set stops immediately after entry</li>
                <li style='margin: 10px 0;'>Take partial profits at 30% gain</li>
                <li style='margin: 10px 0;'>Move stops to breakeven at +20%</li>
                <li style='margin: 10px 0;'>Never turn scalp into investment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Failed to fetch data. Please check the stock symbol and try again.")
    st.info("💡 Try popular symbols like SPY, QQQ, AAPL, or TSLA")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚠️ RISK DISCLAIMER:</strong> This is for educational purposes only. Options trading involves substantial risk and is not suitable for all investors. 
    Past performance is not indicative of future results. Always consult with a qualified financial professional before trading.</p>
    <p>Same-Day Options Trading AI v6.0 • Real-time intraday signals • Exit before market close</p>
</div>
""", unsafe_allow_html=True)