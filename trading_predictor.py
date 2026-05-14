import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, GRU,
                                      BatchNormalization, Input, Attention,
                                      MultiHeadAttention, LayerNormalization,
                                      GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator, PSARIndicator, AroonIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
from datetime import datetime, timedelta
from scipy.stats import linregress, pearsonr
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading AI v4.0 — Transformer Enhanced",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    
    .main-header {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6b35 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -1px;
    }
    .sub-header {
        text-align: center; font-family: 'JetBrains Mono', monospace;
        color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; letter-spacing: 2px;
    }
    .signal-card {
        padding: 28px; border-radius: 20px; text-align: center;
        font-weight: 800; margin: 12px 0; position: relative; overflow: hidden;
    }
    .signal-card::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.08) 0%, transparent 60%);
    }
    .buy-card  { background: linear-gradient(135deg, #0a4f2e 0%, #0d7a45 60%, #14a862 100%); border: 2px solid #22d96a; color: #e0fff0; }
    .sell-card { background: linear-gradient(135deg, #4f0a0a 0%, #8b1515 60%, #c21d1d 100%); border: 2px solid #ff4545; color: #fff0f0; }
    .hold-card { background: linear-gradient(135deg, #2a2a0a 0%, #5a5200 60%, #8a7d00 100%); border: 2px solid #f5c518; color: #fffbe0; }
    
    .metric-pill {
        display: inline-block; padding: 5px 14px; border-radius: 50px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        font-weight: 600; margin: 3px;
    }
    .pill-blue   { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }
    .pill-purple { background: rgba(123,47,247,0.15); color: #a97fff; border: 1px solid rgba(123,47,247,0.3); }
    .pill-green  { background: rgba(34,217,106,0.15); color: #22d96a; border: 1px solid rgba(34,217,106,0.3); }
    .pill-red    { background: rgba(255,69,69,0.15);  color: #ff6b6b; border: 1px solid rgba(255,69,69,0.3); }
    .pill-amber  { background: rgba(245,197,24,0.15); color: #f5c518; border: 1px solid rgba(245,197,24,0.3); }
    
    .level-box {
        padding: 14px 18px; border-radius: 12px; margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .entry-box  { background: rgba(0,120,255,0.1); border-left: 4px solid #0078ff; }
    .sl-box     { background: rgba(255,50,50,0.1);  border-left: 4px solid #ff3232; }
    .t1-box     { background: rgba(20,180,80,0.12); border-left: 4px solid #14b450; }
    .t2-box     { background: rgba(20,200,80,0.1);  border-left: 4px solid #22d96a; }
    .t3-box     { background: rgba(20,220,80,0.08); border-left: 4px solid #44ffaa; }
    
    .reason-item {
        padding: 7px 12px; margin: 4px 0; border-radius: 8px;
        background: rgba(255,255,255,0.04); border-left: 3px solid #444;
        font-size: 0.88rem;
    }
    .score-bar-outer {
        background: rgba(255,255,255,0.08); border-radius: 50px;
        height: 10px; margin: 4px 0; overflow: hidden;
    }
    .score-bar-inner { height: 100%; border-radius: 50px; transition: width 0.8s ease; }
    
    .model-badge {
        display: inline-block; padding: 4px 10px; border-radius: 6px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        background: rgba(123,47,247,0.2); color: #c39fff; border: 1px solid rgba(123,47,247,0.4);
        margin: 2px;
    }
    
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    
    .stAlert { border-radius: 12px; }
    
    .regime-badge {
        padding: 8px 20px; border-radius: 8px; font-weight: 700;
        font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
        display: inline-block;
    }
    .regime-bull { background: rgba(34,217,106,0.2); color: #22d96a; border: 1px solid rgba(34,217,106,0.4); }
    .regime-bear { background: rgba(255,69,69,0.2);  color: #ff6b6b; border: 1px solid rgba(255,69,69,0.4); }
    .regime-side { background: rgba(245,197,24,0.2); color: #f5c518; border: 1px solid rgba(245,197,24,0.4); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_intraday_data(symbol, interval="1m"):
    try:
        ticker = yf.Ticker(symbol)
        period = "7d" if interval in ["1m", "2m", "5m"] else "1mo"
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            st.error(f"No data for {symbol}. Market may be closed or invalid symbol.")
            return None
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# ─────────────────────────────────────────────────
# FEATURE ENGINEERING (50+ indicators)
# ─────────────────────────────────────────────────
def calculate_indicators(data):
    df = data.copy()
    if len(df) < 20:
        return df

    try:
        # --- Price-derived ---
        df['Returns']         = df['Close'].pct_change()
        df['Log_Returns']     = np.log(df['Close'] / df['Close'].shift(1))
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['HL_Range']        = (df['High'] - df['Low']) / df['Close'] * 100
        df['Body_Size']       = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
        df['Upper_Shadow']    = (df['High'] - df[['Close','Open']].max(axis=1)) / (df['High'] - df['Low'] + 1e-9)
        df['Lower_Shadow']    = (df[['Close','Open']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-9)
        df['Bullish_Candle']  = (df['Close'] > df['Open']).astype(int)
        df['Doji']            = (df['Body_Size'] < 0.05).astype(int)
        df['Hammer']          = ((df['Lower_Shadow'] > 0.6) & (df['Upper_Shadow'] < 0.15)).astype(int)
        df['Shooting_Star']   = ((df['Upper_Shadow'] > 0.6) & (df['Lower_Shadow'] < 0.15)).astype(int)

        # --- Moving Averages ---
        for p in [5, 9, 13, 21, 34, 50, 89]:
            if len(df) >= p:
                df[f'EMA_{p}']  = EMAIndicator(df['Close'], window=p).ema_indicator()
                df[f'SMA_{p}']  = SMAIndicator(df['Close'], window=p).sma_indicator()

        # MA cross signals
        if 'EMA_5' in df.columns and 'EMA_21' in df.columns:
            df['EMA_Spread'] = (df['EMA_5'] - df['EMA_21']) / df['EMA_21'] * 100
            df['EMA_Cross']  = np.sign(df['EMA_5'] - df['EMA_21'])

        # --- Momentum ---
        for p in [6, 9, 14, 21]:
            if len(df) >= p:
                df[f'RSI_{p}'] = RSIIndicator(df['Close'], window=p).rsi()

        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K']    = stoch.stoch()
        df['Stoch_D']    = stoch.stoch_signal()
        df['Stoch_Cross'] = np.sign(df['Stoch_K'] - df['Stoch_D'])

        df['Williams_R'] = WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()

        for p in [5, 10, 20]:
            if len(df) >= p:
                df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100

        if len(df) >= 25:
            tsi = TSIIndicator(df['Close'], window_slow=25, window_fast=13)
            df['TSI']        = tsi.tsi()

        # --- Trend ---
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD']        = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist']   = macd.macd_diff()
        df['MACD_Hist_Slope'] = df['MACD_Hist'].diff()

        if len(df) >= 25:
            adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX']       = adx.adx()
            df['ADX_Pos']   = adx.adx_pos()
            df['ADX_Neg']   = adx.adx_neg()
            df['DI_Spread'] = df['ADX_Pos'] - df['ADX_Neg']

        df['CCI']         = CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()

        if len(df) >= 25:
            aroon = AroonIndicator(df['High'], df['Low'], window=25)
            df['Aroon_Up']   = aroon.aroon_up()
            df['Aroon_Down'] = aroon.aroon_down()
            df['Aroon_Osc']  = df['Aroon_Up'] - df['Aroon_Down']

        # --- Volatility ---
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper']   = bb.bollinger_hband()
        df['BB_Lower']   = bb.bollinger_lband()
        df['BB_Middle']  = bb.bollinger_mavg()
        df['BB_Width']   = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Pos']     = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).mean() * 0.85).astype(int)

        df['ATR']        = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ATR_Pct']    = df['ATR'] / df['Close'] * 100

        kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        df['KC_Upper']   = kc.keltner_channel_hband()
        df['KC_Lower']   = kc.keltner_channel_lband()
        df['KC_Pos']     = (df['Close'] - df['KC_Lower']) / (df['KC_Upper'] - df['KC_Lower'] + 1e-9)

        dc = DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        df['DC_Upper']   = dc.donchian_channel_hband()
        df['DC_Lower']   = dc.donchian_channel_lband()
        df['DC_Pos']     = (df['Close'] - df['DC_Lower']) / (df['DC_Upper'] - df['DC_Lower'] + 1e-9)

        # Rolling volatility (multiple windows)
        for p in [5, 10, 20]:
            df[f'Volatility_{p}'] = df['Returns'].rolling(p).std() * np.sqrt(252) * 100
        df['Vol_Ratio']  = df['Volatility_5'] / (df['Volatility_20'] + 1e-9)   # vol regime

        # --- Volume ---
        df['VWAP']       = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df['VWAP_Dev']   = (df['Close'] - df['VWAP']) / df['VWAP'] * 100

        df['OBV']        = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['OBV_EMA']    = df['OBV'].ewm(span=20).mean()
        df['OBV_Slope']  = df['OBV'].diff(5) / (df['OBV'].abs().rolling(5).mean() + 1e-9)

        df['MFI']        = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()

        df['CMF']        = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()

        df['AD']         = AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
        df['AD_Slope']   = df['AD'].diff(5) / (df['AD'].abs().rolling(5).mean() + 1e-9)

        df['Vol_SMA']    = df['Volume'].rolling(20).mean()
        df['Vol_Ratio_Raw'] = df['Volume'] / (df['Vol_SMA'] + 1e-9)
        df['Vol_Spike']  = (df['Vol_Ratio_Raw'] > 2.5).astype(int)

        # --- Support / Resistance ---
        df['Support']    = df['Low'].rolling(20, center=True).min()
        df['Resistance'] = df['High'].rolling(20, center=True).max()
        df['Dist_Sup']   = (df['Close'] - df['Support'])    / df['Close'] * 100
        df['Dist_Res']   = (df['Resistance'] - df['Close']) / df['Close'] * 100

        # --- Pivot Points ---
        pivot = (df['High'].iloc[0] + df['Low'].iloc[0] + df['Close'].iloc[0]) / 3
        r = df['High'].iloc[0] - df['Low'].iloc[0]
        df['Pivot'] = pivot
        df['R1'] = 2 * pivot - df['Low'].iloc[0]
        df['S1'] = 2 * pivot - df['High'].iloc[0]
        df['R2'] = pivot + r
        df['S2'] = pivot - r

        # --- Trend strength via linear regression ---
        def slope_norm(s):
            if len(s) < 2: return 0.0
            x = np.arange(len(s))
            sl, *_ = linregress(x, s)
            return float(sl) / (s.mean() + 1e-9) * 100
        df['LR_Slope_20'] = df['Close'].rolling(20).apply(slope_norm, raw=False)
        df['LR_Slope_5']  = df['Close'].rolling(5).apply(slope_norm, raw=False)

        # --- Market Regime (Hidden Markov-inspired heuristic) ---
        # Trend: ADX > 25 + DI
        if 'ADX' in df.columns:
            df['Trending']  = (df['ADX'] > 25).astype(int)
            df['Bull_Trend']= ((df['ADX'] > 25) & (df['DI_Spread'] > 0)).astype(int)
            df['Bear_Trend']= ((df['ADX'] > 25) & (df['DI_Spread'] < 0)).astype(int)

        # Price momentum multi-window
        for p in [3, 5, 10, 15, 20]:
            df[f'Mom_{p}'] = df['Close'].pct_change(p) * 100

        # Consecutive bullish/bearish candles
        df['Bull_Streak'] = df['Bullish_Candle'].groupby(
            (df['Bullish_Candle'] != df['Bullish_Candle'].shift()).cumsum()).cumcount() + 1
        df['Bull_Streak'] *= df['Bullish_Candle']  # zero out bear streaks
        df['Bear_Streak'] = (1 - df['Bullish_Candle']).groupby(
            (df['Bullish_Candle'] != df['Bullish_Candle'].shift()).cumsum()).cumcount() + 1
        df['Bear_Streak'] *= (1 - df['Bullish_Candle'])

        # ── NaN cleanup ──
        df = df.ffill().bfill()
        df = df.dropna(axis=1, how='all')
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df = df.dropna()
        return df

    except Exception as e:
        st.warning(f"Indicator error: {e}")
        return data.ffill().bfill().fillna(0)


# ─────────────────────────────────────────────────
# SIGNAL GENERATION (multi-factor scoring)
# ─────────────────────────────────────────────────
def generate_signals(data):
    MIN_ROWS = 30
    if len(data) < MIN_ROWS:
        return dict(Signal='WAIT', Strength=0, Entry_Price=None, Stop_Loss=None,
                    Target_1=None, Target_2=None, Target_3=None,
                    Confidence=0, Reasons=['Insufficient data'],
                    Risk_Reward='N/A', Win_Probability=50,
                    Buy_Score=0, Sell_Score=0, Net_Score=0)

    cp    = data['Close'].iloc[-1]
    atr   = data['ATR'].iloc[-1] if 'ATR' in data.columns else cp * 0.01
    buy   = 0
    sell  = 0
    reas  = []
    MAX   = 120   # max possible score

    # ── 1. TREND (25 pts) ──
    if all(c in data.columns for c in ['EMA_5','EMA_21','EMA_50']):
        e5, e21, e50 = data['EMA_5'].iloc[-1], data['EMA_21'].iloc[-1], data['EMA_50'].iloc[-1]
        e5p, e21p    = data['EMA_5'].iloc[-2],  data['EMA_21'].iloc[-2]
        if e5 > e21 > e50:
            buy += 18; reas.append("🔥 Perfect Bullish EMA Stack (5>21>50)")
        elif e5 < e21 < e50:
            sell += 18; reas.append("🔥 Perfect Bearish EMA Stack (5<21<50)")
        elif e5 > e21 and e5p <= e21p:
            buy += 22; reas.append("⭐ Golden Cross: EMA5 crossed above EMA21")
        elif e5 < e21 and e5p >= e21p:
            sell += 22; reas.append("⭐ Death Cross: EMA5 crossed below EMA21")
        elif e5 > e21:
            buy += 8; reas.append("📈 Short-term uptrend")
        else:
            sell += 8; reas.append("📉 Short-term downtrend")

    if 'Aroon_Osc' in data.columns:
        ao = data['Aroon_Osc'].iloc[-1]
        if ao > 60:  buy += 5;  reas.append(f"📈 Aroon bullish dominance ({ao:.0f})")
        elif ao < -60: sell += 5; reas.append(f"📉 Aroon bearish dominance ({ao:.0f})")

    if 'LR_Slope_20' in data.columns:
        s20 = data['LR_Slope_20'].iloc[-1]
        if s20 > 0.15:  buy += 5;  reas.append("📊 Positive linear trend (20-bar)")
        elif s20 < -0.15: sell += 5; reas.append("📊 Negative linear trend (20-bar)")

    # ── 2. MOMENTUM (30 pts) ──
    for rsi_col in ['RSI_14','RSI_9']:
        if rsi_col in data.columns:
            rsi  = data[rsi_col].iloc[-1]
            rsip = data[rsi_col].iloc[-2] if len(data) > 2 else 50
            if rsi < 20:   buy  += 14; reas.append(f"💪 {rsi_col} Extremely Oversold ({rsi:.1f})")
            elif rsi < 30: buy  += 9;  reas.append(f"💪 {rsi_col} Oversold ({rsi:.1f})")
            elif rsi > 80: sell += 14; reas.append(f"⚠️ {rsi_col} Extremely Overbought ({rsi:.1f})")
            elif rsi > 70: sell += 9;  reas.append(f"⚠️ {rsi_col} Overbought ({rsi:.1f})")
            if rsi > 50 and rsip <= 50:  buy  += 4; reas.append(f"📊 {rsi_col} crossed above 50")
            elif rsi < 50 and rsip >= 50: sell += 4; reas.append(f"📊 {rsi_col} crossed below 50")
            break   # use first available RSI

    if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
        sk, sd  = data['Stoch_K'].iloc[-1], data['Stoch_D'].iloc[-1]
        skp, sdp = data['Stoch_K'].iloc[-2], data['Stoch_D'].iloc[-2]
        if sk < 20 and sk > sd and skp <= sdp:  buy  += 9; reas.append("🎯 Stochastic bullish cross in oversold zone")
        elif sk > 80 and sk < sd and skp >= sdp: sell += 9; reas.append("🎯 Stochastic bearish cross in overbought zone")
        elif sk < 25:  buy  += 4
        elif sk > 75:  sell += 4

    if 'TSI' in data.columns:
        tsi  = data['TSI'].iloc[-1]
        tsip = data['TSI'].iloc[-2] if len(data) > 2 else 0
        if tsi > 0 and tsip <= 0:  buy  += 6; reas.append("⚡ TSI bullish zero-cross")
        elif tsi < 0 and tsip >= 0: sell += 6; reas.append("⚡ TSI bearish zero-cross")

    # ── 3. MACD (15 pts) ──
    if 'MACD_Hist' in data.columns:
        mh  = data['MACD_Hist'].iloc[-1]
        mhs = data['MACD_Hist_Slope'].iloc[-1] if 'MACD_Hist_Slope' in data.columns else 0
        m,  ms  = data['MACD'].iloc[-1], data['MACD_Signal'].iloc[-1]
        mp, msp = data['MACD'].iloc[-2], data['MACD_Signal'].iloc[-2]
        if m > ms and mp <= msp:  buy  += 13; reas.append("⚡ MACD bullish crossover")
        elif m < ms and mp >= msp: sell += 13; reas.append("⚡ MACD bearish crossover")
        if mh > 0 and mhs > 0:  buy  += 5; reas.append("📈 MACD histogram expanding bullishly")
        elif mh < 0 and mhs < 0: sell += 5; reas.append("📉 MACD histogram expanding bearishly")
        elif mh > 0:  buy  += 2
        elif mh < 0:  sell += 2

    # ── 4. VOLUME (15 pts) ──
    if 'Vol_Ratio_Raw' in data.columns:
        vr = data['Vol_Ratio_Raw'].iloc[-1]
        if vr > 3.0:
            if buy > sell:  buy  += 12; reas.append("🔥 Massive volume spike confirming bullish move")
            elif sell > buy: sell += 12; reas.append("🔥 Massive volume spike confirming bearish move")
        elif vr > 1.8:
            if buy > sell:  buy  += 6; reas.append("📊 Above-average volume with bullish bias")
            elif sell > buy: sell += 6; reas.append("📊 Above-average volume with bearish bias")

    if 'CMF' in data.columns:
        cmf = data['CMF'].iloc[-1]
        if cmf > 0.15:  buy  += 6; reas.append(f"💰 Chaikin Money Flow strongly positive ({cmf:.2f})")
        elif cmf > 0.05: buy += 3
        elif cmf < -0.15: sell += 6; reas.append(f"💰 Chaikin Money Flow strongly negative ({cmf:.2f})")
        elif cmf < -0.05: sell += 3

    if 'OBV_Slope' in data.columns:
        obv_s = data['OBV_Slope'].iloc[-1]
        if obv_s > 0.02:  buy  += 3; reas.append("📦 OBV trending up (accumulation)")
        elif obv_s < -0.02: sell += 3; reas.append("📦 OBV trending down (distribution)")

    # ── 5. PRICE LEVELS (15 pts) ──
    if 'VWAP_Dev' in data.columns:
        vd = data['VWAP_Dev'].iloc[-1]
        if vd > 0.6:    buy  += 9;  reas.append(f"✅ Price {vd:.2f}% above VWAP")
        elif vd < -0.6: sell += 9;  reas.append(f"❌ Price {abs(vd):.2f}% below VWAP")
        elif 0 < vd < 0.6:  buy  += 3
        elif -0.6 < vd < 0: sell += 3

    if 'BB_Pos' in data.columns:
        bp = data['BB_Pos'].iloc[-1]
        if bp < 0.08:    buy  += 8; reas.append("🎯 Price at lower Bollinger Band")
        elif bp > 0.92:  sell += 8; reas.append("🎯 Price at upper Bollinger Band")

        if 'BB_Squeeze' in data.columns and data['BB_Squeeze'].iloc[-1] == 1:
            reas.append("⚡ Bollinger Squeeze — breakout imminent")

    # ── 6. MONEY FLOW (10 pts) ──
    if 'MFI' in data.columns:
        mfi = data['MFI'].iloc[-1]
        if mfi < 20:    buy  += 9; reas.append(f"💰 MFI Extremely Oversold ({mfi:.1f})")
        elif mfi < 30:  buy  += 5; reas.append(f"💰 MFI Oversold ({mfi:.1f})")
        elif mfi > 80:  sell += 9; reas.append(f"💰 MFI Extremely Overbought ({mfi:.1f})")
        elif mfi > 70:  sell += 5; reas.append(f"💰 MFI Overbought ({mfi:.1f})")

    # ── 7. CANDLESTICK PATTERNS (5 pts) ──
    if 'Hammer' in data.columns and data['Hammer'].iloc[-1]:
        buy  += 5; reas.append("🕯️ Hammer pattern (bullish reversal)")
    if 'Shooting_Star' in data.columns and data['Shooting_Star'].iloc[-1]:
        sell += 5; reas.append("🕯️ Shooting Star (bearish reversal)")

    # ── 8. MARKET REGIME FILTER ──
    if 'Bull_Trend' in data.columns:
        if data['Bull_Trend'].iloc[-1]:  buy  += 3
        elif data['Bear_Trend'].iloc[-1] if 'Bear_Trend' in data.columns else False: sell += 3

    # ── COMPUTE SIGNAL ──
    net      = buy - sell
    total    = buy + sell
    conf     = min((abs(net) / MAX) * 100, 97) if total > 0 else 50
    win_prob = max(22, min(94, 50 + net / MAX * 50))

    # Dynamic ATR multipliers (scale with volatility)
    vol_factor = 1.0
    if 'Vol_Ratio' in data.columns:
        vf = data['Vol_Ratio'].iloc[-1]
        vol_factor = max(0.8, min(1.6, vf))

    sl_mult = 1.5 * vol_factor
    t1_mult = 2.0 * vol_factor
    t2_mult = 3.5 * vol_factor
    t3_mult = 5.5 * vol_factor

    if abs(net) >= 45:   strength, stype = 10, 'STRONG'
    elif abs(net) >= 30: strength, stype = 8,  'STRONG'
    elif abs(net) >= 18: strength, stype = 6,  ''
    elif abs(net) >= 10: strength, stype = 4,  ''
    else:                strength, stype = 0,  ''

    if net >= 15:
        sig = f'{stype} BUY'.strip()
        ep  = cp
        sl  = ep - atr * sl_mult
        t1  = ep + atr * t1_mult
        t2  = ep + atr * t2_mult
        t3  = ep + atr * t3_mult
    elif net <= -15:
        sig = f'{stype} SELL'.strip()
        ep  = cp
        sl  = ep + atr * sl_mult
        t1  = ep - atr * t1_mult
        t2  = ep - atr * t2_mult
        t3  = ep - atr * t3_mult
    else:
        sig = 'HOLD'
        ep  = cp; sl = t1 = t2 = t3 = None; strength = 0
        reas.append("⚖️ Market consolidating — no directional edge")

    rr = 'N/A'
    if sl and t1:
        risk   = abs(ep - sl)
        reward = abs(t1 - ep)
        rr = f'1:{reward/risk:.2f}' if risk > 0 else 'N/A'

    return dict(Signal=sig, Strength=strength, Entry_Price=ep, Stop_Loss=sl,
                Target_1=t1, Target_2=t2, Target_3=t3,
                Confidence=conf, Reasons=reas, Risk_Reward=rr,
                Win_Probability=win_prob, Buy_Score=buy, Sell_Score=sell, Net_Score=net)


# ─────────────────────────────────────────────────
# FEATURE SELECTION — top features for ML
# ─────────────────────────────────────────────────
ML_FEATURES = [
    'Close','Volume','RSI_14','RSI_9','MACD','MACD_Signal','MACD_Hist',
    'Stoch_K','Stoch_D','ATR','ATR_Pct','BB_Pos','BB_Width',
    'VWAP_Dev','OBV_Slope','CMF','MFI','Vol_Ratio_Raw','Vol_Ratio',
    'EMA_Spread','EMA_Cross','LR_Slope_20','LR_Slope_5',
    'Body_Size','Upper_Shadow','Lower_Shadow','Bull_Streak','Bear_Streak',
    'Mom_5','Mom_10','Mom_15','Dist_Sup','Dist_Res',
]


# ─────────────────────────────────────────────────
# MODEL 1: Temporal Convolutional + Attention LSTM
# ─────────────────────────────────────────────────
def build_tcn_attention_model(lookback, n_features):
    inp = Input(shape=(lookback, n_features))

    # CNN branch — capture local patterns
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inp)
    x = Conv1D(64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = Conv1D(32, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Multi-Head Self-Attention
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    attn = LayerNormalization()(attn + x)

    # Bidirectional LSTM
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(attn)
    lstm1 = Dropout(0.25)(lstm1)
    lstm2 = LSTM(32)(lstm1)
    lstm2 = BatchNormalization()(lstm2)

    # MLP head
    out = Dense(64, activation='gelu')(lstm2)
    out = Dropout(0.2)(out)
    out = Dense(32, activation='gelu')(out)
    out = Dense(1)(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='huber',
        metrics=['mae']
    )
    return model


# ─────────────────────────────────────────────────
# MODEL 2: GRU with Residual connections
# ─────────────────────────────────────────────────
def build_residual_gru(lookback, n_features):
    inp  = Input(shape=(lookback, n_features))
    x    = Bidirectional(GRU(64, return_sequences=True))(inp)
    skip = x
    x    = BatchNormalization()(x)
    x    = Dropout(0.2)(x)
    x    = Bidirectional(GRU(64, return_sequences=True))(x)
    x    = BatchNormalization()(x + skip)   # residual
    x    = GRU(32)(x)
    x    = Dense(32, activation='relu')(x)
    x    = Dropout(0.15)(x)
    out  = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0004, clipnorm=1.0), loss='huber')
    return model


# ─────────────────────────────────────────────────
# ENSEMBLE ML PREDICTION
# ─────────────────────────────────────────────────
def ensemble_prediction(data, steps=15):
    try:
        avail_cols = [c for c in ML_FEATURES if c in data.columns]
        if len(avail_cols) < 5 or len(data) < 60:
            return None, None, None, None, None, None

        feat_df = data[avail_cols].copy().ffill().bfill().dropna()
        if len(feat_df) < 50:
            return None, None, None, None, None, None

        # ── Scaling: RobustScaler is better for financial data ──
        scaler      = RobustScaler()
        price_scaler = RobustScaler()

        scaled       = scaler.fit_transform(feat_df)
        close_idx    = avail_cols.index('Close')

        prices_raw   = feat_df[['Close']].values
        prices_scaled = price_scaler.fit_transform(prices_raw)

        lookback = min(50, len(scaled) // 2)

        # ── Build sequences ──
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(prices_scaled[i, 0])

        X = np.array(X); y = np.array(y)
        if len(X) < 30:
            return None, None, None, None, None, None

        # ── Train/val split (time-aware) ──
        split = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
        ]

        # ── Model 1: TCN-Attention-LSTM ──
        m1 = build_tcn_attention_model(lookback, len(avail_cols))
        m1.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=25, batch_size=16, callbacks=callbacks, verbose=0)

        # ── Model 2: Residual GRU ──
        m2 = build_residual_gru(lookback, len(avail_cols))
        m2.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=25, batch_size=16, callbacks=callbacks, verbose=0)

        # ── Model 3: HistGradientBoosting ──
        X_gb = scaled[:-1]
        y_gb = prices_scaled[1:, 0]
        m3 = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=6,
            min_samples_leaf=5, random_state=42,
            validation_fraction=0.1, n_iter_no_change=15
        )
        m3.fit(X_gb, y_gb)

        # ── Model 4: ExtraTreesRegressor ──
        m4 = ExtraTreesRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        m4.fit(X_gb, y_gb)

        # ── Model 5: ElasticNet (linear baseline) ──
        m5 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
        m5.fit(X_gb, y_gb)

        # ── Validate DL models — compute val MAPE ──
        v1_pred = m1.predict(X_val, verbose=0).flatten()
        v2_pred = m2.predict(X_val, verbose=0).flatten()
        mape1   = mean_absolute_percentage_error(y_val, v1_pred) if len(y_val) > 0 else 0.1
        mape2   = mean_absolute_percentage_error(y_val, v2_pred) if len(y_val) > 0 else 0.1
        # Lower MAPE → higher weight
        w1 = 1 / (mape1 + 1e-6); w2 = 1 / (mape2 + 1e-6)
        w_sum = w1 + w2; w1 /= w_sum; w2 /= w_sum

        # ── Multi-step prediction ──
        cur_seq   = scaled[-lookback:].copy()
        cur_feat  = scaled[-1:].copy()
        dl_preds, gb_preds = [], []

        for step in range(steps):
            # DL predictions
            Xin   = cur_seq.reshape(1, lookback, len(avail_cols))
            p1    = m1.predict(Xin, verbose=0)[0, 0]
            p2    = m2.predict(Xin, verbose=0)[0, 0]
            pdl   = w1 * p1 + w2 * p2

            # Update rolling window (teacher-free)
            new_row         = cur_seq[-1].copy()
            new_row[close_idx] = pdl
            cur_seq         = np.vstack([cur_seq[1:], new_row])

            # Inverse transform DL
            dummy         = np.zeros((1, len(avail_cols)))
            dummy[0, close_idx] = pdl
            dl_price      = scaler.inverse_transform(dummy)[0, close_idx]
            dl_preds.append(dl_price)

            # GB/ET/EN predictions (only 1-step)
            if step == 0:
                gb_price  = price_scaler.inverse_transform([[m3.predict(cur_feat)[0]]])[0, 0]
                et_price  = price_scaler.inverse_transform([[m4.predict(cur_feat)[0]]])[0, 0]
                en_price  = price_scaler.inverse_transform([[m5.predict(cur_feat)[0]]])[0, 0]
                gb_preds  = [gb_price, et_price, en_price]

        # ── Ensemble weights: DL 55%, HGB 20%, ET 15%, EN 10% ──
        ensemble = []
        for i, dp in enumerate(dl_preds):
            if gb_preds:
                ep = dp * 0.55 + gb_preds[0] * 0.20 + gb_preds[1] * 0.15 + gb_preds[2] * 0.10
            else:
                ep = dp
            ensemble.append(ep)

        final_pred = ensemble[-1]
        mid_pred   = ensemble[steps // 2]

        # ── Confidence: inversely proportional to prediction spread ──
        pred_std   = np.std(ensemble)
        pred_mean  = abs(np.mean(ensemble))
        spread_pct = pred_std / (pred_mean + 1e-9)
        confidence = max(0.52, min(0.94, 1 - spread_pct * 5))

        # ── Model agreement: std across the 5 model outputs ──
        cur_price = data['Close'].iloc[-1]
        all_1step = [dl_preds[0]] + (gb_preds if gb_preds else [])
        agreement = 1 - (np.std(all_1step) / (cur_price + 1e-9))
        agreement = max(0.0, min(1.0, agreement))

        return final_pred, mid_pred, confidence, ensemble, agreement, gb_preds

    except Exception as e:
        st.warning(f"ML Prediction error: {e}")
        return None, None, None, None, None, None


# ─────────────────────────────────────────────────
# MARKET REGIME DETECTOR
# ─────────────────────────────────────────────────
def detect_regime(data):
    regime = "SIDEWAYS"
    color  = "regime-side"
    if 'ADX' in data.columns and 'DI_Spread' in data.columns:
        adx = data['ADX'].iloc[-1]
        di  = data['DI_Spread'].iloc[-1]
        if adx > 28 and di > 5:   regime, color = "UPTREND",   "regime-bull"
        elif adx > 28 and di < -5: regime, color = "DOWNTREND", "regime-bear"
        elif adx < 18:             regime, color = "RANGING",   "regime-side"
    vol_regime = "NORMAL"
    if 'Vol_Ratio' in data.columns:
        vr = data['Vol_Ratio'].iloc[-1]
        if vr > 1.5:   vol_regime = "HIGH VOL"
        elif vr < 0.7: vol_regime = "LOW VOL"
    return regime, color, vol_regime


# ─────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>INTRADAY TRADING AI v4.0</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>TRANSFORMER · ATTENTION · ENSEMBLE · 50+ INDICATORS</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    symbol      = st.text_input("Symbol", value="^NSEBANK").upper()
    interval    = st.selectbox("Interval", ["1m","2m","5m","15m","30m"], index=0)
    pred_steps  = st.slider("Forecast Steps (minutes)", 5, 30, 15)
    show_adv    = st.checkbox("Advanced Charts", value=True)
    auto_ref    = st.checkbox("Auto-refresh (30s)", value=False)

    st.markdown("---")
    st.markdown("""
    **v4.0 Upgrades:**
    - 🧠 TCN + Multi-Head Attention + LSTM
    - 🔁 Residual GRU (parallel model)
    - 🌲 HistGradientBoosting + ExtraTrees
    - 📐 ElasticNet linear baseline
    - ⚖️ Adaptive ensemble weighting (MAPE)
    - 📏 RobustScaler (handles outliers)
    - ⏱️ EarlyStopping + LR scheduling
    - 🕯️ 50+ indicators incl. candlestick
    - 📊 Market regime detection
    - 🎯 ADX-based trend filter
    """)
    st.markdown("---")
    st.warning("⚠️ Educational use only. Not financial advice.")

    if auto_ref:
        import time
        if 'lr' not in st.session_state: st.session_state.lr = time.time()
        if time.time() - st.session_state.lr > 30:
            st.rerun(); st.session_state.lr = time.time()
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Fetch ──
with st.spinner('Fetching market data…'):
    raw = fetch_intraday_data(symbol, interval)

if raw is None or raw.empty:
    st.error("❌ No data. Check the symbol, market hours, or try a different interval.")
    st.stop()

# ── Indicators ──
with st.spinner('Computing 50+ technical indicators…'):
    data = calculate_indicators(raw)

# ── Signals ──
with st.spinner('Generating multi-factor signals…'):
    sig = generate_signals(data)

# ── ML Prediction ──
with st.spinner(f'Training 5-model ensemble & forecasting {pred_steps} steps…'):
    ml_out = ensemble_prediction(data, steps=pred_steps)
    pred_final, pred_mid, pred_conf, all_preds, agreement, ml_1step = ml_out

cp   = data['Close'].iloc[-1]
ts   = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
regime, reg_color, vol_regime = detect_regime(data)

# ─────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.metric("Symbol",        symbol)
h2.metric("Current Price", f"${cp:.2f}")
h3.metric("Last Update",   data.index[-1].strftime('%H:%M:%S'))
h4.metric("Data Points",   len(data))

st.markdown(f"""
<div style='text-align:center; margin: 8px 0;'>
    <span class='regime-badge {reg_color}'>{regime}</span>&nbsp;
    <span class='metric-pill pill-blue'>VOL: {vol_regime}</span>&nbsp;
    <span class='metric-pill pill-purple'>NET SCORE: {sig["Net_Score"]:+d}</span>&nbsp;
    <span class='metric-pill pill-green'>WIN PROB: {sig["Win_Probability"]:.0f}%</span>&nbsp;
    <span class='metric-pill pill-amber'>CONFIDENCE: {sig["Confidence"]:.1f}%</span>
    <br><br>
    <span class='model-badge'>TCN-Attention</span>
    <span class='model-badge'>Residual GRU</span>
    <span class='model-badge'>HistGBM</span>
    <span class='model-badge'>ExtraTrees</span>
    <span class='model-badge'>ElasticNet</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# MAIN SIGNAL CARD
# ─────────────────────────────────────────────────
_, mid_col, _ = st.columns([1,2,1])
with mid_col:
    card_class = 'buy-card' if 'BUY' in sig['Signal'] else 'sell-card' if 'SELL' in sig['Signal'] else 'hold-card'
    emoji      = '🚀' if 'BUY' in sig['Signal'] else '📉' if 'SELL' in sig['Signal'] else '⏸️'
    st.markdown(f"""
    <div class='signal-card {card_class}'>
        <div style='font-size:2.5rem'>{emoji} {sig['Signal']} {emoji}</div>
        <div style='font-size:1rem; margin-top:10px; opacity:0.9'>
            Strength: {sig['Strength']}/10 &nbsp;|&nbsp; Confidence: {sig['Confidence']:.1f}%
        </div>
        <div style='font-size:0.95rem; margin-top:4px; opacity:0.8'>
            Risk-Reward: {sig['Risk_Reward']} &nbsp;|&nbsp; Win Rate: {sig['Win_Probability']:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# ENTRY / EXIT LEVELS
# ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Entry & Exit Levels")
lc = st.columns(5)
fmt = lambda v: f"${v:.2f}" if v else "N/A"
pct = lambda v: f"{abs((v - cp)/cp*100):.2f}%" if v else ""

with lc[0]:
    st.markdown(f"<div class='level-box entry-box'><b>Entry</b><br>{fmt(sig['Entry_Price'])}</div>", unsafe_allow_html=True)
with lc[1]:
    st.markdown(f"<div class='level-box sl-box'><b>Stop Loss</b><br>{fmt(sig['Stop_Loss'])}<br><small>Risk: {pct(sig['Stop_Loss'])}</small></div>", unsafe_allow_html=True)
with lc[2]:
    st.markdown(f"<div class='level-box t1-box'><b>Target 1</b><br>{fmt(sig['Target_1'])}<br><small>+{pct(sig['Target_1'])}</small></div>", unsafe_allow_html=True)
with lc[3]:
    st.markdown(f"<div class='level-box t2-box'><b>Target 2</b><br>{fmt(sig['Target_2'])}<br><small>+{pct(sig['Target_2'])}</small></div>", unsafe_allow_html=True)
with lc[4]:
    st.markdown(f"<div class='level-box t3-box'><b>Target 3</b><br>{fmt(sig['Target_3'])}<br><small>+{pct(sig['Target_3'])}</small></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# AI FORECAST
# ─────────────────────────────────────────────────
if pred_final and pred_conf and all_preds:
    st.markdown("---")
    st.subheader(f"🤖 5-Model Ensemble Forecast — Next {pred_steps} Minutes")

    chg_mid   = (pred_mid   - cp) / cp * 100
    chg_final = (pred_final - cp) / cp * 100

    fc = st.columns(5)
    fc[0].metric("Current",       f"${cp:.2f}")
    fc[1].metric(f"{pred_steps//2}-Min Pred", f"${pred_mid:.2f}",   f"{chg_mid:+.2f}%")
    fc[2].metric(f"{pred_steps}-Min Pred",    f"${pred_final:.2f}", f"{chg_final:+.2f}%")
    fc[3].metric("AI Confidence", f"{pred_conf:.1%}")
    fc[4].metric("Model Agreement", f"{agreement:.1%}" if agreement else "N/A")

    if ml_1step:
        st.markdown(f"""
        <div style='font-size:0.8rem; opacity:0.7; text-align:center; margin-top:4px;'>
        Next-bar: &nbsp;
        HGB: <b>${ml_1step[0]:.2f}</b> &nbsp;|&nbsp;
        ExtraTrees: <b>${ml_1step[1]:.2f}</b> &nbsp;|&nbsp;
        ElasticNet: <b>${ml_1step[2]:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    # Directional message
    if chg_final > 2 and pred_conf > 0.78:
        st.success(f"🚀 **STRONG BULLISH FORECAST** — Expected {chg_final:.2f}% rise (High confidence)")
    elif chg_final > 0.5:
        st.info(f"📈 Bullish forecast: +{chg_final:.2f}% expected")
    elif chg_final < -2 and pred_conf > 0.78:
        st.error(f"📉 **STRONG BEARISH FORECAST** — Expected {abs(chg_final):.2f}% fall (High confidence)")
    elif chg_final < -0.5:
        st.warning(f"🔻 Bearish forecast: {chg_final:.2f}% expected")
    else:
        st.info(f"➡️ Neutral forecast: {chg_final:+.2f}% expected")

    # ── Forecast chart ──
    st.subheader(f"📈 {pred_steps}-Bar AI Forecast Timeline")

    hist_n  = min(60, len(data))
    ht      = data.index[-hist_n:]
    hp      = data['Close'].iloc[-hist_n:].values
    now_t   = data.index[-1]
    fut_t   = [now_t + timedelta(minutes=i) for i in range(len(all_preds) + 1)]
    all_v   = [cp] + all_preds
    std_b   = np.std(all_preds)
    ub      = [p + 1.6 * std_b for p in all_v]
    lb      = [p - 1.6 * std_b for p in all_v]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=ht, y=hp, name='Historical', line=dict(color='#4fc3f7', width=2.5)))
    fig_pred.add_trace(go.Scatter(
        x=fut_t + fut_t[::-1], y=ub + lb[::-1],
        fill='toself', fillcolor='rgba(255,100,100,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig_pred.add_trace(go.Scatter(x=fut_t, y=all_v, name='Ensemble Prediction',
        line=dict(color='#ff6b6b', width=3, dash='dot'), mode='lines+markers', marker=dict(size=5)))
    fig_pred.add_trace(go.Scatter(
        x=[fut_t[pred_steps//2], fut_t[-1]],
        y=[pred_mid, pred_final],
        mode='markers+text',
        marker=dict(size=14, color=['#ff9800','#f44336'], symbol='star'),
        text=[f'${pred_mid:.2f}', f'${pred_final:.2f}'],
        textposition='top center', textfont=dict(size=12),
        name='Key Targets', showlegend=False))
    fig_pred.update_layout(
        template='plotly_dark', hovermode='x unified', height=420,
        title=f'Ensemble AI Forecast — {symbol}',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_pred, use_container_width=True)

    # Forecast stats
    sc = st.columns(5)
    max_p = max(all_preds); min_p = min(all_preds); avg_p = np.mean(all_preds)
    sc[0].metric("Predicted High", f"${max_p:.2f}", f"+{(max_p-cp)/cp*100:.2f}%")
    sc[1].metric("Predicted Low",  f"${min_p:.2f}", f"{(min_p-cp)/cp*100:.2f}%")
    sc[2].metric("Avg Forecast",   f"${avg_p:.2f}", f"{(avg_p-cp)/cp*100:+.2f}%")
    sc[3].metric("Forecast Volatility", f"{np.std(all_preds)/cp*100:.2f}%")
    sc[4].metric("Expected Range", f"${max_p-min_p:.2f}", f"{(max_p-min_p)/cp*100:.2f}%")

# ─────────────────────────────────────────────────
# SIGNAL ANALYSIS
# ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Signal Analysis")
sa1, sa2 = st.columns(2)

with sa1:
    st.markdown("**Supporting Indicators**")
    for i, r in enumerate(sig['Reasons'], 1):
        st.markdown(f"<div class='reason-item'>{i}. {r}</div>", unsafe_allow_html=True)

with sa2:
    st.markdown("**Scoring Breakdown**")
    bs = sig['Buy_Score']; ss = sig['Sell_Score']
    st.markdown(f"""
    <div style='margin:8px 0;'>
        <b>Bullish</b> {bs}/120
        <div class='score-bar-outer'><div class='score-bar-inner' style='width:{bs/120*100:.0f}%;background:linear-gradient(90deg,#22d96a,#14b450);'></div></div>
    </div>
    <div style='margin:8px 0;'>
        <b>Bearish</b> {ss}/120
        <div class='score-bar-outer'><div class='score-bar-inner' style='width:{ss/120*100:.0f}%;background:linear-gradient(90deg,#ff6b6b,#cc2222);'></div></div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    cols[0].metric("Net Score",   f"{sig['Net_Score']:+d}")
    cols[0].metric("Strength",    f"{sig['Strength']}/10")
    cols[1].metric("Confidence",  f"{sig['Confidence']:.1f}%")
    cols[1].metric("Win Prob",    f"{sig['Win_Probability']:.1f}%")

# ─────────────────────────────────────────────────
# ADVANCED CHARTS
# ─────────────────────────────────────────────────
if show_adv:
    st.markdown("---")
    st.subheader("📊 Professional Technical Analysis")

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=('Price · EMAs · VWAP · BB · Targets',
                        'Volume · CMF', 'RSI (6/9/14)', 'MACD', 'Stochastic + MFI'),
        vertical_spacing=0.035,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20]
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price', showlegend=False), row=1, col=1)

    for ema, col in [('EMA_5','#ff9800'),('EMA_13','#2196F3'),('EMA_21','#9c27b0'),('EMA_50','#f44336')]:
        if ema in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[ema], name=ema,
                line=dict(color=col, width=1.4)), row=1, col=1)

    if 'VWAP' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP',
            line=dict(color='#795548', width=2.2, dash='dash')), row=1, col=1)

    if 'BB_Upper' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
            line=dict(color='rgba(150,150,150,0.6)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB',
            fill='tonexty', fillcolor='rgba(150,150,150,0.08)',
            line=dict(color='rgba(150,150,150,0.6)', width=1)), row=1, col=1)

    for level, color, label in [
        (sig['Entry_Price'], 'blue',  'Entry'),
        (sig['Stop_Loss'],   'red',   'SL'),
        (sig['Target_1'],    '#22d96a','T1'),
        (sig['Target_2'],    '#14b450','T2'),
        (sig['Target_3'],    '#0d7a45','T3'),
    ]:
        if level:
            fig.add_hline(y=level, line_color=color, line_dash='dash', line_width=1.5,
                annotation_text=f"{label}: ${level:.2f}", annotation_position="right", row=1, col=1)

    # 2. Volume + CMF
    v_colors = ['#ef5350' if data['Close'].iloc[i] < data['Open'].iloc[i] else '#26a69a'
                for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=v_colors,
        name='Volume', showlegend=False), row=2, col=1)
    if 'CMF' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['CMF']*data['Volume'].max()*0.3,
            name='CMF', line=dict(color='#ff9800', width=1.5)), row=2, col=1)

    # 3. Multi RSI
    for rsi_col, col in [('RSI_6','#ff9800'),('RSI_9','#2196F3'),('RSI_14','#9c27b0')]:
        if rsi_col in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[rsi_col], name=rsi_col,
                line=dict(color=col, width=1.8)), row=3, col=1)
    for level, color in [(70,'red'),(50,'gray'),(30,'green')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color, line_width=1, row=3, col=1)

    # 4. MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
            line=dict(color='#2196F3', width=2)), row=4, col=1)
    if 'MACD_Signal' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal',
            line=dict(color='#f44336', width=2)), row=4, col=1)
    if 'MACD_Hist' in data.columns:
        h_colors = ['#26a69a' if v > 0 else '#ef5350' for v in data['MACD_Hist']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], marker_color=h_colors,
            name='Histogram'), row=4, col=1)

    # 5. Stochastic + MFI
    if 'Stoch_K' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], name='%K',
            line=dict(color='#2196F3', width=2)), row=5, col=1)
    if 'Stoch_D' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], name='%D',
            line=dict(color='#f44336', width=2, dash='dash')), row=5, col=1)
    if 'MFI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MFI'], name='MFI',
            line=dict(color='#9c27b0', width=1.5, dash='dot')), row=5, col=1)
    for level, color in [(80,'red'),(50,'gray'),(20,'green')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color, line_width=1, row=5, col=1)

    fig.update_yaxes(title_text="Price",      row=1, col=1)
    fig.update_yaxes(title_text="Volume",     row=2, col=1)
    fig.update_yaxes(title_text="RSI",  range=[0,100], row=3, col=1)
    fig.update_yaxes(title_text="MACD",       row=4, col=1)
    fig.update_yaxes(title_text="Oscillator", range=[0,100], row=5, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        height=1250, template='plotly_dark',
        title=f"{symbol} — Professional Technical Analysis ({interval})",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(15,15,20,1)',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────
# RISK & RECOMMENDATIONS
# ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Risk & Position Guidance")

risk_level = ("LOW"    if sig['Strength'] >= 8 and sig['Confidence'] > 80 else
              "HIGH"   if sig['Strength'] <= 3 else "MEDIUM")
risk_color = {"LOW": "pill-green", "MEDIUM": "pill-amber", "HIGH": "pill-red"}[risk_level]

if   risk_level == "LOW"    and 'BUY'  in sig['Signal']: pos = "Aggressive (3–5% of capital)";  rec = "STRONG ENTRY — high-conviction setup"
elif risk_level == "LOW"    and 'SELL' in sig['Signal']: pos = "Moderate (2–3% of capital)";    rec = "STRONG SHORT — good risk/reward"
elif risk_level == "MEDIUM" and sig['Strength'] >= 6:    pos = "Moderate (1–2% of capital)";    rec = "MODERATE ENTRY — confirm on next bar"
else:                                                     pos = "Small (0.5–1%) or skip";        rec = "LOW CONVICTION — wait for better setup"

rc = st.columns(3)
rc[0].info(f"**🎯 Recommendation**\n\n{rec}")
rc[1].info(f"**💰 Position Size**\n\n{pos}")
rc[2].info(f"**⚠️ Risk Level:** `{risk_level}`")

mc = st.columns(4)
if 'ATR_Pct' in data.columns:
    atr_p = data['ATR_Pct'].iloc[-1]
    mc[0].metric("ATR %", f"{atr_p:.2f}%", "High Vol" if atr_p > 3 else "Normal")
if 'Volatility_20' in data.columns:
    mc[1].metric("Annualized Vol", f"{data['Volatility_20'].iloc[-1]:.1f}%")
if 'HL_Range' in data.columns:
    mc[2].metric("H-L Range %", f"{data['HL_Range'].iloc[-1]:.2f}%")
if 'Vol_Ratio_Raw' in data.columns:
    mc[3].metric("Volume Ratio", f"{data['Vol_Ratio_Raw'].iloc[-1]:.1f}x",
                 "Spike!" if data['Vol_Spike'].iloc[-1] else "Normal")

st.markdown("---")
st.markdown("""
<div style='text-align:center; opacity:0.5; font-size:0.8rem; padding:10px;'>
    🚀 Intraday Trading AI v4.0 &nbsp;|&nbsp; TCN · Attention · BiLSTM · ResGRU · HGB · ExtraTrees · ElasticNet
    <br>⚠️ For educational purposes only. Not financial advice. Always use proper risk management.
</div>
""", unsafe_allow_html=True)
