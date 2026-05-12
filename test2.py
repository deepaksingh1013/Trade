import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced CSS for intraday trading
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .intraday-signal-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .buy-signal {
        background: linear-gradient(135deg, #00c851 0%, #007E33 100%);
        color: white;
        border: 3px solid #00c851;
    }
    .sell-signal {
        background: linear-gradient(135deg, #ff4444 0%, #CC0000 100%);
        color: white;
        border: 3px solid #ff4444;
    }
    .hold-signal {
        background: linear-gradient(135deg, #ffbb33 0%, #FF8800 100%);
        color: white;
        border: 3px solid #ffbb33;
    }
    .entry-exit-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .price-target {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 10px;
        background: #e8f4f8;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stoploss-box {
        background: #ffe6e6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
    }
    .target-box {
        background: #e6ffe6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00c851;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute only for real-time data
def fetch_intraday_data(symbol, interval="1m"):
    """Fetch real-time intraday data"""
    try:
        ticker = yf.Ticker(symbol)
        # Get 1 day data with chosen interval
        data = ticker.history(period="1d", interval=interval)
        if data.empty:
            st.error(f"No data for {symbol}. Market might be closed or invalid symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_intraday_indicators(data):
    """Calculate comprehensive intraday trading indicators"""
    df = data.copy()
    
    if len(df) < 14:
        return df
    
    try:
        # Price and Returns
        df['Returns'] = df['Close'].pct_change()
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # EMAs for intraday (faster moving averages)
        df['EMA_5'] = EMAIndicator(df['Close'], window=5).ema_indicator()
        df['EMA_9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = EMAIndicator(df['Close'], window=21).ema_indicator()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['RSI_Signal'] = df['RSI'].apply(lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral')
        
        # Stochastic Oscillator (Great for intraday)
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # MACD for momentum
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        df['MACD_Crossover'] = (df['MACD'] > df['MACD_Signal']).astype(int).diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR for volatility and stop loss calculation
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        
        # VWAP (Critical for intraday)
        df['VWAP'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df['Price_vs_VWAP'] = ((df['Close'] - df['VWAP']) / df['VWAP'] * 100)
        
        # Money Flow Index (Volume-weighted RSI)
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
        
        # OBV
        df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        
        # Volume Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Signal'] = df['Volume_Ratio'].apply(lambda x: 'High' if x > 1.5 else 'Low' if x < 0.7 else 'Normal')
        
        # ADX for trend strength
        if len(df) >= 25:
            adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX'] = adx.adx()
            df['Trend_Strength'] = df['ADX'].apply(lambda x: 'Strong' if x > 25 else 'Weak')
        
        # Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Price Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Support and Resistance (intraday levels)
        df['Intraday_High'] = df['High'].expanding().max()
        df['Intraday_Low'] = df['Low'].expanding().min()
        
        # Pivot Points (Critical for intraday)
        if len(df) > 0:
            pivot = (df['High'].iloc[0] + df['Low'].iloc[0] + df['Close'].iloc[0]) / 3
            df['Pivot'] = pivot
            df['R1'] = 2 * pivot - df['Low'].iloc[0]
            df['S1'] = 2 * pivot - df['High'].iloc[0]
            df['R2'] = pivot + (df['High'].iloc[0] - df['Low'].iloc[0])
            df['S2'] = pivot - (df['High'].iloc[0] - df['Low'].iloc[0])
        
        return df.ffill().bfill()
    
    except Exception as e:
        st.warning(f"Indicator calculation error: {str(e)}")
        return df

def generate_intraday_signals(data):
    """Generate precise intraday trading signals with entry/exit points"""
    
    if len(data) < 14:
        return {
            'Signal': 'WAIT',
            'Strength': 0,
            'Entry_Price': None,
            'Stop_Loss': None,
            'Target_1': None,
            'Target_2': None,
            'Confidence': 0,
            'Reasons': ['Insufficient data'],
            'Risk_Reward': 'N/A'
        }
    
    current_price = data['Close'].iloc[-1]
    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
    current_macd = data['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in data.columns else 0
    atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.01
    
    # Scoring system
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # 1. EMA Crossover Strategy (Weight: 3)
    if 'EMA_5' in data.columns and 'EMA_21' in data.columns:
        ema5 = data['EMA_5'].iloc[-1]
        ema21 = data['EMA_21'].iloc[-1]
        ema5_prev = data['EMA_5'].iloc[-2]
        ema21_prev = data['EMA_21'].iloc[-2]
        
        if ema5 > ema21 and ema5_prev <= ema21_prev:
            buy_score += 3
            reasons.append("🔥 Bullish EMA Crossover (5 crossed above 21)")
        elif ema5 < ema21 and ema5_prev >= ema21_prev:
            sell_score += 3
            reasons.append("🔥 Bearish EMA Crossover (5 crossed below 21)")
        elif ema5 > ema21:
            buy_score += 1
            reasons.append("📈 Price above EMA 21")
        elif ema5 < ema21:
            sell_score += 1
            reasons.append("📉 Price below EMA 21")
    
    # 2. RSI Strategy (Weight: 2)
    if current_rsi < 30:
        buy_score += 2
        reasons.append(f"💪 RSI Oversold ({current_rsi:.1f})")
    elif current_rsi > 70:
        sell_score += 2
        reasons.append(f"⚠️ RSI Overbought ({current_rsi:.1f})")
    elif current_rsi < 40:
        buy_score += 1
        reasons.append(f"📊 RSI Bullish ({current_rsi:.1f})")
    elif current_rsi > 60:
        sell_score += 1
        reasons.append(f"📊 RSI Bearish ({current_rsi:.1f})")
    
    # 3. MACD Strategy (Weight: 2)
    if 'MACD_Crossover' in data.columns:
        if data['MACD_Crossover'].iloc[-1] > 0:
            buy_score += 2
            reasons.append("⚡ MACD Bullish Crossover")
        elif data['MACD_Crossover'].iloc[-1] < 0:
            sell_score += 2
            reasons.append("⚡ MACD Bearish Crossover")
        elif current_macd > 0:
            buy_score += 1
        elif current_macd < 0:
            sell_score += 1
    
    # 4. VWAP Strategy (Weight: 2)
    if 'VWAP' in data.columns:
        vwap = data['VWAP'].iloc[-1]
        if current_price > vwap * 1.002:
            buy_score += 2
            reasons.append(f"✅ Price above VWAP (${vwap:.2f})")
        elif current_price < vwap * 0.998:
            sell_score += 2
            reasons.append(f"❌ Price below VWAP (${vwap:.2f})")
    
    # 5. Bollinger Bands (Weight: 2)
    if 'BB_Position' in data.columns:
        bb_pos = data['BB_Position'].iloc[-1]
        if bb_pos < 0.2:
            buy_score += 2
            reasons.append("🎯 Price near lower Bollinger Band")
        elif bb_pos > 0.8:
            sell_score += 2
            reasons.append("🎯 Price near upper Bollinger Band")
    
    # 6. Volume Confirmation (Weight: 2)
    if 'Volume_Ratio' in data.columns:
        vol_ratio = data['Volume_Ratio'].iloc[-1]
        if vol_ratio > 1.5:
            if buy_score > sell_score:
                buy_score += 2
                reasons.append("📊 High volume confirming bullish move")
            elif sell_score > buy_score:
                sell_score += 2
                reasons.append("📊 High volume confirming bearish move")
    
    # 7. Stochastic Oscillator (Weight: 1)
    if 'Stoch_K' in data.columns:
        stoch_k = data['Stoch_K'].iloc[-1]
        if stoch_k < 20:
            buy_score += 1
            reasons.append(f"📉 Stochastic Oversold ({stoch_k:.1f})")
        elif stoch_k > 80:
            sell_score += 1
            reasons.append(f"📈 Stochastic Overbought ({stoch_k:.1f})")
    
    # 8. Money Flow Index (Weight: 1)
    if 'MFI' in data.columns:
        mfi = data['MFI'].iloc[-1]
        if mfi < 20:
            buy_score += 1
            reasons.append(f"💰 Money Flow Oversold ({mfi:.1f})")
        elif mfi > 80:
            sell_score += 1
            reasons.append(f"💰 Money Flow Overbought ({mfi:.1f})")
    
    # Determine signal
    total_score = buy_score + sell_score
    confidence = min((abs(buy_score - sell_score) / max(total_score, 1)) * 100, 99)
    
    # Calculate Stop Loss and Targets based on ATR
    stop_loss_distance = atr * 1.5
    target1_distance = atr * 2
    target2_distance = atr * 3
    
    if buy_score >= sell_score + 3:  # Strong buy
        signal = 'STRONG BUY'
        entry_price = current_price
        stop_loss = entry_price - stop_loss_distance
        target_1 = entry_price + target1_distance
        target_2 = entry_price + target2_distance
        strength = min(buy_score, 10)
    elif buy_score > sell_score:  # Moderate buy
        signal = 'BUY'
        entry_price = current_price
        stop_loss = entry_price - stop_loss_distance
        target_1 = entry_price + target1_distance
        target_2 = entry_price + target2_distance
        strength = min(buy_score, 10)
    elif sell_score >= buy_score + 3:  # Strong sell
        signal = 'STRONG SELL'
        entry_price = current_price
        stop_loss = entry_price + stop_loss_distance
        target_1 = entry_price - target1_distance
        target_2 = entry_price - target2_distance
        strength = min(sell_score, 10)
    elif sell_score > buy_score:  # Moderate sell
        signal = 'SELL'
        entry_price = current_price
        stop_loss = entry_price + stop_loss_distance
        target_1 = entry_price - target1_distance
        target_2 = entry_price - target2_distance
        strength = min(sell_score, 10)
    else:
        signal = 'HOLD'
        entry_price = current_price
        stop_loss = None
        target_1 = None
        target_2 = None
        strength = 0
        reasons.append("⚖️ No clear directional bias")
    
    # Calculate Risk-Reward Ratio
    if stop_loss and target_1:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_1 - entry_price)
        risk_reward = f"1:{reward/risk:.2f}" if risk > 0 else "N/A"
    else:
        risk_reward = "N/A"
    
    return {
        'Signal': signal,
        'Strength': strength,
        'Entry_Price': entry_price,
        'Stop_Loss': stop_loss,
        'Target_1': target_1,
        'Target_2': target_2,
        'Confidence': confidence,
        'Reasons': reasons,
        'Risk_Reward': risk_reward,
        'Buy_Score': buy_score,
        'Sell_Score': sell_score
    }

def build_intraday_lstm_model(input_shape):
    """Enhanced Bidirectional LSTM for intraday prediction"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model

def predict_next_price_movement(data, minutes_ahead=15):
    """Predict price movement for next 15 minutes with multiple predictions"""
    try:
        if len(data) < 30:
            return None, None, None, None
        
        # Prepare features
        feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'VWAP']
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return None, None, None, None
        
        feature_data = data[available_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Use last 30 data points for better accuracy
        lookback = min(30, len(scaled_data) - 1)
        
        # Quick model training
        model = build_intraday_lstm_model((lookback, len(available_cols)))
        
        # Prepare training data
        X_train, y_train = [], []
        for i in range(lookback, len(scaled_data)):
            X_train.append(scaled_data[i-lookback:i])
            y_train.append(scaled_data[i, 0])
        
        if len(X_train) < 10:
            return None, None, None, None
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Quick training with more epochs for better prediction
        model.fit(X_train, y_train, epochs=15, batch_size=8, verbose=0, validation_split=0.1)
        
        # Multi-step prediction for next 15 minutes
        current_sequence = scaled_data[-lookback:].copy()
        predictions = []
        
        # Predict next 15 steps (minutes)
        for step in range(minutes_ahead):
            X = current_sequence.reshape(1, lookback, len(available_cols))
            pred_scaled = model.predict(X, verbose=0)
            
            # Create new data point
            new_point = current_sequence[-1].copy()
            new_point[0] = pred_scaled[0, 0]  # Update close price
            
            # Append and shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            # Inverse transform to get actual price
            dummy = np.zeros((1, len(available_cols)))
            dummy[0, 0] = pred_scaled[0, 0]
            predicted_price = scaler.inverse_transform(dummy)[0, 0]
            predictions.append(predicted_price)
        
        # Get final prediction (15 minutes ahead)
        final_prediction = predictions[-1]
        
        # Get mid-point prediction (7-8 minutes ahead)
        mid_prediction = predictions[len(predictions)//2]
        
        # Calculate confidence based on prediction stability
        prediction_volatility = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0.1
        confidence = max(0.5, min(0.95, 1 - prediction_volatility * 5))
        
        return final_prediction, mid_prediction, confidence, predictions
        
    except Exception as e:
        st.warning(f"Prediction error: {str(e)}")
        return None, None, None, None

# Streamlit UI
st.set_page_config(page_title="Intraday Trading AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 class='main-header'>🚀 INTRADAY TRADING AI PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Real-Time Buy/Sell Signals for Same-Day Trading</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Trading Configuration")
symbol = st.sidebar.text_input("Stock Symbol", value="RELIANCE.NS").upper()
interval = st.sidebar.selectbox("Time Interval", ["1m", "2m", "5m", "15m", "30m"], index=2)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)

st.sidebar.markdown("---")
st.sidebar.header("📚 Quick Guide")
st.sidebar.info("""
**Intraday Trading Tips:**
- Trade only during market hours
- Always use stop-loss orders
- Don't risk more than 2% per trade
- Follow the signals but verify
- Exit all positions before market close

**Signal Strength:**
- 🟢 STRONG BUY: 7-10 score
- 🟢 BUY: 4-6 score  
- 🔴 SELL: 4-6 score
- 🔴 STRONG SELL: 7-10 score
- 🟡 HOLD: Wait for better signal

**15-Minute Prediction:**
- AI analyzes patterns to predict next 15 mins
- Shows minute-by-minute price trajectory
- High confidence (>70%) = Strong signal
- Use with technical signals for best results
""")

# Auto-refresh logic
if auto_refresh:
    st.sidebar.info("🔄 Auto-refreshing... Next update in 60 seconds")
    import time
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 60:
        st.rerun()
        st.session_state.last_refresh = time.time()

# Main content
with st.spinner('📊 Fetching real-time market data...'):
    data = fetch_intraday_data(symbol, interval)

if data is not None and not data.empty:
    # Calculate indicators
    data_with_indicators = calculate_intraday_indicators(data)
    
    # Generate signals
    signals = generate_intraday_signals(data_with_indicators)
    
    # Predict next price with error handling
    try:
        prediction_result = predict_next_price_movement(data_with_indicators, minutes_ahead=15)
        
        if prediction_result and prediction_result[0] is not None:
            predicted_price_15min, predicted_price_7min, pred_confidence, all_predictions = prediction_result
        else:
            predicted_price_15min = None
            predicted_price_7min = None
            pred_confidence = None
            all_predictions = None
    except Exception as e:
        st.warning(f"AI prediction temporarily unavailable: {str(e)}")
        predicted_price_15min = None
        predicted_price_7min = None
        pred_confidence = None
        all_predictions = None
    
    current_price = data['Close'].iloc[-1]
    current_time = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    
    # Display last update time
    st.markdown(f"<p style='text-align: center; color: #666;'>📅 Last Update: {current_time} | 💰 Current Price: ${current_price:.2f}</p>", unsafe_allow_html=True)
    
    # Main signal display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        signal_class = 'buy-signal' if 'BUY' in signals['Signal'] else 'sell-signal' if 'SELL' in signals['Signal'] else 'hold-signal'
        signal_emoji = '🚀' if 'BUY' in signals['Signal'] else '📉' if 'SELL' in signals['Signal'] else '⏸️'
        
        st.markdown(f"""
        <div class='intraday-signal-box {signal_class}'>
            <div>{signal_emoji} {signals['Signal']} {signal_emoji}</div>
            <div style='font-size: 1.2rem; margin-top: 10px;'>Confidence: {signals['Confidence']:.1f}% | Strength: {signals['Strength']}/10</div>
            <div style='font-size: 1rem; margin-top: 5px;'>Risk-Reward: {signals['Risk_Reward']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Entry/Exit levels
    st.markdown("---")
    st.subheader("🎯 Entry & Exit Levels")
    
    col_entry1, col_entry2, col_entry3, col_entry4 = st.columns(4)
    
    with col_entry1:
        st.markdown("<div class='entry-exit-box'>", unsafe_allow_html=True)
        st.metric("Entry Price", f"${signals['Entry_Price']:.2f}" if signals['Entry_Price'] else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_entry2:
        st.markdown("<div class='stoploss-box'>", unsafe_allow_html=True)
        st.metric("Stop Loss", f"${signals['Stop_Loss']:.2f}" if signals['Stop_Loss'] else "N/A")
        if signals['Stop_Loss']:
            sl_pct = abs((signals['Stop_Loss'] - current_price) / current_price * 100)
            st.write(f"Risk: {sl_pct:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_entry3:
        st.markdown("<div class='target-box'>", unsafe_allow_html=True)
        st.metric("Target 1", f"${signals['Target_1']:.2f}" if signals['Target_1'] else "N/A")
        if signals['Target_1']:
            t1_pct = abs((signals['Target_1'] - current_price) / current_price * 100)
            st.write(f"Gain: {t1_pct:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_entry4:
        st.markdown("<div class='target-box'>", unsafe_allow_html=True)
        st.metric("Target 2", f"${signals['Target_2']:.2f}" if signals['Target_2'] else "N/A")
        if signals['Target_2']:
            t2_pct = abs((signals['Target_2'] - current_price) / current_price * 100)
            st.write(f"Gain: {t2_pct:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # AI Prediction Section - Enhanced for 15 minutes
    if predicted_price_15min and pred_confidence and all_predictions:
        st.markdown("---")
        st.subheader("🤖 AI Price Prediction - Next 15 Minutes")
        
        # Create prediction timeline
        pred_change_7min = ((predicted_price_7min - current_price) / current_price) * 100
        pred_change_15min = ((predicted_price_15min - current_price) / current_price) * 100
        
        # Display prediction metrics
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
        
        with col_pred1:
            st.metric("Current Price", f"${current_price:.2f}", "Now")
        
        with col_pred2:
            st.metric(
                "7-Min Prediction", 
                f"${predicted_price_7min:.2f}", 
                f"{pred_change_7min:+.2f}%",
                delta_color="normal" if pred_change_7min > 0 else "inverse"
            )
        
        with col_pred3:
            st.metric(
                "15-Min Prediction", 
                f"${predicted_price_15min:.2f}", 
                f"{pred_change_15min:+.2f}%",
                delta_color="normal" if pred_change_15min > 0 else "inverse"
            )
        
        with col_pred4:
            st.metric("AI Confidence", f"{pred_confidence:.1%}")
        
        # Prediction trend analysis
        if pred_change_15min > 2:
            st.success(f"🚀 **Strong Bullish Prediction**: Expected to rise {pred_change_15min:.2f}% in next 15 minutes")
        elif pred_change_15min > 0.5:
            st.info(f"📈 **Bullish Prediction**: Expected to rise {pred_change_15min:.2f}% in next 15 minutes")
        elif pred_change_15min < -2:
            st.error(f"📉 **Strong Bearish Prediction**: Expected to fall {pred_change_15min:.2f}% in next 15 minutes")
        elif pred_change_15min < -0.5:
            st.warning(f"🔻 **Bearish Prediction**: Expected to fall {pred_change_15min:.2f}% in next 15 minutes")
        else:
            st.info(f"➡️ **Neutral Prediction**: Expected to move {pred_change_15min:+.2f}% in next 15 minutes")
        
        # Create prediction timeline chart
        st.subheader("📊 15-Minute Price Prediction Timeline")
        
        # Generate time labels
        current_time_obj = data.index[-1]
        time_labels = [current_time_obj + timedelta(minutes=i) for i in range(len(all_predictions) + 1)]
        
        # Create figure for prediction
        fig_pred = go.Figure()
        
        # Historical price (last 30 minutes)
        hist_prices = data['Close'].iloc[-30:].values if len(data) >= 30 else data['Close'].values
        hist_times = data.index[-30:] if len(data) >= 30 else data.index
        
        fig_pred.add_trace(go.Scatter(
            x=hist_times,
            y=hist_prices,
            name='Historical Price',
            line=dict(color='blue', width=2),
            mode='lines+markers'
        ))
        
        # Predicted prices
        all_pred_prices = [current_price] + all_predictions
        fig_pred.add_trace(go.Scatter(
            x=time_labels,
            y=all_pred_prices,
            name='AI Prediction',
            line=dict(color='red', width=2, dash='dash'),
            mode='lines+markers'
        ))
        
        # Add confidence band
        std_dev = np.std(all_predictions) if len(all_predictions) > 1 else current_price * 0.005
        upper_band = [p + std_dev for p in all_pred_prices]
        lower_band = [p - std_dev for p in all_pred_prices]
        
        fig_pred.add_trace(go.Scatter(
            x=time_labels + time_labels[::-1],
            y=upper_band + lower_band[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Band',
            showlegend=True
        ))
        
        # Mark key prediction points
        fig_pred.add_trace(go.Scatter(
            x=[time_labels[7], time_labels[-1]],
            y=[predicted_price_7min, predicted_price_15min],
            mode='markers+text',
            marker=dict(size=12, color=['orange', 'red']),
            text=['7-Min', '15-Min'],
            textposition='top center',
            name='Key Predictions',
            showlegend=False
        ))
        
        fig_pred.update_layout(
            title='Next 15 Minutes - AI Price Prediction',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction statistics
        st.subheader("📈 Prediction Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            max_pred = max(all_predictions)
            max_gain = ((max_pred - current_price) / current_price) * 100
            st.metric("Expected High", f"${max_pred:.2f}", f"+{max_gain:.2f}%")
        
        with col_stat2:
            min_pred = min(all_predictions)
            max_loss = ((min_pred - current_price) / current_price) * 100
            st.metric("Expected Low", f"${min_pred:.2f}", f"{max_loss:.2f}%")
        
        with col_stat3:
            avg_pred = np.mean(all_predictions)
            avg_change = ((avg_pred - current_price) / current_price) * 100
            st.metric("Average Prediction", f"${avg_pred:.2f}", f"{avg_change:+.2f}%")
        
        with col_stat4:
            pred_volatility = np.std(all_predictions) / current_price * 100
            st.metric("Predicted Volatility", f"{pred_volatility:.2f}%")
        
        # Trading suggestion based on 15-min prediction
        st.subheader("💡 15-Minute Trading Strategy")
        
        if pred_change_15min > 1 and pred_confidence > 0.7:
            st.success(f"""
            **🚀 HIGH PROBABILITY LONG SETUP**
            
            Based on AI prediction showing {pred_change_15min:.2f}% upside in next 15 minutes:
            
            1. ✅ **Entry**: Buy at current price ${current_price:.2f}
            2. 🎯 **Target (7-min)**: ${predicted_price_7min:.2f} (Quick scalp: {pred_change_7min:.2f}%)
            3. 🎯 **Target (15-min)**: ${predicted_price_15min:.2f} (Full target: {pred_change_15min:.2f}%)
            4. 🛡️ **Stop Loss**: ${current_price * 0.995:.2f} (-0.5% safety)
            5. ⏰ **Time Frame**: Exit within 15 minutes or at target
            
            **Confidence Level**: {pred_confidence:.0%} | **Risk-Reward**: Favorable
            """)
        
        elif pred_change_15min < -1 and pred_confidence > 0.7:
            st.error(f"""
            **📉 HIGH PROBABILITY SHORT SETUP**
            
            Based on AI prediction showing {pred_change_15min:.2f}% downside in next 15 minutes:
            
            1. ⚠️ **Entry**: Short at current price ${current_price:.2f}
            2. 🎯 **Target (7-min)**: ${predicted_price_7min:.2f} (Quick profit: {abs(pred_change_7min):.2f}%)
            3. 🎯 **Target (15-min)**: ${predicted_price_15min:.2f} (Full target: {abs(pred_change_15min):.2f}%)
            4. 🛡️ **Stop Loss**: ${current_price * 1.005:.2f} (+0.5% safety)
            5. ⏰ **Time Frame**: Exit within 15 minutes or at target
            
            **Confidence Level**: {pred_confidence:.0%} | **Risk-Reward**: Favorable
            """)
        
        elif abs(pred_change_15min) > 0.5:
            direction = "upward" if pred_change_15min > 0 else "downward"
            st.info(f"""
            **📊 MODERATE PROBABILITY SETUP**
            
            AI predicts {direction} move of {abs(pred_change_15min):.2f}% in next 15 minutes:
            
            - **Confidence**: {pred_confidence:.0%} (Moderate)
            - **Suggestion**: Wait for confirmation from technical signals
            - **Action**: Consider smaller position size if trading
            - **Alternative**: Wait for stronger setup
            """)
        
        else:
            st.warning("""
            **⏸️ LOW PROBABILITY - AVOID TRADING**
            
            AI prediction shows minimal price movement in next 15 minutes:
            
            - 💡 **Best Action**: Stay out and wait for clearer opportunity
            - ⏰ **Next Check**: Monitor after 15 minutes for new signals
            - 📊 **Alternative**: Look for other stocks with stronger signals
            """)
        
        # Minute-by-minute prediction table
        with st.expander("📋 Detailed Minute-by-Minute Predictions"):
            pred_df = pd.DataFrame({
                'Minute': range(1, 16),
                'Predicted Price': [f"${p:.2f}" for p in all_predictions],
                'Change from Now': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in all_predictions]
            })
            st.dataframe(pred_df, use_container_width=True)
    
    # Signal reasons
    st.markdown("---")
    st.subheader("📋 Signal Analysis")
    
    col_reasons1, col_reasons2 = st.columns(2)
    
    with col_reasons1:
        st.write("**📊 Technical Indicators Supporting Signal:**")
        for reason in signals['Reasons']:
            st.write(f"✓ {reason}")
    
    with col_reasons2:
        st.write("**📈 Signal Breakdown:**")
        st.write(f"• Bullish Score: {signals['Buy_Score']}/10")
        st.write(f"• Bearish Score: {signals['Sell_Score']}/10")
        st.write(f"• Net Score: {signals['Buy_Score'] - signals['Sell_Score']:+d}")
        
        if signals['Signal'] != 'HOLD':
            st.success(f"💡 **Action**: {signals['Signal']} at ${signals['Entry_Price']:.2f}")
            if signals['Stop_Loss']:
                st.error(f"🛡️ **Stop Loss**: ${signals['Stop_Loss']:.2f}")
            if signals['Target_1']:
                st.info(f"🎯 **Take Profit 1**: ${signals['Target_1']:.2f}")
            if signals['Target_2']:
                st.info(f"🎯 **Take Profit 2**: ${signals['Target_2']:.2f}")
    
    # Advanced charts
    st.markdown("---")
    st.subheader("📊 Real-Time Technical Charts")
    
    # Create comprehensive chart
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=('Price Action with Entry/Exit Levels', 'Volume Profile', 'RSI & Stochastic', 'MACD'),
        vertical_spacing=0.05,
        row_heights=[0.45, 0.20, 0.20, 0.15]
    )
    
    # 1. Candlestick chart with indicators
    fig.add_trace(go.Candlestick(
        x=data_with_indicators.index,
        open=data_with_indicators['Open'],
        high=data_with_indicators['High'],
        low=data_with_indicators['Low'],
        close=data_with_indicators['Close'],
        name='Price',
        showlegend=False
    ), row=1, col=1)
    
    # Add EMAs
    if 'EMA_5' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['EMA_5'],
            name='EMA 5',
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)
    
    if 'EMA_9' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['EMA_9'],
            name='EMA 9',
            line=dict(color='blue', width=1.5)
        ), row=1, col=1)
    
    if 'EMA_21' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['EMA_21'],
            name='EMA 21',
            line=dict(color='purple', width=1.5)
        ), row=1, col=1)
    
    # Add VWAP
    if 'VWAP' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['VWAP'],
            name='VWAP',
            line=dict(color='brown', width=2, dash='dash')
        ), row=1, col=1)
    
    # Add Bollinger Bands
    if 'BB_Upper' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ), row=1, col=1)
    
    # Add Entry/Stop Loss/Target levels
    if signals['Entry_Price']:
        fig.add_hline(y=signals['Entry_Price'], line_dash="solid", line_color="blue", 
                     annotation_text=f"Entry: ${signals['Entry_Price']:.2f}", row=1, col=1)
    
    if signals['Stop_Loss']:
        fig.add_hline(y=signals['Stop_Loss'], line_dash="dash", line_color="red", 
                     annotation_text=f"Stop Loss: ${signals['Stop_Loss']:.2f}", row=1, col=1)
    
    if signals['Target_1']:
        fig.add_hline(y=signals['Target_1'], line_dash="dash", line_color="green", 
                     annotation_text=f"Target 1: ${signals['Target_1']:.2f}", row=1, col=1)
    
    if signals['Target_2']:
        fig.add_hline(y=signals['Target_2'], line_dash="dot", line_color="darkgreen", 
                     annotation_text=f"Target 2: ${signals['Target_2']:.2f}", row=1, col=1)
    
    # 2. Volume chart
    colors = ['red' if data_with_indicators['Close'].iloc[i] < data_with_indicators['Open'].iloc[i] 
             else 'green' for i in range(len(data_with_indicators))]
    
    fig.add_trace(go.Bar(
        x=data_with_indicators.index,
        y=data_with_indicators['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    if 'Volume_SMA' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['Volume_SMA'],
            name='Volume SMA',
            line=dict(color='orange', width=2)
        ), row=2, col=1)
    
    # 3. RSI and Stochastic
    if 'RSI' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    if 'Stoch_K' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['Stoch_K'],
            name='Stochastic K',
            line=dict(color='blue', width=1.5)
        ), row=3, col=1)
    
    if 'Stoch_D' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['Stoch_D'],
            name='Stochastic D',
            line=dict(color='red', width=1.5, dash='dash')
        ), row=3, col=1)
    
    # 4. MACD
    if 'MACD' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=4, col=1)
    
    if 'MACD_Signal' in data_with_indicators.columns:
        fig.add_trace(go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD_Signal'],
            name='Signal',
            line=dict(color='red', width=2)
        ), row=4, col=1)
    
    if 'MACD_Histogram' in data_with_indicators.columns:
        colors_macd = ['green' if val > 0 else 'red' for val in data_with_indicators['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD_Histogram'],
            name='Histogram',
            marker_color=colors_macd
        ), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        title=f"{symbol} - Intraday Analysis ({interval} interval)",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market Summary
    st.markdown("---")
    st.subheader("📊 Market Summary & Key Levels")
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        open_price = data['Open'].iloc[0]
        day_change = ((current_price - open_price) / open_price) * 100
        st.metric("Day's Change", f"{day_change:+.2f}%", f"${current_price - open_price:+.2f}")
    
    with col_sum2:
        day_high = data['High'].max()
        day_low = data['Low'].min()
        st.metric("Day Range", f"${day_high:.2f} - ${day_low:.2f}")
        range_pct = ((day_high - day_low) / day_low) * 100
        st.write(f"Range: {range_pct:.2f}%")
    
    with col_sum3:
        total_volume = data['Volume'].sum()
        avg_volume = data['Volume'].mean()
        st.metric("Total Volume", f"{total_volume:,.0f}")
        st.write(f"Avg: {avg_volume:,.0f}")
    
    with col_sum4:
        if 'ATR' in data_with_indicators.columns:
            atr_val = data_with_indicators['ATR'].iloc[-1]
            atr_pct = (atr_val / current_price) * 100
            st.metric("ATR (Volatility)", f"${atr_val:.2f}")
            st.write(f"{atr_pct:.2f}% of price")
    
    # Pivot Points Table
    st.markdown("---")
    st.subheader("🎯 Key Intraday Levels (Pivot Points)")
    
    if 'Pivot' in data_with_indicators.columns:
        col_pivot1, col_pivot2, col_pivot3 = st.columns(3)
        
        with col_pivot1:
            st.markdown("**Resistance Levels**")
            if 'R2' in data_with_indicators.columns:
                r2 = data_with_indicators['R2'].iloc[0]
                st.write(f"🔴 R2: ${r2:.2f}")
            if 'R1' in data_with_indicators.columns:
                r1 = data_with_indicators['R1'].iloc[0]
                st.write(f"🟠 R1: ${r1:.2f}")
        
        with col_pivot2:
            st.markdown("**Pivot Point**")
            pivot = data_with_indicators['Pivot'].iloc[0]
            st.write(f"⚪ Pivot: ${pivot:.2f}")
            if current_price > pivot:
                st.success("Price above pivot (Bullish)")
            else:
                st.error("Price below pivot (Bearish)")
        
        with col_pivot3:
            st.markdown("**Support Levels**")
            if 'S1' in data_with_indicators.columns:
                s1 = data_with_indicators['S1'].iloc[0]
                st.write(f"🟢 S1: ${s1:.2f}")
            if 'S2' in data_with_indicators.columns:
                s2 = data_with_indicators['S2'].iloc[0]
                st.write(f"🔵 S2: ${s2:.2f}")
    
    # Trading Tips based on current signal
    st.markdown("---")
    st.subheader("💡 Trading Recommendations")
    
    if signals['Signal'] == 'STRONG BUY':
        st.success("""
        **🚀 STRONG BUY Signal Detected!**
        
        **Action Plan:**
        1. ✅ Enter LONG position at current price: $""" + f"{signals['Entry_Price']:.2f}" + """
        2. 🛡️ Place Stop Loss at: $""" + f"{signals['Stop_Loss']:.2f}" + """
        3. 🎯 Take partial profit at Target 1: $""" + f"{signals['Target_1']:.2f}" + """
        4. 🎯 Trail stop loss to Target 2: $""" + f"{signals['Target_2']:.2f}" + """
        5. ⚠️ Exit all positions before market close
        
        **Risk Management:** Risk only 1-2% of your capital on this trade.
        """)
    
    elif signals['Signal'] == 'BUY':
        st.success("""
        **📈 BUY Signal Detected**
        
        **Action Plan:**
        1. ✅ Consider LONG position at: $""" + f"{signals['Entry_Price']:.2f}" + """
        2. 🛡️ Mandatory Stop Loss: $""" + f"{signals['Stop_Loss']:.2f}" + """
        3. 🎯 First Target: $""" + f"{signals['Target_1']:.2f}" + """
        4. 📊 Monitor price action closely
        
        **Risk Management:** Use smaller position size due to moderate signal strength.
        """)
    
    elif signals['Signal'] == 'STRONG SELL':
        st.error("""
        **📉 STRONG SELL Signal Detected!**
        
        **Action Plan:**
        1. ⚠️ Enter SHORT position at: $""" + f"{signals['Entry_Price']:.2f}" + """
        2. 🛡️ Place Stop Loss at: $""" + f"{signals['Stop_Loss']:.2f}" + """
        3. 🎯 Take profit at Target 1: $""" + f"{signals['Target_1']:.2f}" + """
        4. 🎯 Trail stop for Target 2: $""" + f"{signals['Target_2']:.2f}" + """
        5. ⚠️ Exit all positions before market close
        
        **Risk Management:** Risk only 1-2% of your capital on this trade.
        """)
    
    elif signals['Signal'] == 'SELL':
        st.error("""
        **🔻 SELL Signal Detected**
        
        **Action Plan:**
        1. ⚠️ Consider SHORT position at: $""" + f"{signals['Entry_Price']:.2f}" + """
        2. 🛡️ Mandatory Stop Loss: $""" + f"{signals['Stop_Loss']:.2f}" + """
        3. 🎯 First Target: $""" + f"{signals['Target_1']:.2f}" + """
        4. 📊 Watch for reversal signals
        
        **Risk Management:** Use smaller position size due to moderate signal strength.
        """)
    
    else:
        st.warning("""
        **⏸️ HOLD - No Clear Signal**
        
        **Action Plan:**
        1. ⏰ Wait for clearer directional bias
        2. 📊 Monitor for signal strength to increase
        3. 🔍 Look for breakout or breakdown
        4. 💡 Consider staying in cash
        
        **Remember:** Not trading is also a position. Wait for high-probability setups.
        """)
    
    # Risk Warning
    st.markdown("---")
    st.error("""
    ⚠️ **CRITICAL RISK DISCLAIMER - INTRADAY TRADING**
    
    **Before You Trade:**
    - ✋ This is AI-generated analysis for EDUCATIONAL purposes only
    - 📉 Intraday trading carries EXTREMELY HIGH RISK of loss
    - 💰 Never risk more than 1-2% of your capital per trade
    - 🛡️ ALWAYS use stop-loss orders - No exceptions!
    - ⏰ Square off all positions before market close
    - 📚 Paper trade first to test strategies
    - 🧠 Trading psychology is as important as technical analysis
    - ⚖️ Past performance does not guarantee future results
    
    **This is NOT financial advice. Consult a qualified financial advisor before trading.**
    """)
    
    # Performance Stats
    st.markdown("---")
    st.subheader("📈 Today's Performance Stats")
    
    col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)
    
    with col_perf1:
        bullish_candles = sum(1 for i in range(len(data)) if data['Close'].iloc[i] > data['Open'].iloc[i])
        st.metric("Bullish Candles", f"{bullish_candles}/{len(data)}")
    
    with col_perf2:
        bearish_candles = sum(1 for i in range(len(data)) if data['Close'].iloc[i] < data['Open'].iloc[i])
        st.metric("Bearish Candles", f"{bearish_candles}/{len(data)}")
    
    with col_perf3:
        if 'Returns' in data_with_indicators.columns:
            avg_return = data_with_indicators['Returns'].mean() * 100
            st.metric("Avg Return/Candle", f"{avg_return:.3f}%")
    
    with col_perf4:
        if 'Returns' in data_with_indicators.columns:
            volatility = data_with_indicators['Returns'].std() * 100
            st.metric("Volatility", f"{volatility:.3f}%")
    
    with col_perf5:
        momentum = "📈 Bullish" if bullish_candles > bearish_candles else "📉 Bearish" if bearish_candles > bullish_candles else "⚖️ Neutral"
        st.metric("Momentum", momentum)
    
    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Create summary DataFrame
        summary_data = {
            'Timestamp': [current_time],
            'Symbol': [symbol],
            'Current_Price': [current_price],
            'Signal': [signals['Signal']],
            'Confidence': [f"{signals['Confidence']:.1f}%"],
            'Entry_Price': [signals['Entry_Price']],
            'Stop_Loss': [signals['Stop_Loss']],
            'Target_1': [signals['Target_1']],
            'Target_2': [signals['Target_2']],
            'Risk_Reward': [signals['Risk_Reward']]
        }
        summary_df = pd.DataFrame(summary_data)
        
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Signal Summary (CSV)",
            data=csv,
            file_name=f"{symbol}_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_exp2:
        # Full data export
        csv_full = data_with_indicators.to_csv()
        st.download_button(
            label="📥 Download Full Data with Indicators (CSV)",
            data=csv_full,
            file_name=f"{symbol}_fulldata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.error("""
    ❌ **Unable to fetch intraday data**
    
    **Possible Reasons:**
    1. 🕐 Market is currently closed
    2. ❌ Invalid stock symbol
    3. 🌐 Internet connection issues
    4. 🚫 Symbol doesn't support selected interval
    
    **Please verify:**
    - Market hours (9:15 AM - 3:30 PM IST for Indian markets)
    - Stock symbol format (.NS for NSE, .BO for BSE)
    - Use valid symbols like: RELIANCE.NS, TCS.NS, AAPL, TSLA
    """)
    
    st.info("""
    **📚 Popular Intraday Trading Symbols:**
    
    **Indian Stocks (NSE):**
    - RELIANCE.NS - Reliance Industries
    - TCS.NS - Tata Consultancy Services
    - INFY.NS - Infosys
    - HDFCBANK.NS - HDFC Bank
    - SBIN.NS - State Bank of India
    - TATASTEEL.NS - Tata Steel
    - ITC.NS - ITC Limited
    
    **US Stocks:**
    - AAPL - Apple Inc.
    - TSLA - Tesla Inc.
    - MSFT - Microsoft
    - GOOGL - Alphabet
    - AMZN - Amazon
    - NVDA - NVIDIA
    - META - Meta Platforms
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🤖 Intraday Trading AI Predictor v2.0</strong></p>
    <p>Built with Advanced Machine Learning & Real-Time Technical Analysis</p>
    <p><em>For educational purposes only. Trade at your own risk.</em></p>
</div>
""", unsafe_allow_html=True)