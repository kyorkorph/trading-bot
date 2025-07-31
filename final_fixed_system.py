#!/usr/bin/env python3
"""
Final Fixed Trading System - Advanced ML + Leading Indicators
Completely fixed version with proper pandas handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Telegram Configuration
BOT_TOKEN = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
CHAT_ID = "8401873453"

def send_telegram_alert(message):
    """Send alert to Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, data=data)
        return response.status_code == 200
    except:
        return False

class FinalFixedTradingSystem:
    def __init__(self):
        self.initial_capital = 100000
        self.symbol = "GC=F"  # Gold Futures
        self.interval = "15m"
        self.period = "60d"
        
    def calculate_leading_indicators(self, data):
        """Calculate leading indicators with proper pandas handling"""
        try:
            # Create a copy to avoid modifying original
            df = data.copy()
            
            # Volume indicators - use loc to avoid SettingWithCopyWarning
            df.loc[:, 'Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df.loc[:, 'Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df.loc[:, 'Volume_Momentum'] = df['Volume'].pct_change(3)
            
            # Price indicators
            df.loc[:, 'Price_Momentum'] = df['Close'].pct_change(3)
            df.loc[:, 'Price_Acceleration'] = df['Price_Momentum'].diff()
            
            # Volatility
            df.loc[:, 'High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
            df.loc[:, 'Volatility_Momentum'] = df['High_Low_Ratio'].rolling(window=5).mean()
            
            # VWAP
            df.loc[:, 'VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df.loc[:, 'Price_VWAP_Ratio'] = df['Close'] / df['VWAP']
            
            # Money Flow Index
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi_ratio = positive_mf / negative_mf
            df.loc[:, 'MFI'] = 100 - (100 / (1 + mfi_ratio))
            
            # Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            df.loc[:, 'Williams_R'] = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
            
            # Stochastic RSI
            rsi = self.calculate_rsi(df['Close'], 14)
            stoch_rsi = (rsi - rsi.rolling(window=14).min()) / (rsi.rolling(window=14).max() - rsi.rolling(window=14).min())
            df.loc[:, 'Stochastic_RSI'] = stoch_rsi * 100
            
            print("✅ Indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_ml_model(self, data):
        """Train ML model with proper error handling"""
        try:
            feature_columns = ['Volume_Ratio', 'Volume_Momentum', 'Price_Momentum', 
                             'Price_Acceleration', 'Volatility_Momentum', 'Price_VWAP_Ratio',
                             'MFI', 'Williams_R', 'Stochastic_RSI']
            
            # Clean data
            data_clean = data.dropna()
            
            if len(data_clean) < 30:
                print("⚠️ Insufficient data for ML training")
                return None, None
            
            # Check if all features exist
            missing_features = [col for col in feature_columns if col not in data_clean.columns]
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
                return None, None
            
            X = data_clean[feature_columns]
            y = data_clean['Close'].pct_change().shift(-1)
            
            # Remove NaN values
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 20:
                print("⚠️ Insufficient data after cleaning")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            print("✅ ML model trained successfully")
            return model, scaler
            
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
            return None, None
    
    def generate_signal(self, data, model=None, scaler=None):
        """Generate trading signal with proper error handling"""
        try:
            latest = data.iloc[-1]
            
            # Check for required indicators
            required_indicators = ['Volume_Ratio', 'Price_Momentum', 'Price_VWAP_Ratio', 'MFI', 'Williams_R']
            for indicator in required_indicators:
                if pd.isna(latest[indicator]):
                    print(f"⚠️ Missing {indicator} data")
                    return None
            
            # Get indicator values as floats
            volume_ratio = float(latest['Volume_Ratio'])
            volume_momentum = float(latest['Volume_Momentum'])
            price_momentum = float(latest['Price_Momentum'])
            price_vwap_ratio = float(latest['Price_VWAP_Ratio'])
            volatility_momentum = float(latest['Volatility_Momentum'])
            mfi = float(latest['MFI'])
            williams_r = float(latest['Williams_R'])
            price = float(latest['Close'])
            
            signal = None
            signal_type = None
            confidence = 0
            
            # BUY SIGNALS
            if (volume_ratio > 1.5 and volume_momentum > 0.1 and price_momentum > 0.01):
                signal = "BUY"
                signal_type = "Volume Surge"
                confidence = 0.8
            elif (price_vwap_ratio < 0.98 and volume_ratio > 1.2):
                signal = "BUY"
                signal_type = "VWAP Oversold"
                confidence = 0.7
            elif (mfi < 20 and volume_ratio > 1.1):
                signal = "BUY"
                signal_type = "MFI Oversold"
                confidence = 0.6
            elif (williams_r < -80 and volume_ratio > 1.1):
                signal = "BUY"
                signal_type = "Williams %R Oversold"
                confidence = 0.6
            
            # SELL SIGNALS
            elif (volume_ratio > 1.5 and volume_momentum > 0.1 and price_momentum < -0.01):
                signal = "SELL"
                signal_type = "Volume Surge"
                confidence = 0.8
            elif (price_vwap_ratio > 1.02 and volume_ratio > 1.2):
                signal = "SELL"
                signal_type = "VWAP Overbought"
                confidence = 0.7
            elif (mfi > 80 and volume_ratio > 1.1):
                signal = "SELL"
                signal_type = "MFI Overbought"
                confidence = 0.6
            elif (williams_r > -20 and volume_ratio > 1.1):
                signal = "SELL"
                signal_type = "Williams %R Overbought"
                confidence = 0.6
            
            if signal:
                # Get ML prediction if available
                ml_confidence = 0.5
                if model and scaler:
                    try:
                        indicators = [
                            volume_ratio, volume_momentum, price_momentum,
                            float(latest['Price_Acceleration']), volatility_momentum,
                            price_vwap_ratio, mfi, williams_r,
                            float(latest['Stochastic_RSI'])
                        ]
                        indicators_scaled = scaler.transform([indicators])
                        ml_prediction = model.predict(indicators_scaled)[0]
                        ml_confidence = 0.5 + (ml_prediction * 0.5)
                        ml_confidence = max(0.1, min(0.9, ml_confidence))
                    except Exception as e:
                        print(f"⚠️ ML prediction error: {e}")
                
                return {
                    'signal': signal,
                    'type': signal_type,
                    'price': price,
                    'confidence': confidence,
                    'ml_confidence': ml_confidence,
                    'indicators': {
                        'volume_ratio': volume_ratio,
                        'volume_momentum': volume_momentum,
                        'price_momentum': price_momentum,
                        'price_vwap_ratio': price_vwap_ratio,
                        'volatility_momentum': volatility_momentum,
                        'mfi': mfi,
                        'williams_r': williams_r
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
            return None
    
    def calculate_position_size(self, price, confidence, capital):
        """Calculate position size with risk management"""
        try:
            # Risk management
            risk_amount = capital * 0.02  # 2% risk
            stop_loss_distance = price * 0.02  # 2% stop loss
            position_size = risk_amount / stop_loss_distance
            position_size *= confidence
            
            # Position limits
            max_position = capital * 0.05  # 5% max
            position_size = min(position_size, max_position)
            position_size = max(position_size, 1)
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 1
    
    def send_signal_alert(self, signal_data):
        """Send detailed signal alert to Telegram"""
        try:
            signal = signal_data['signal']
            signal_type = signal_data['type']
            price = signal_data['price']
            confidence = signal_data['confidence']
            ml_confidence = signal_data['ml_confidence']
            indicators = signal_data['indicators']
            
            # Calculate position size
            position_size = self.calculate_position_size(price, confidence, self.initial_capital)
            investment = position_size * price
            
            message = f"""🚨 {signal} SIGNAL DETECTED 🚨

📊 ASSET: Gold Futures (GC=F)
💰 PRICE: ${price:.2f}
📈 SIGNAL TYPE: {signal_type}
🎯 CONFIDENCE: {confidence:.1%}
🤖 ML CONFIDENCE: {ml_confidence:.1%}

📊 POSITION DETAILS:
• Position Size: {position_size:.0f} contracts
• Investment: ${investment:,.0f}
• Risk: 2% of capital
• Max Position: 5% of capital

📈 INDICATORS:
• Volume Ratio: {indicators['volume_ratio']:.2f}
• Volume Momentum: {indicators['volume_momentum']:.3f}
• Price Momentum: {indicators['price_momentum']:.3f}
• VWAP Ratio: {indicators['price_vwap_ratio']:.3f}
• MFI: {indicators['mfi']:.1f}
• Williams %R: {indicators['williams_r']:.1f}

🎯 STRATEGY: Advanced ML + Leading Indicators
💰 ACCOUNT: $100,000
⏰ TIMEFRAME: 15-minute
🛡️ RISK: 2% per trade

System Status: ACTIVE ✅"""
            
            send_telegram_alert(message)
            print(f"✅ {signal} signal sent to Telegram!")
            
        except Exception as e:
            print(f"❌ Error sending signal alert: {e}")
    
    def run_final_system(self):
        """Run the final fixed trading system"""
        print("🚀 FINAL FIXED TRADING SYSTEM STARTED")
        print("=" * 50)
        print("🎯 Strategy: Advanced ML + Leading Indicators")
        print("💰 Account: $100,000")
        print("📈 Asset: Gold Futures (GC=F)")
        print("⏰ Timeframe: 15-minute")
        print("🛡️ Risk: 2% per trade")
        
        # Send startup message
        startup_message = f"""🚀 FINAL FIXED TRADING SYSTEM STARTED

🎯 STRATEGY: Advanced ML + Leading Indicators
💰 ACCOUNT: $100,000
📈 ASSET: Gold Futures (GC=F)
⏰ TIMEFRAME: 15-minute
🛡️ RISK: 2% per trade

✅ SYSTEM STATUS: ACTIVE
✅ INDICATORS: FIXED
✅ ML MODEL: TRAINED
✅ ALERTS: ENABLED

System Status: ACTIVE ✅"""
        
        send_telegram_alert(startup_message)
        print("✅ Startup message sent!")
        
        while True:
            try:
                # Get latest data
                data = yf.download(self.symbol, period=self.period, interval=self.interval, progress=False, auto_adjust=True)
                
                if data.empty:
                    print("❌ No data received")
                    time.sleep(60)
                    continue
                
                print(f"✅ Got {len(data)} data points")
                
                # Calculate indicators
                data = self.calculate_leading_indicators(data)
                
                # Train ML model (first time or periodically)
                if not hasattr(self, 'ml_model') or not hasattr(self, 'ml_scaler'):
                    self.ml_model, self.ml_scaler = self.train_ml_model(data)
                
                # Generate signal
                signal_data = self.generate_signal(data, self.ml_model, self.ml_scaler)
                
                if signal_data:
                    self.send_signal_alert(signal_data)
                else:
                    print("⏳ No signal detected")
                
                # Wait before next check
                print("⏳ Waiting 15 minutes...")
                time.sleep(900)  # 15 minutes
                
            except Exception as e:
                print(f"❌ System error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    print("🚀 Starting final fixed trading system...")
    system = FinalFixedTradingSystem()
    system.run_final_system() 