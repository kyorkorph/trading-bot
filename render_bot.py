#!/usr/bin/env python3
"""
Render Cloud Trading Bot
Optimized for Render.com deployment
"""

import os
import requests
import json
import time
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

# Telegram Configuration
BOT_TOKEN = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
CHAT_ID = "8401873453"

# Flask app for Render
app = Flask(__name__)

def send_telegram_alert(message):
    """Send alert to Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

class RenderTradingSystem:
    def __init__(self):
        self.initial_capital = 100000
        self.symbol = "GC=F"  # Gold Futures
        self.interval = "15m"
        self.period = "60d"
        self.performance_file = "render_performance.json"
        self.ml_model = None
        self.scaler = None
        
    def load_performance(self):
        """Load trading performance data"""
        try:
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0,
                "current_balance": self.initial_capital,
                "daily_pnl": 0,
                "last_trade_date": None,
                "signals_today": 0,
                "missed_signals": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "trade_history": [],
                "ml_predictions": 0,
                "ml_accuracy": 0
            }
    
    def save_performance(self, performance):
        """Save trading performance data"""
        with open(self.performance_file, 'w') as f:
            json.dump(performance, f, indent=2)
    
    def get_price_data_safe(self, data):
        """Get price data safely without pandas issues"""
        try:
            data_list = []
            for index, row in data.iterrows():
                data_list.append({
                    'Close': float(row['Close']),
                    'Volume': float(row['Volume']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'timestamp': index
                })
            return data_list
        except Exception as e:
            print(f"❌ Error getting price data: {e}")
            return None
    
    def calculate_leading_indicators(self, data_list, i):
        """Calculate leading indicators"""
        if i < 20:
            return None
            
        try:
            # Volume analysis
            recent_volumes = [data_list[j]['Volume'] for j in range(i-5, i)]
            volume_ma = np.mean(recent_volumes)
            current_volume = data_list[i]['Volume']
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
            
            # Price momentum
            recent_prices = [data_list[j]['Close'] for j in range(i-5, i)]
            price_momentum = (data_list[i]['Close'] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            
            # Price acceleration
            if i >= 10:
                prev_momentum = (recent_prices[0] - data_list[i-10]['Close']) / data_list[i-10]['Close'] if data_list[i-10]['Close'] > 0 else 0
                price_acceleration = price_momentum - prev_momentum
            else:
                price_acceleration = 0
            
            # Volatility momentum
            highs = [data_list[j]['High'] for j in range(i-5, i)]
            lows = [data_list[j]['Low'] for j in range(i-5, i)]
            volatility = np.mean([h - l for h, l in zip(highs, lows)])
            volatility_momentum = volatility / np.mean([data_list[j]['Close'] for j in range(i-5, i)]) if np.mean([data_list[j]['Close'] for j in range(i-5, i)]) > 0 else 0
            
            # VWAP approximation
            typical_prices = [(data_list[j]['High'] + data_list[j]['Low'] + data_list[j]['Close']) / 3 for j in range(i-20, i)]
            vwap = np.mean(typical_prices)
            price_vwap_ratio = data_list[i]['Close'] / vwap if vwap > 0 else 1
            
            # Money Flow Index approximation
            positive_flow = sum([max(0, data_list[j]['Close'] - data_list[j-1]['Close']) for j in range(i-14, i)])
            negative_flow = sum([max(0, data_list[j-1]['Close'] - data_list[j]['Close']) for j in range(i-14, i)])
            mfi = 100 - (100 / (1 + positive_flow / negative_flow)) if negative_flow > 0 else 50
            
            # Williams %R approximation
            highest_14 = max([data_list[j]['High'] for j in range(i-14, i)])
            lowest_14 = min([data_list[j]['Low'] for j in range(i-14, i)])
            williams_r = ((highest_14 - data_list[i]['Close']) / (highest_14 - lowest_14)) * -100 if (highest_14 - lowest_14) > 0 else -50
            
            # Stochastic RSI approximation
            rsi_values = []
            for k in range(i-14, i):
                gains = sum([max(0, data_list[j]['Close'] - data_list[j-1]['Close']) for j in range(k-13, k+1)])
                losses = sum([max(0, data_list[j-1]['Close'] - data_list[j]['Close']) for j in range(k-13, k+1)])
                rs = gains / losses if losses > 0 else 1
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
            
            if len(rsi_values) >= 14:
                stoch_rsi = ((data_list[i]['Close'] - min(rsi_values)) / (max(rsi_values) - min(rsi_values))) * 100 if (max(rsi_values) - min(rsi_values)) > 0 else 50
            else:
                stoch_rsi = 50
            
            return {
                'Volume_Ratio': volume_ratio,
                'Volume_Momentum': (current_volume - volume_ma) / volume_ma if volume_ma > 0 else 0,
                'Price_Momentum': price_momentum,
                'Price_Acceleration': price_acceleration,
                'Volatility_Momentum': volatility_momentum,
                'Price_VWAP_Ratio': price_vwap_ratio,
                'MFI': mfi,
                'Williams_R': williams_r,
                'Stochastic_RSI': stoch_rsi
            }
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return None
    
    def train_ml_model(self, data_list):
        """Train ML model on historical data"""
        try:
            if len(data_list) < 50:
                return False
                
            features = []
            targets = []
            
            for i in range(20, len(data_list) - 1):
                indicators = self.calculate_leading_indicators(data_list, i)
                if indicators:
                    feature_vector = [
                        indicators['Volume_Ratio'],
                        indicators['Price_Momentum'],
                        indicators['MFI'],
                        indicators['Williams_R'],
                        indicators['Stochastic_RSI']
                    ]
                    features.append(feature_vector)
                    
                    # Target: next period return
                    current_price = data_list[i]['Close']
                    next_price = data_list[i + 1]['Close']
                    target = (next_price - current_price) / current_price if current_price > 0 else 0
                    targets.append(target)
            
            if len(features) < 10:
                return False
                
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            self.ml_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.ml_model.fit(features_scaled, targets)
            
            return True
            
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
            return False
    
    def get_ml_prediction(self, indicators):
        """Get ML prediction"""
        try:
            if self.ml_model is None or self.scaler is None:
                return 0
                
            feature_vector = [
                indicators['Volume_Ratio'],
                indicators['Price_Momentum'],
                indicators['MFI'],
                indicators['Williams_R'],
                indicators['Stochastic_RSI']
            ]
            
            features_scaled = self.scaler.transform([feature_vector])
            prediction = self.ml_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error getting ML prediction: {e}")
            return 0
    
    def generate_signal(self, indicators, ml_prediction):
        """Generate trading signal"""
        try:
            # Leading indicators analysis
            volume_signal = 1 if indicators['Volume_Ratio'] > 1.2 else (-1 if indicators['Volume_Ratio'] < 0.8 else 0)
            momentum_signal = 1 if indicators['Price_Momentum'] > 0.01 else (-1 if indicators['Price_Momentum'] < -0.01 else 0)
            mfi_signal = 1 if indicators['MFI'] < 20 else (-1 if indicators['MFI'] > 80 else 0)
            williams_signal = 1 if indicators['Williams_R'] < -80 else (-1 if indicators['Williams_R'] > -20 else 0)
            stoch_signal = 1 if indicators['Stochastic_RSI'] < 20 else (-1 if indicators['Stochastic_RSI'] > 80 else 0)
            
            # ML confidence
            ml_confidence = 1 if ml_prediction > 0.005 else (-1 if ml_prediction < -0.005 else 0)
            
            # Combined signal
            total_score = volume_signal + momentum_signal + mfi_signal + williams_signal + stoch_signal + ml_confidence
            
            if total_score >= 3:
                return "BUY", total_score
            elif total_score <= -3:
                return "SELL", total_score
            else:
                return "HOLD", total_score
                
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
            return "HOLD", 0
    
    def calculate_position_size(self, price, confidence, capital, ml_prediction):
        """Calculate position size based on risk"""
        try:
            # Base risk: 2% of capital
            risk_amount = capital * 0.02
            
            # Adjust for confidence
            confidence_multiplier = min(abs(confidence) / 6, 1.5)
            
            # Adjust for ML prediction
            ml_multiplier = 1 + abs(ml_prediction) * 10
            
            # Final position size
            position_size = risk_amount * confidence_multiplier * ml_multiplier
            
            # Limit to 5% of capital
            max_position = capital * 0.05
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return capital * 0.02
    
    def send_signal_alert(self, signal_data, performance):
        """Send signal alert to Telegram"""
        try:
            signal_type = signal_data['signal']
            confidence = signal_data['confidence']
            price = signal_data['price']
            position_size = signal_data['position_size']
            ml_prediction = signal_data['ml_prediction']
            
            # Performance metrics
            balance = performance['current_balance']
            total_trades = performance['total_trades']
            win_rate = (performance['winning_trades'] / performance['total_trades'] * 100) if performance['total_trades'] > 0 else 0
            
            alert = f"""
🎯 <b>TRADING SIGNAL DETECTED</b>

📈 <b>Signal:</b> {signal_type}
💪 <b>Confidence:</b> {confidence}/6
💰 <b>Price:</b> ${price:,.2f}
📊 <b>Position Size:</b> ${position_size:,.2f}
🤖 <b>ML Prediction:</b> {ml_prediction:.4f}

📊 <b>Performance:</b>
• Balance: ${balance:,.2f}
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
☁️ <b>System:</b> Cloud (Render)
🌐 <b>Location:</b> Render Cloud
"""
            
            send_telegram_alert(alert)
            
        except Exception as e:
            print(f"❌ Error sending signal alert: {e}")
    
    def run_render_system(self):
        """Run the Render trading system"""
        try:
            print("🚀 Starting Render Trading System...")
            send_telegram_alert("🚀 <b>Render Trading System Started!</b>\n\n☁️ Running in Render cloud\n📱 Commands from your phone\n💰 Trading Gold Futures")
            
            # Load performance
            performance = self.load_performance()
            
            # Get market data
            print("📊 Getting market data...")
            data = yf.download(self.symbol, period=self.period, interval=self.interval)
            
            if data.empty:
                send_telegram_alert("❌ <b>Error:</b> Could not get market data")
                return False
            
            # Convert to safe format
            data_list = self.get_price_data_safe(data)
            if not data_list:
                send_telegram_alert("❌ <b>Error:</b> Could not process market data")
                return False
            
            print(f"✅ Got {len(data_list)} data points")
            
            # Train ML model
            print("🤖 Training ML model...")
            if self.train_ml_model(data_list):
                print("✅ ML model trained successfully")
                performance['ml_predictions'] += 1
            else:
                print("⚠️ ML model training failed")
            
            # Analyze latest data
            if len(data_list) >= 20:
                latest_indicators = self.calculate_leading_indicators(data_list, len(data_list) - 1)
                
                if latest_indicators:
                    # Get ML prediction
                    ml_prediction = self.get_ml_prediction(latest_indicators)
                    
                    # Generate signal
                    signal, confidence = self.generate_signal(latest_indicators, ml_prediction)
                    
                    if signal != "HOLD":
                        # Calculate position size
                        current_price = data_list[-1]['Close']
                        position_size = self.calculate_position_size(current_price, confidence, performance['current_balance'], ml_prediction)
                        
                        # Send signal alert
                        signal_data = {
                            'signal': signal,
                            'confidence': confidence,
                            'price': current_price,
                            'position_size': position_size,
                            'ml_prediction': ml_prediction
                        }
                        
                        self.send_signal_alert(signal_data, performance)
                        
                        # Update performance
                        performance['signals_today'] += 1
                        performance['total_trades'] += 1
                        performance['last_trade_date'] = datetime.now().isoformat()
                        
                        print(f"✅ Signal sent: {signal} (confidence: {confidence})")
                    else:
                        print("⏳ No signal detected")
                        
                        # Send status update
                        status_msg = f"""
📊 <b>Market Analysis Complete</b>

📈 <b>Current Price:</b> ${data_list[-1]['Close']:,.2f}
📊 <b>Volume Ratio:</b> {latest_indicators['Volume_Ratio']:.2f}
💪 <b>Price Momentum:</b> {latest_indicators['Price_Momentum']:.4f}
🤖 <b>ML Prediction:</b> {ml_prediction:.4f}
📊 <b>Signal:</b> HOLD

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
☁️ <b>System:</b> Cloud (Render)
🌐 <b>Location:</b> Render Cloud
"""
                        send_telegram_alert(status_msg)
            
            # Save performance
            self.save_performance(performance)
            
            print("✅ Render trading system completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"❌ <b>Render System Error:</b> {str(e)}"
            send_telegram_alert(error_msg)
            print(f"❌ Error in Render system: {e}")
            return False

# Flask routes for Render
@app.route('/')
def health_check():
    return "Trading Bot is running on Render!"

@app.route('/run', methods=['POST'])
def run_trading():
    system = RenderTradingSystem()
    success = system.run_render_system()
    return jsonify({"success": success})

@app.route('/status')
def status():
    return jsonify({
        "status": "running",
        "system": "Render Cloud",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    # Send startup message
    send_telegram_alert("☁️ <b>Render Trading Bot Started!</b>\n\n🌐 Running on Render cloud\n📱 Ready for commands\n💰 24/7 operation")
    
    # Run Flask app for Render
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
