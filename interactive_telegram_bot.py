#!/usr/bin/env python3
"""
Interactive Telegram Trading Bot
Commands to run system, check performance, get summaries
"""

import requests
import json
import time
import subprocess
import os
from datetime import datetime
import threading

class InteractiveTradingBot:
    def __init__(self):
        self.bot_token = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
        self.chat_id = "8401873453"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.last_update_id = 0
        self.running = True
        
    def setup_bot_commands(self):
        """Set up bot commands"""
        commands = [
            {"command": "start", "description": "Start the trading bot"},
            {"command": "run", "description": "Run trading system now"},
            {"command": "status", "description": "Check system status"},
            {"command": "performance", "description": "Get performance summary"},
            {"command": "summary", "description": "Get daily trading summary"},
            {"command": "balance", "description": "Check current balance"},
            {"command": "signals", "description": "Get recent signals"},
            {"command": "stop", "description": "Stop the bot"},
            {"command": "help", "description": "Show all commands"},
            {"command": "auto_on", "description": "Enable daily automation"},
            {"command": "auto_off", "description": "Disable daily automation"},
            {"command": "logs", "description": "Get recent logs"}
        ]
        
        url = f"{self.base_url}/setMyCommands"
        data = {"commands": json.dumps(commands)}
        
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ Bot commands set up successfully!")
            else:
                print(f"❌ Failed to set commands: {response.status_code}")
        except Exception as e:
            print(f"❌ Error setting commands: {e}")
    
    def send_message(self, text):
        """Send message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return False
    
    def run_trading_system(self):
        """Run the trading system"""
        try:
            # Change to trading directory
            os.chdir("/Users/edisonlin/Documents/Quant Trading")
            
            # Run the system
            result = subprocess.run(["python3", "ultimate_trading_system.py"], 
                                  capture_output=True, text=True, timeout=300)
            
            # Log the result
            with open("bot_run_log.txt", "a") as f:
                f.write(f"\n=== {datetime.now()} BOT RUN ===\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Output: {result.stdout}\n")
                f.write(f"Errors: {result.stderr}\n")
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "System timed out after 5 minutes", ""
        except Exception as e:
            return False, "", str(e)
    
    def get_performance_summary(self):
        """Get performance summary from saved data"""
        try:
            if os.path.exists("ultimate_performance.json"):
                with open("ultimate_performance.json", "r") as f:
                    data = json.load(f)
                
                balance = data.get("current_balance", 100000)
                total_trades = data.get("total_trades", 0)
                winning_trades = data.get("winning_trades", 0)
                total_pnl = data.get("total_pnl", 0)
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                summary = f"""📊 PERFORMANCE SUMMARY

💰 Current Balance: ${balance:,.2f}
📈 Total P&L: ${total_pnl:,.2f}
📊 Total Trades: {total_trades}
✅ Winning Trades: {winning_trades}
📈 Win Rate: {win_rate:.1f}%
📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

System Status: ACTIVE ✅"""
                
                return summary
            else:
                return "📊 No performance data available yet. Run the system first!"
                
        except Exception as e:
            return f"❌ Error getting performance: {e}"
    
    def get_system_status(self):
        """Check if system is running"""
        try:
            # Check if process is running
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            is_running = "ultimate_trading_system.py" in result.stdout
            
            # Check last log
            last_log = "No recent logs"
            if os.path.exists("trading_log.txt"):
                with open("trading_log.txt", "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_log = lines[-1].strip()
            
            status = f"""🤖 SYSTEM STATUS

🟢 Running: {'YES' if is_running else 'NO'}
📅 Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📝 Last Log: {last_log[:100]}...

Commands Available:
/run - Run system now
/performance - Check performance
/summary - Get daily summary
/balance - Check balance
/signals - Recent signals
/auto_on - Enable automation
/auto_off - Disable automation
/logs - View logs

System Status: {'🟢 ACTIVE' if is_running else '🔴 INACTIVE'}"""
            
            return status
            
        except Exception as e:
            return f"❌ Error checking status: {e}"
    
    def get_recent_signals(self):
        """Get recent trading signals"""
        try:
            if os.path.exists("ultimate_performance.json"):
                with open("ultimate_performance.json", "r") as f:
                    data = json.load(f)
                
                trades = data.get("paper_trades", [])
                recent_trades = trades[-5:] if len(trades) > 5 else trades
                
                if recent_trades:
                    signals = "📊 RECENT SIGNALS\n\n"
                    for trade in recent_trades:
                        signal_type = trade.get("signal_type", "UNKNOWN")
                        timestamp = trade.get("timestamp", "UNKNOWN")
                        pnl = trade.get("pnl", 0)
                        status = "✅ WIN" if pnl > 0 else "❌ LOSS"
                        
                        signals += f"🕐 {timestamp}\n"
                        signals += f"📈 {signal_type}\n"
                        signals += f"💰 P&L: ${pnl:,.2f}\n"
                        signals += f"📊 {status}\n\n"
                    
                    return signals
                else:
                    return "📊 No recent signals found. Run the system to generate signals!"
            else:
                return "📊 No trading data available. Run the system first!"
                
        except Exception as e:
            return f"❌ Error getting signals: {e}"
    
    def handle_command(self, command):
        """Handle bot commands"""
        command = command.lower().strip()
        
        if command == "/start":
            return """🤖 TRADING BOT STARTED!

Welcome to your interactive trading system!

Available commands:
/run - Run trading system now
/status - Check system status  
/performance - Get performance summary
/summary - Get daily summary
/balance - Check current balance
/signals - Get recent signals
/auto_on - Enable daily automation
/auto_off - Disable daily automation
/logs - View recent logs
/help - Show this help

System Status: ACTIVE ✅"""
        
        elif command == "/run":
            self.send_message("🚀 Starting trading system...")
            success, stdout, stderr = self.run_trading_system()
            
            if success:
                return "✅ Trading system completed successfully!\n\nCheck /performance for results."
            else:
                return f"❌ Trading system failed:\n\n{stderr[:500]}"
        
        elif command == "/status":
            return self.get_system_status()
        
        elif command == "/performance":
            return self.get_performance_summary()
        
        elif command == "/summary":
            return self.get_performance_summary()  # Same as performance for now
        
        elif command == "/balance":
            try:
                if os.path.exists("ultimate_performance.json"):
                    with open("ultimate_performance.json", "r") as f:
                        data = json.load(f)
                    balance = data.get("current_balance", 100000)
                    return f"💰 Current Balance: ${balance:,.2f}"
                else:
                    return "💰 No balance data available. Run the system first!"
            except Exception as e:
                return f"❌ Error getting balance: {e}"
        
        elif command == "/signals":
            return self.get_recent_signals()
        
        elif command == "/auto_on":
            # Set up cron job
            try:
                cron_cmd = '0 20 * * * cd "/Users/edisonlin/Documents/Quant Trading" && python3 ultimate_trading_system.py >> trading_log.txt 2>&1'
                os.system(f'(crontab -l 2>/dev/null; echo "{cron_cmd}") | crontab -')
                return "✅ Daily automation enabled! System will run at 8 PM daily."
            except Exception as e:
                return f"❌ Error enabling automation: {e}"
        
        elif command == "/auto_off":
            # Remove cron job
            try:
                os.system("crontab -l | grep -v 'ultimate_trading_system.py' | crontab -")
                return "✅ Daily automation disabled!"
            except Exception as e:
                return f"❌ Error disabling automation: {e}"
        
        elif command == "/logs":
            try:
                if os.path.exists("trading_log.txt"):
                    with open("trading_log.txt", "r") as f:
                        lines = f.readlines()
                        recent_logs = lines[-10:] if len(lines) > 10 else lines
                        logs = "".join(recent_logs)
                        return f"📝 Recent Logs:\n\n{logs[-1000:]}"  # Limit to 1000 chars
                else:
                    return "📝 No logs available yet."
            except Exception as e:
                return f"❌ Error getting logs: {e}"
        
        elif command == "/help":
            return """🤖 TRADING BOT HELP

Commands:
/start - Start the bot
/run - Run trading system now
/status - Check system status
/performance - Get performance summary
/summary - Get daily summary
/balance - Check current balance
/signals - Get recent signals
/auto_on - Enable daily automation
/auto_off - Disable daily automation
/logs - View recent logs
/help - Show this help

System Status: ACTIVE ✅"""
        
        else:
            return "❓ Unknown command. Type /help for available commands."
    
    def get_updates(self):
        """Get updates from Telegram"""
        url = f"{self.base_url}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 30}
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data["ok"] and data["result"]:
                    for update in data["result"]:
                        self.last_update_id = update["update_id"]
                        if "message" in update:
                            message = update["message"]
                            if "text" in message:
                                text = message["text"]
                                response_text = self.handle_command(text)
                                self.send_message(response_text)
        except Exception as e:
            print(f"❌ Error getting updates: {e}")
    
    def run_bot(self):
        """Run the interactive bot"""
        print("🤖 Starting interactive trading bot...")
        self.setup_bot_commands()
        self.send_message("🤖 Interactive Trading Bot Started!\n\nType /help for commands.")
        
        while self.running:
            try:
                self.get_updates()
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                self.running = False
            except Exception as e:
                print(f"❌ Bot error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = InteractiveTradingBot()
    bot.run_bot() 