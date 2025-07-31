#!/usr/bin/env python3
"""
Persistent Telegram Bot Launcher
Keeps the trading bot running even when laptop is closed
"""

import subprocess
import time
import os
import signal
import sys
from datetime import datetime

def send_telegram_message(message):
    """Send a message to Telegram"""
    import requests
    
    BOT_TOKEN = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
    CHAT_ID = "8345130003"
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def check_bot_running():
    """Check if the bot is currently running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'interactive_telegram_bot.py' in result.stdout
    except:
        return False

def start_bot_in_screen():
    """Start the bot in a screen session for persistence"""
    try:
        # Kill any existing screen session
        subprocess.run(['screen', '-S', 'trading_bot', '-X', 'quit'], 
                      capture_output=True)
        
        # Start new screen session with the bot
        cmd = [
            'screen', '-dmS', 'trading_bot',
            'python3', 'interactive_telegram_bot.py'
        ]
        
        subprocess.run(cmd, check=True)
        
        # Wait a moment for the bot to start
        time.sleep(3)
        
        if check_bot_running():
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error starting bot in screen: {e}")
        return False

def main():
    """Main persistent launcher"""
    print("🤖 PERSISTENT BOT LAUNCHER")
    print("=" * 50)
    
    # Send startup notification
    startup_msg = f"""
🚀 <b>PERSISTENT BOT LAUNCHER STARTED</b>

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💻 Status: Running on laptop
🔄 Auto-restart: Enabled
📱 Commands: /run, /status, /performance

<i>The bot will keep running even when you close your laptop!</i>
"""
    
    send_telegram_message(startup_msg)
    print("✅ Startup notification sent to Telegram")
    
    # Start the bot in screen session
    if start_bot_in_screen():
        print("✅ Bot started successfully in screen session")
        send_telegram_message("✅ <b>Bot is now running persistently!</b>\n\nYou can close your laptop and the bot will keep working!")
    else:
        print("❌ Failed to start bot")
        send_telegram_message("❌ <b>Failed to start persistent bot</b>\n\nPlease check the laptop and restart.")
        return
    
    # Monitor and restart if needed
    while True:
        try:
            if not check_bot_running():
                print(f"⚠️ Bot stopped at {datetime.now()}, restarting...")
                send_telegram_message("⚠️ <b>Bot stopped, restarting...</b>")
                
                if start_bot_in_screen():
                    print("✅ Bot restarted successfully")
                    send_telegram_message("✅ <b>Bot restarted successfully!</b>")
                else:
                    print("❌ Failed to restart bot")
                    send_telegram_message("❌ <b>Failed to restart bot</b>\n\nPlease check the laptop.")
            
            # Check every 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down persistent launcher...")
            send_telegram_message("🛑 <b>Persistent launcher shutting down</b>")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main() 