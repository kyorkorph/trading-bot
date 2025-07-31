#!/usr/bin/env python3

import requests
import json

def setup_bot_commands():
    """Set up bot commands in Telegram"""
    bot_token = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
    base_url = f"https://api.telegram.org/bot{bot_token}"
    
    commands = [
        {"command": "start", "description": "Start the trading bot"},
        {"command": "run", "description": "Run trading system now"},
        {"command": "status", "description": "Check system status"},
        {"command": "performance", "description": "Get performance summary"},
        {"command": "summary", "description": "Get daily trading summary"},
        {"command": "balance", "description": "Check current balance"},
        {"command": "signals", "description": "Get recent signals"},
        {"command": "help", "description": "Show all commands"},
        {"command": "auto_on", "description": "Enable daily automation"},
        {"command": "auto_off", "description": "Disable daily automation"},
        {"command": "logs", "description": "Get recent logs"}
    ]
    
    url = f"{base_url}/setMyCommands"
    data = {"commands": json.dumps(commands)}
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("✅ Bot commands set up successfully!")
            print("📱 You can now use these commands in Telegram:")
            for cmd in commands:
                print(f"   /{cmd['command']} - {cmd['description']}")
        else:
            print(f"❌ Failed to set commands: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error setting commands: {e}")

if __name__ == "__main__":
    setup_bot_commands() 