#!/bin/bash

echo "🤖 Starting Persistent Trading Bot..."
echo "=================================="

# Kill any existing processes
pkill -f "interactive_telegram_bot.py"
pkill -f "persistent_bot_launcher.py"

# Start the persistent launcher
nohup python3 persistent_bot_launcher.py > persistent_bot.log 2>&1 &

echo "✅ Persistent bot launcher started!"
echo "📱 You can now close your laptop and the bot will keep running"
echo "📊 Check the log: tail -f persistent_bot.log"
echo "🛑 To stop: pkill -f persistent_bot_launcher.py" 