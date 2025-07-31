#!/usr/bin/env python3
"""
GitHub Setup Guide for Railway Deployment
Simple steps to deploy to Railway for free
"""

import requests
import subprocess
import os

def send_telegram_message(message):
    """Send message to Telegram"""
    BOT_TOKEN = "8345130003:AAHHVWGjawOv0eJ7N1jH6fpxaV4HelDEHNk"
    CHAT_ID = "8401873453"
    
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

def check_git_installed():
    """Check if git is installed"""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def setup_git_repo():
    """Set up git repository"""
    try:
        # Initialize git repo
        subprocess.run(['git', 'init'], check=True)
        print("✅ Git repository initialized")
        
        # Add all files
        subprocess.run(['git', 'add', '.'], check=True)
        print("✅ Files added to git")
        
        # Initial commit
        subprocess.run(['git', 'commit', '-m', 'Initial Railway deployment setup'], check=True)
        print("✅ Initial commit created")
        
        return True
        
    except Exception as e:
        print(f"❌ Git setup failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 GITHUB SETUP GUIDE")
    print("=" * 50)
    
    # Check git installation
    if not check_git_installed():
        print("❌ Git not installed. Please install git first.")
        return
    
    # Set up git repo
    if setup_git_repo():
        print("✅ Git repository set up successfully!")
        
        # Send setup guide
        guide = """
🚀 <b>GITHUB SETUP COMPLETE!</b>

Your local git repository is ready!

<b>📋 NEXT STEPS:</b>

1️⃣ <b>Create GitHub Repository</b>
   • Go to https://github.com
   • Click "New repository"
   • Name it "trading-bot"
   • Make it PUBLIC
   • Don't initialize with README

2️⃣ <b>Push to GitHub</b>
   • Copy the commands GitHub shows you
   • Run them in your terminal:
   ```
   git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
   git branch -M main
   git push -u origin main
   ```

3️⃣ <b>Deploy to Railway</b>
   • Go to https://railway.app
   • Sign up with GitHub
   • Click "New Project"
   • Select "Deploy from GitHub repo"
   • Choose your trading-bot repo
   • Railway will auto-deploy!

4️⃣ <b>Test Your Bot</b>
   • Railway will give you a URL
   • Your bot runs 24/7 in cloud
   • Send /run in Telegram
   • Bot responds from cloud!

<b>💡 BENEFITS:</b>
✅ <b>Completely FREE</b>
✅ <b>24/7 operation</b>
✅ <b>Phone-only control</b>
✅ <b>No laptop required</b>
✅ <b>Professional reliability</b>

<b>Need help with any step?</b>
"""
        
        send_telegram_message(guide)
        print("✅ GitHub setup guide sent to Telegram!")
        
        print("\n🎯 READY FOR DEPLOYMENT:")
        print("1. Create GitHub repo")
        print("2. Push code to GitHub") 
        print("3. Deploy to Railway")
        print("4. Bot runs 24/7 for FREE!")
        
    else:
        print("❌ Git setup failed")

if __name__ == "__main__":
    main() 