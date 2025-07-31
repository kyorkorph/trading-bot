#!/usr/bin/env python3
"""
Cleanup Unnecessary Files
Safely removes files that are not needed for the current trading system
"""

import os
import shutil
import json
from datetime import datetime

# ESSENTIAL FILES - DO NOT DELETE
ESSENTIAL_FILES = {
    "interactive_telegram_bot.py",
    "ultimate_trading_system.py", 
    "persistent_bot_launcher.py",
    "start_persistent_bot.sh",
    "ultimate_performance.json",
    "ultimate_last_run.json",
    "bot_run_log.txt",
    "trading_system.log",
    "persistent_bot.log",
    "setup_bot_commands.py",
    "cleanup_unnecessary_files.py"  # This script itself
}

# FILES THAT CAN BE SAFELY DELETED
UNNECESSARY_FILES = [
    # Old strategy files
    "main_backtester.py",
    "live_trading_alerts.py",
    "advanced_strategies.py",
    "test_advanced_strategies.py",
    "gold_futures_testing.py",
    "gold_futures_alerts.py",
    "genetic_strategy_optimizer.py",
    "self_improving_strategies.py",
    "self_improving_gold_strategy.py",
    "advanced_ml_models.py",
    "web_dashboard.py",
    "trading_system_status.py",
    "automated_monitor.py",
    "million_dollar_trading_system.py",
    "test_million_dollar_alert.py",
    "advanced_ml_strategy.py",
    "test_advanced_ml_signal.py",
    "gold_futures_specs.py",
    "automatic_performance_tracker.py",
    "test_performance_report.py",
    "hundred_k_trading_system.py",
    "connectivity_guide.py",
    "current_strategy_breakdown.py",
    "strategy_summary.py",
    "test_message.py",
    "test_inactive_message.py",
    "apr_calculation.py",
    "laptop_usage_guide.py",
    "fixed_trading_system.py",
    "final_answer.py",
    "comprehensive_backtest.py",
    "realistic_backtest.py",
    "derivatives_backtest.py",
    "simple_derivatives_backtest.py",
    "derivatives_summary.py",
    "fixed_trading_system_v2.py",
    "system_status_check.py",
    "system_restart_confirmation.py",
    "colab_fixed_system.py",
    "colab_fix_instructions.py",
    "google_colab_setup_guide.py",
    "complete_colab_notebook.py",
    "quick_colab_setup.py",
    "cloud_trading_setup.py",
    "final_status.py",
    "smart_daily_trading.py",
    "daily_reminder_system.py",
    "setup_smart_daily.py",
    "working_trading_system.py",
    "final_working_system.py",
    "ultra_simple_system.py",
    "working_system_final.py",
    "strategy_breakdown.py",
    "apr_backtest.py",
    "enhanced_daily_system.py",
    "working_final_performance.json",
    "daily_trading_performance.json",
    "enhanced_performance.json",
    "last_run_timestamp.json",
    "simple_working_system.py",
    "final_working_system.py",
    "working_trading_system.py",
    "final_trading_system.log",
    "fixed_system.log",
    "test_alert_demo.py",
    "alert_explanation.py",
    "what_to_do_now.py",
    "start_interactive_bot.py",
    "auto_run_trading.py",
    "automation_setup_guide.py",
    "trading_timeline_guide.py",
    "enhanced_performance.json",
    "working_final_performance.json",
    "daily_trading_performance.json",
    "last_run_timestamp.json",
    "enhanced_daily_system.py",
    "apr_backtest.py",
    "strategy_breakdown.py",
    "working_system_final.py",
    "ultra_simple_system.py",
    "simple_working_system.py",
    "final_working_system.py",
    "working_trading_system.py",
    "setup_smart_daily.py",
    "daily_reminder_system.py",
    "smart_daily_trading.py",
    "colab_fix_instructions.py",
    "colab_fixed_system.py",
    "quick_colab_setup.py",
    "complete_colab_notebook.py",
    "google_colab_setup_guide.py",
    "google_colab_trading.py",
    "cloud_trading_setup.py",
    "final_status.py",
    "system_restart_confirmation.py",
    "system_status_check.py",
    "fixed_trading_system_v2.py",
    "derivatives_summary.py",
    "simple_derivatives_backtest.py",
    "derivatives_backtest.py",
    "backtest_summary.py",
    "realistic_backtest.py",
    "comprehensive_backtest.py",
    "final_answer.py",
    "fixed_system.log",
    "fixed_trading_system.py",
    "laptop_usage_guide.py",
    "apr_calculation.py",
    "test_inactive_message.py",
    "test_message.py",
    "strategy_summary.py",
    "current_strategy_breakdown.py",
    "connectivity_guide.py",
    "hundred_k_trading_system.py",
    "test_performance_report.py",
    "automatic_performance_tracker.py",
    "gold_futures_specs.py",
    "test_advanced_ml_signal.py",
    "advanced_ml_strategy.py",
    "test_million_dollar_alert.py",
    "million_dollar_trading_system.py",
    "test_detailed_alert.py",
    "smart_trading_alerts.py",
    "cleanup_files.py",
    "automated_monitor.py",
    "trading_system_status.py",
    "web_dashboard.py",
    "advanced_ml_models.py",
    "advanced_risk_management.py",
    "self_improving_gold_strategy.py",
    "leading_analysis.py",
    "leading_gold_strategy.py",
    "quick_analysis.py",
    "simple_gold_alerts.py",
    "live_trading_alerts.py",
    "README_SELF_IMPROVING.md",
    "self_improving_strategies.py",
    "genetic_strategy_optimizer.py",
    "README_GOLD_FUTURES.md",
    "gold_futures_alerts.py",
    "gold_futures_testing.py",
    "README_ADVANCED_STRATEGIES.md",
    "advanced_strategies.py",
    "README_ALERTS.md",
    "main_backtester.py"
]

def test_essential_files():
    """Test that essential files exist and are accessible"""
    print("🧪 Testing essential files...")
    
    missing_files = []
    for file in ESSENTIAL_FILES:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    else:
        print("✅ All essential files present")
        return True

def test_trading_system():
    """Test that the trading system can still run"""
    print("🧪 Testing trading system...")
    
    try:
        # Test importing the main system
        import subprocess
        result = subprocess.run(["python3", "-c", "import ultimate_trading_system"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Trading system import test passed")
            return True
        else:
            print(f"❌ Trading system import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Trading system test failed: {e}")
        return False

def cleanup_files():
    """Remove unnecessary files"""
    print("🧹 Starting cleanup...")
    
    deleted_count = 0
    failed_deletions = []
    
    for file in UNNECESSARY_FILES:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"🗑️ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {file}: {e}")
                failed_deletions.append(file)
    
    print(f"\n📊 Cleanup Summary:")
    print(f"✅ Deleted: {deleted_count} files")
    print(f"❌ Failed: {len(failed_deletions)} files")
    
    if failed_deletions:
        print(f"Failed deletions: {failed_deletions}")
    
    return deleted_count, failed_deletions

def main():
    """Main cleanup function"""
    print("🧹 TRADING SYSTEM CLEANUP")
    print("=" * 50)
    
    # Test essential files first
    if not test_essential_files():
        print("❌ Cannot proceed - essential files missing!")
        return
    
    # Test trading system
    if not test_trading_system():
        print("❌ Cannot proceed - trading system test failed!")
        return
    
    # Show what will be deleted
    print(f"\n📋 Files to be deleted: {len(UNNECESSARY_FILES)}")
    print("This will free up significant disk space!")
    
    # Confirm deletion
    response = input("\n🤔 Proceed with cleanup? (y/N): ").lower().strip()
    if response != 'y':
        print("❌ Cleanup cancelled")
        return
    
    # Perform cleanup
    deleted_count, failed_deletions = cleanup_files()
    
    # Final test
    print("\n🧪 Final system test...")
    if test_essential_files() and test_trading_system():
        print("✅ System is working correctly after cleanup!")
        
        # Calculate space saved
        total_size = 0
        for file in UNNECESSARY_FILES:
            if os.path.exists(file):
                total_size += os.path.getsize(file)
        
        print(f"💾 Space saved: {total_size / 1024 / 1024:.2f} MB")
    else:
        print("❌ System test failed after cleanup!")

if __name__ == "__main__":
    main() 