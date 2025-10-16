#!/usr/bin/env python3
"""
Simple GUI Launcher - Auto-launches without prompting
"""

import sys
import os
from pathlib import Path

# Add the custom_model directory to the path
sys.path.append(str(Path(__file__).parent))

def main():
    """Launch the GUI directly."""
    print("🎸 Starting Guitar Tab GUI...")
    
    try:
        from guitar_tab_gui import GuitarTabGUI
        
        app = GuitarTabGUI()
        print("✅ GUI initialized successfully")
        print("🎸 Guitar Tab Generator is ready!")
        
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the custom_model directory")
        print("2. Check that all required files exist")
        print("3. Try running: python test_integration.py")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
