#!/usr/bin/env python3
"""
Integration Test Script
Tests the basic functionality of the GUI integration components.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the custom_model directory to the path
sys.path.append(str(Path(__file__).parent))

def test_config_system():
    """Test the configuration system."""
    print("🧪 Testing Configuration System")
    print("-" * 40)
    
    try:
        from config import ConfigManager, ConfigPresets
        
        # Test configuration manager
        config_manager = ConfigManager()
        print("✅ ConfigManager created successfully")
        
        # Test getting settings
        audio_settings = config_manager.get_audio_settings()
        print(f"✅ Audio settings: {audio_settings.sample_rate}Hz")
        
        model_settings = config_manager.get_model_settings()
        print(f"✅ Model settings: {model_settings.model_type}")
        
        # Test preset application
        ConfigPresets.apply_preset(config_manager, 'real_time')
        print("✅ Preset applied successfully")
        
        # Test validation
        issues = config_manager.validate_config()
        if issues:
            print(f"⚠️  Config issues: {issues}")
        else:
            print("✅ Configuration is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Config system test failed: {e}")
        return False

def test_audio_recorder():
    """Test the audio recorder (without actual recording)."""
    print("\n🧪 Testing Audio Recorder")
    print("-" * 40)
    
    try:
        from audio_recorder import AudioRecorder, AudioLevelMonitor
        
        # Test audio level monitor
        level_monitor = AudioLevelMonitor()
        level_monitor.update(0.5)
        level_monitor.update(0.3)
        
        avg_level = level_monitor.get_average_level()
        peak_level = level_monitor.get_peak_level()
        
        print(f"✅ Audio level monitor: avg={avg_level:.3f}, peak={peak_level:.3f}")
        
        # Test silence detection
        is_silent = level_monitor.is_silent(0.1)
        print(f"✅ Silence detection: {is_silent}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio recorder test failed: {e}")
        return False

def test_custom_model_structure():
    """Test the custom model structure without loading."""
    print("\n🧪 Testing Custom Model Structure")
    print("-" * 40)
    
    try:
        from models.guitar_model import create_model
        
        # Create model (without loading weights)
        model = create_model(num_chords=7)
        print("✅ Custom model created successfully")
        
        # Test model info
        print(f"✅ Model has {len(model.chord_names)} chord classes")
        print(f"✅ Chord names: {model.chord_names}")
        
        # Test dummy forward pass
        import torch
        dummy_input = torch.randn(1, 1, 854)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✅ Model forward pass successful")
        for key, value in outputs.items():
            print(f"   {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom model test failed: {e}")
        return False

def test_tab_generator_structure():
    """Test the tab generator structure."""
    print("\n🧪 Testing Tab Generator Structure")
    print("-" * 40)
    
    try:
        from tab_generator import TabGenerator
        
        # Test creating tab generator (without full initialization)
        print("✅ TabGenerator class imported successfully")
        
        # Test model status method exists
        if hasattr(TabGenerator, 'get_model_status'):
            print("✅ get_model_status method exists")
        else:
            print("❌ get_model_status method missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Tab generator test failed: {e}")
        return False

def test_gui_imports():
    """Test GUI imports."""
    print("\n🧪 Testing GUI Imports")
    print("-" * 40)
    
    try:
        import tkinter as tk
        from tkinter import ttk
        print("✅ tkinter imports successful")
        
        # Test creating a simple window (without showing it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        button = ttk.Button(frame, text="Test Button")
        
        root.destroy()
        print("✅ GUI components created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure")
    print("-" * 40)
    
    required_files = [
        'custom_model_transcriber.py',
        'tab_generator.py',
        'audio_recorder.py',
        'guitar_tab_gui.py',
        'config.py',
        'models/guitar_model.py',
        'preprocessing/audio_processor.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all integration tests."""
    print("🎸 GUITAR TAB GUI INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_config_system,
        test_audio_recorder,
        test_custom_model_structure,
        test_tab_generator_structure,
        test_gui_imports
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
