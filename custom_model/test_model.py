"""
Test script for the custom guitar model
Quick test to make sure everything works together.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import librosa
from models.guitar_model import create_model
from preprocessing.audio_processor import GuitarAudioProcessor

def test_model():
    """Test the custom guitar model with real audio."""
    print("🎸 Testing Custom Guitar Model")
    print("=" * 50)
    
    # Create model
    print("Creating model...")
    model = create_model()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create audio processor
    print("Creating audio processor...")
    processor = GuitarAudioProcessor()
    print("✅ Audio processor created")
    
    # Load test audio (using one of your chord files)
    test_audio_path = "../project/test_chords/17569__danglada__c-major.wav"
    
    if os.path.exists(test_audio_path):
        print(f"Loading test audio: {test_audio_path}")
        audio, sr = librosa.load(test_audio_path, sr=22050)
        print(f"✅ Audio loaded: {len(audio)} samples, {sr} Hz")
        
        # Preprocess audio
        print("Preprocessing audio...")
        processed_audio = processor.preprocess_audio(audio)
        print(f"✅ Audio preprocessed: {len(processed_audio)} samples")
        
        # Extract features
        print("Extracting features...")
        spectral_features = processor.extract_spectral_features(processed_audio)
        print("✅ Features extracted:")
        for key, value in spectral_features.items():
            print(f"   {key}: {value.shape}")
        
        # Create training data format
        print("Creating training data...")
        target_chord = "C"
        target_tablature = {'E': 3, 'A': 3, 'D': 0, 'G': 0, 'B': 1, 'e': 0}
        
        training_data = processor.create_training_data(processed_audio, target_chord, target_tablature)
        print("✅ Training data created:")
        print(f"   Input shape: {training_data['input'].shape}")
        print(f"   Chord target shape: {training_data['chord_target'].shape}")
        print(f"   Tablature target shape: {training_data['tablature_target'].shape}")
        
        # Test model forward pass
        print("Testing model forward pass...")
        model.eval()
        
        # Reshape input for model (add batch and channel dimensions)
        # The model expects [batch, channels, sequence_length]
        input_tensor = training_data['input'].mean(dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print("✅ Model forward pass successful:")
        for key, value in outputs.items():
            print(f"   {key}: {value.shape}")
        
        # Test predictions
        print("Testing predictions...")
        chord_name, confidence = model.predict_chord(input_tensor)
        print(f"✅ Predicted chord: {chord_name} (confidence: {confidence:.3f})")
        
        tablature = model.predict_tablature(input_tensor)
        print(f"✅ Predicted tablature: {tablature}")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        
    else:
        print(f"❌ Test audio not found: {test_audio_path}")
        print("Creating dummy test...")
        
        # Create dummy audio
        dummy_audio = np.random.randn(22050)  # 1 second of random audio
        processed_audio = processor.preprocess_audio(dummy_audio)
        spectral_features = processor.extract_spectral_features(processed_audio)
        
        # Test with dummy data
        input_tensor = torch.randn(1, 1, 1024)  # Dummy input
        outputs = model(input_tensor)
        
        print("✅ Dummy test passed!")
        for key, value in outputs.items():
            print(f"   {key}: {value.shape}")

def test_audio_processor():
    """Test the audio processor with different chord files."""
    print("\n🎵 Testing Audio Processor with Multiple Chords")
    print("=" * 50)
    
    processor = GuitarAudioProcessor()
    
    # Test with different chord files
    chord_files = [
        ("../project/test_chords/17569__danglada__c-major.wav", "C"),
        ("../project/test_chords/17570__danglada__d-major.wav", "D"),
        ("../project/test_chords/17571__danglada__e-major.wav", "E"),
        ("../project/test_chords/17572__danglada__f-major.wav", "F"),
        ("../project/test_chords/17573__danglada__g-major.wav", "G"),
    ]
    
    for file_path, expected_chord in chord_files:
        if os.path.exists(file_path):
            print(f"\nTesting {expected_chord} chord...")
            
            # Load and process
            audio, sr = librosa.load(file_path, sr=22050)
            processed_audio = processor.preprocess_audio(audio)
            spectral_features = processor.extract_spectral_features(processed_audio)
            
            # Analyze guitar chroma
            guitar_chroma = spectral_features['guitar_chroma']
            chroma_peaks = np.argmax(guitar_chroma, axis=0)
            
            print(f"   Audio length: {len(audio)} samples")
            print(f"   Chroma shape: {guitar_chroma.shape}")
            print(f"   Dominant chroma: {chroma_peaks[:5]}")  # First 5 time frames
            
        else:
            print(f"   ❌ File not found: {file_path}")

if __name__ == "__main__":
    # Test the model
    test_model()
    
    # Test audio processor with multiple chords
    test_audio_processor()
    
    print("\n🚀 Ready to start training!")
    print("Next steps:")
    print("1. Collect more training data")
    print("2. Set up data loading pipeline")
    print("3. Start training the model")
    print("4. Evaluate against CREPE system")
