"""
Custom Model Transcriber
Bridge between the custom trained guitar model and existing tab generation pipeline.
"""

import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Add the custom_model directory to the path
sys.path.append(str(Path(__file__).parent))

from models.guitar_model import create_model
from preprocessing.audio_processor import GuitarAudioProcessor

class CustomModelTranscriber:
    """Transcriber using the custom trained guitar model."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """Initialize the custom model transcriber."""
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.audio_processor = GuitarAudioProcessor()
        
        # Default model paths
        self.default_model_paths = [
            'trained_models/best_model.pth',
            'trained_models/final_model.pth',
            'trained_models_large/best_model.pth',
            'trained_models_large/final_model.pth'
        ]
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            self._auto_load_model()
    
    def _auto_load_model(self):
        """Automatically find and load the best available model."""
        for model_path in self.default_model_paths:
            if os.path.exists(model_path):
                print(f"🎸 Auto-loading model: {model_path}")
                self.load_model(model_path)
                return
        
        print("⚠️  No trained model found. Please train a model first or specify a model path.")
    
    def load_model(self, model_path: str):
        """Load a trained model from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🎸 Loading custom model from: {model_path}")
        
        try:
            # Create model (assuming 7 chords for simple model, 24 for large model)
            num_chords = 24 if 'large' in model_path else 7
            self.model = create_model(num_chords=num_chords)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray, sample_rate: int = 22050) -> torch.Tensor:
        """Preprocess audio for the custom model."""
        # Ensure audio is 2 seconds long
        target_length = int(2.0 * sample_rate)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return audio_tensor
    
    def predict_chord(self, audio: np.ndarray, sample_rate: int = 22050) -> Tuple[str, float]:
        """Predict chord from audio using the custom model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio, sample_rate)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            
            # Get chord prediction
            chord_probs = torch.softmax(outputs['chord_logits'], dim=1)
            chord_idx = torch.argmax(chord_probs, dim=1).item()
            chord_confidence = chord_probs[0, chord_idx].item()
            
            # Get chord name
            chord_name = self.model.chord_names[chord_idx]
            
            return chord_name, chord_confidence
    
    def predict_tablature(self, audio: np.ndarray, sample_rate: int = 22050) -> Dict[str, Optional[int]]:
        """Predict guitar tablature from audio using the custom model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio, sample_rate)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            
            # Get fret positions and string activations
            fret_positions = outputs['fret_positions'].cpu().numpy()[0]
            string_activations = outputs['string_activations'].cpu().numpy()[0]
            
            # Create tablature representation
            tablature = {}
            string_names = ['E', 'A', 'D', 'G', 'B', 'e']
            
            for i, (fret, activation) in enumerate(zip(fret_positions, string_activations)):
                if activation > 0.5:  # String is played
                    tablature[string_names[i]] = int(round(fret))
                else:
                    tablature[string_names[i]] = None  # String not played
            
            return tablature
    
    def transcribe_file(self, audio_file: str, output_dir: str = "./output") -> Dict:
        """Transcribe an audio file using the custom model."""
        print(f"🎵 Transcribing: {os.path.basename(audio_file)}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Predict chord and tablature
        chord_name, chord_confidence = self.predict_chord(audio, sr)
        tablature = self.predict_tablature(audio, sr)
        
        # Create results
        results = {
            'file': audio_file,
            'chord': chord_name,
            'confidence': chord_confidence,
            'tablature': tablature,
            'duration': len(audio) / sr,
            'sample_rate': sr
        }
        
        # Export results
        self._export_results(results, output_dir)
        
        return results
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict:
        """Transcribe audio data directly."""
        print("🎵 Transcribing audio data...")
        
        # Predict chord and tablature
        chord_name, chord_confidence = self.predict_chord(audio_data, sample_rate)
        tablature = self.predict_tablature(audio_data, sample_rate)
        
        # Create results
        results = {
            'chord': chord_name,
            'confidence': chord_confidence,
            'tablature': tablature,
            'duration': len(audio_data) / sample_rate,
            'sample_rate': sample_rate
        }
        
        return results
    
    def _export_results(self, results: Dict, output_dir: str):
        """Export transcription results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        base_name = os.path.splitext(os.path.basename(results['file']))[0] if 'file' in results else "transcription"
        
        # Export text tablature
        tab_file = os.path.join(output_dir, f"{base_name}_custom_tab.txt")
        self._export_text_tablature(results, tab_file)
        
        # Export JSON results
        import json
        json_file = os.path.join(output_dir, f"{base_name}_custom_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results exported to: {output_dir}")
    
    def _export_text_tablature(self, results: Dict, output_file: str):
        """Export tablature to text file."""
        with open(output_file, 'w') as f:
            f.write("🎸 CUSTOM MODEL GUITAR TABLATURE 🎸\n")
            f.write("=" * 50 + "\n")
            f.write(f"Predicted Chord: {results['chord']} (confidence: {results['confidence']:.3f})\n")
            f.write("=" * 50 + "\n\n")
            
            # Create tablature representation
            string_names = ['e', 'B', 'G', 'D', 'A', 'E']
            tab_lines = []
            
            for string_name in string_names:
                fret = results['tablature'].get(string_name.upper())
                if fret is not None:
                    tab_lines.append(f"{string_name}|--{fret:2d}--")
                else:
                    tab_lines.append(f"{string_name}|-----")
            
            # Write in guitar order (high E to low E)
            for line in reversed(tab_lines):
                f.write(line + "\n")
            
            f.write("\n📝 Notes detected:\n")
            for string_name, fret in results['tablature'].items():
                if fret is not None:
                    f.write(f"  • {string_name} string: fret {fret}\n")
                else:
                    f.write(f"  • {string_name} string: not played\n")
    
    def batch_transcribe(self, audio_files: List[str], output_dir: str = "./output") -> List[Dict]:
        """Transcribe multiple audio files."""
        results = []
        
        for audio_file in audio_files:
            try:
                result = self.transcribe_file(audio_file, output_dir)
                results.append(result)
            except Exception as e:
                print(f"❌ Error transcribing {audio_file}: {e}")
                results.append({
                    'file': audio_file,
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'status': 'Model loaded',
            'device': self.device,
            'num_chords': len(self.model.chord_names),
            'chord_names': self.model.chord_names
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the transcriber
    transcriber = CustomModelTranscriber()
    
    # Get model info
    print("Model Info:")
    info = transcriber.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with a sample file if available
    test_file = "/Users/james/Documents/CS3642/CS3642/project/test_chords/17569__danglada__c-major.wav"
    if os.path.exists(test_file):
        print(f"\n🎵 Testing with: {test_file}")
        results = transcriber.transcribe_file(test_file)
        
        print(f"Results:")
        print(f"  Chord: {results['chord']} (confidence: {results['confidence']:.3f})")
        print(f"  Tablature: {results['tablature']}")
    else:
        print(f"\n⚠️  Test file not found: {test_file}")
