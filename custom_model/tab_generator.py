"""
Unified Tab Generator
Provides a unified interface for both CREPE-based and custom model tablature generation.
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Literal
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "project"))

from custom_model_transcriber import CustomModelTranscriber

# Try to import the CREPE-based transcriber
try:
    from guitar_transcription import GuitarTranscription
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("⚠️  CREPE-based transcriber not available. Only custom model will be used.")

class TabGenerator:
    """Unified interface for guitar tablature generation using different models."""
    
    def __init__(self, 
                 model_type: Literal['custom', 'crepe', 'both'] = 'custom',
                 custom_model_path: str = None,
                 output_formats: List[str] = ['text', 'json']):
        """Initialize the tab generator.
        
        Args:
            model_type: Which model(s) to use ('custom', 'crepe', 'both')
            custom_model_path: Path to custom model (optional, will auto-detect)
            output_formats: List of output formats ('text', 'json', 'latex', 'pdf')
        """
        self.model_type = model_type
        self.output_formats = output_formats
        
        # Initialize transcribers
        self.custom_transcriber = None
        self.crepe_transcriber = None
        
        # Initialize based on model type
        if model_type in ['custom', 'both']:
            try:
                self.custom_transcriber = CustomModelTranscriber(custom_model_path)
                print("✅ Custom model transcriber initialized")
            except Exception as e:
                print(f"❌ Failed to initialize custom model: {e}")
                if model_type == 'custom':
                    raise
        
        if model_type in ['crepe', 'both']:
            if CREPE_AVAILABLE:
                try:
                    self.crepe_transcriber = GuitarTranscription()
                    print("✅ CREPE transcriber initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize CREPE transcriber: {e}")
                    if model_type == 'crepe':
                        raise
            else:
                print("❌ CREPE transcriber not available")
                if model_type == 'crepe':
                    raise RuntimeError("CREPE transcriber not available")
    
    def transcribe_file(self, 
                       audio_file: str, 
                       output_dir: str = "./output",
                       model_preference: Literal['custom', 'crepe'] = None) -> Dict:
        """Transcribe an audio file to tablature.
        
        Args:
            audio_file: Path to audio file
            output_dir: Output directory for results
            model_preference: Which model to use (overrides self.model_type)
        
        Returns:
            Dictionary containing transcription results
        """
        print(f"🎸 Transcribing: {os.path.basename(audio_file)}")
        
        # Determine which model to use
        use_custom = self._should_use_custom(model_preference)
        use_crepe = self._should_use_crepe(model_preference)
        
        results = {
            'file': audio_file,
            'models_used': [],
            'results': {}
        }
        
        # Custom model transcription
        if use_custom and self.custom_transcriber:
            try:
                print("🎯 Using custom model...")
                custom_results = self.custom_transcriber.transcribe_file(audio_file, output_dir)
                results['results']['custom'] = custom_results
                results['models_used'].append('custom')
                print(f"✅ Custom model: {custom_results['chord']} (conf: {custom_results['confidence']:.3f})")
            except Exception as e:
                print(f"❌ Custom model failed: {e}")
                results['results']['custom'] = {'error': str(e)}
        
        # CREPE transcription
        if use_crepe and self.crepe_transcriber:
            try:
                print("🎯 Using CREPE model...")
                crepe_results = self._transcribe_with_crepe(audio_file, output_dir)
                results['results']['crepe'] = crepe_results
                results['models_used'].append('crepe')
                print(f"✅ CREPE model: {len(crepe_results.get('tab_sequence', []))} chord positions detected")
            except Exception as e:
                print(f"❌ CREPE model failed: {e}")
                results['results']['crepe'] = {'error': str(e)}
        
        # Combine results if both models used
        if len(results['models_used']) > 1:
            results = self._combine_results(results)
        
        # Export results
        self._export_unified_results(results, output_dir)
        
        return results
    
    def transcribe_audio_data(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int = 22050,
                             model_preference: Literal['custom', 'crepe'] = None) -> Dict:
        """Transcribe audio data directly.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            model_preference: Which model to use
        
        Returns:
            Dictionary containing transcription results
        """
        print("🎸 Transcribing audio data...")
        
        # Determine which model to use
        use_custom = self._should_use_custom(model_preference)
        use_crepe = self._should_use_crepe(model_preference)
        
        results = {
            'models_used': [],
            'results': {}
        }
        
        # Custom model transcription
        if use_custom and self.custom_transcriber:
            try:
                print("🎯 Using custom model...")
                custom_results = self.custom_transcriber.transcribe_audio_data(audio_data, sample_rate)
                results['results']['custom'] = custom_results
                results['models_used'].append('custom')
                print(f"✅ Custom model: {custom_results['chord']} (conf: {custom_results['confidence']:.3f})")
            except Exception as e:
                print(f"❌ Custom model failed: {e}")
                results['results']['custom'] = {'error': str(e)}
        
        # CREPE transcription
        if use_crepe and self.crepe_transcriber:
            try:
                print("🎯 Using CREPE model...")
                crepe_results = self._transcribe_audio_with_crepe(audio_data, sample_rate)
                results['results']['crepe'] = crepe_results
                results['models_used'].append('crepe')
                print(f"✅ CREPE model: {len(crepe_results.get('tab_sequence', []))} chord positions detected")
            except Exception as e:
                print(f"❌ CREPE model failed: {e}")
                results['results']['crepe'] = {'error': str(e)}
        
        # Combine results if both models used
        if len(results['models_used']) > 1:
            results = self._combine_results(results)
        
        return results
    
    def _should_use_custom(self, preference: Optional[str]) -> bool:
        """Determine if custom model should be used."""
        if preference:
            return preference == 'custom'
        return self.model_type in ['custom', 'both']
    
    def _should_use_crepe(self, preference: Optional[str]) -> bool:
        """Determine if CREPE model should be used."""
        if preference:
            return preference == 'crepe'
        return self.model_type in ['crepe', 'both']
    
    def _transcribe_with_crepe(self, audio_file: str, output_dir: str) -> Dict:
        """Transcribe using CREPE-based system."""
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Transcribe
        crepe_results = self.crepe_transcriber.transcribe_audio(audio, sr, output_dir)
        
        return crepe_results
    
    def _transcribe_audio_with_crepe(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Transcribe audio data using CREPE-based system."""
        # Transcribe
        crepe_results = self.crepe_transcriber.transcribe_audio(audio_data, sample_rate)
        
        return crepe_results
    
    def _combine_results(self, results: Dict) -> Dict:
        """Combine results from multiple models."""
        print("🔄 Combining results from multiple models...")
        
        # Add comparison analysis
        results['comparison'] = self._analyze_model_comparison(results['results'])
        
        # Add consensus prediction
        results['consensus'] = self._get_consensus_prediction(results['results'])
        
        return results
    
    def _analyze_model_comparison(self, model_results: Dict) -> Dict:
        """Analyze and compare results from different models."""
        comparison = {
            'chord_agreement': False,
            'confidence_comparison': {},
            'tablature_similarity': 0.0,
            'notes': []
        }
        
        # Compare chords if both models predicted
        if 'custom' in model_results and 'crepe' in model_results:
            custom_result = model_results['custom']
            crepe_result = model_results['crepe']
            
            # Chord comparison
            if 'chord' in custom_result and 'chord_sequence' in crepe_result:
                custom_chord = custom_result['chord']
                crepe_chords = crepe_result['chord_sequence']
                
                if crepe_chords and len(crepe_chords) > 0:
                    # Compare with most frequent CREPE chord
                    from collections import Counter
                    chord_counts = Counter(crepe_chords)
                    most_common_crepe = chord_counts.most_common(1)[0][0]
                    
                    comparison['chord_agreement'] = custom_chord == most_common_crepe
                    comparison['notes'].append(f"Custom: {custom_chord}, CREPE: {most_common_crepe}")
            
            # Confidence comparison
            if 'confidence' in custom_result:
                comparison['confidence_comparison']['custom'] = custom_result['confidence']
            
            # Tablature comparison
            if 'tablature' in custom_result and 'tab_sequence' in crepe_result:
                similarity = self._calculate_tablature_similarity(
                    custom_result['tablature'], 
                    crepe_result['tab_sequence']
                )
                comparison['tablature_similarity'] = similarity
        
        return comparison
    
    def _calculate_tablature_similarity(self, custom_tab: Dict, crepe_tab: List) -> float:
        """Calculate similarity between custom and CREPE tablature."""
        if not crepe_tab:
            return 0.0
        
        # Simple similarity based on common notes
        custom_notes = set()
        for string, fret in custom_tab.items():
            if fret is not None:
                custom_notes.add((string, fret))
        
        crepe_notes = set()
        for chord_pos in crepe_tab:
            for note in chord_pos.get('notes', []):
                crepe_notes.add((note['string_name'], note['fret']))
        
        # Calculate Jaccard similarity
        intersection = len(custom_notes.intersection(crepe_notes))
        union = len(custom_notes.union(crepe_notes))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_consensus_prediction(self, model_results: Dict) -> Dict:
        """Get consensus prediction from multiple models."""
        consensus = {
            'chord': None,
            'confidence': 0.0,
            'tablature': {},
            'method': 'weighted_average'
        }
        
        # Weight models by availability and performance
        weights = {'custom': 0.6, 'crepe': 0.4}
        
        chord_votes = {}
        confidence_sum = 0.0
        weight_sum = 0.0
        
        # Collect predictions
        for model_name, results in model_results.items():
            if 'error' not in results and model_name in weights:
                weight = weights[model_name]
                weight_sum += weight
                
                # Chord prediction
                if 'chord' in results:
                    chord = results['chord']
                    chord_votes[chord] = chord_votes.get(chord, 0) + weight
                
                # Confidence
                if 'confidence' in results:
                    confidence_sum += results['confidence'] * weight
        
        # Determine consensus chord
        if chord_votes:
            consensus['chord'] = max(chord_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence
        if weight_sum > 0:
            consensus['confidence'] = confidence_sum / weight_sum
        
        # For tablature, use custom model as primary (more detailed)
        if 'custom' in model_results and 'error' not in model_results['custom']:
            if 'tablature' in model_results['custom']:
                consensus['tablature'] = model_results['custom']['tablature']
        
        return consensus
    
    def _export_unified_results(self, results: Dict, output_dir: str):
        """Export unified results in multiple formats."""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = "unified_transcription"
        if 'file' in results:
            base_name = os.path.splitext(os.path.basename(results['file']))[0]
        
        # Export in requested formats
        for format_type in self.output_formats:
            if format_type == 'text':
                self._export_text_format(results, output_dir, base_name)
            elif format_type == 'json':
                self._export_json_format(results, output_dir, base_name)
            elif format_type == 'latex':
                self._export_latex_format(results, output_dir, base_name)
    
    def _export_text_format(self, results: Dict, output_dir: str, base_name: str):
        """Export results in text format."""
        output_file = os.path.join(output_dir, f"{base_name}_unified_tab.txt")
        
        with open(output_file, 'w') as f:
            f.write("🎸 UNIFIED GUITAR TABLATURE 🎸\n")
            f.write("=" * 60 + "\n")
            
            # Write model information
            f.write(f"Models used: {', '.join(results['models_used'])}\n")
            f.write("=" * 60 + "\n\n")
            
            # Write results from each model
            for model_name, model_results in results['results'].items():
                if 'error' not in model_results:
                    f.write(f"🎯 {model_name.upper()} MODEL RESULTS:\n")
                    f.write("-" * 40 + "\n")
                    
                    if 'chord' in model_results:
                        f.write(f"Predicted Chord: {model_results['chord']}\n")
                        if 'confidence' in model_results:
                            f.write(f"Confidence: {model_results['confidence']:.3f}\n")
                    
                    if 'tablature' in model_results:
                        f.write("\nTablature:\n")
                        self._write_tablature_to_file(f, model_results['tablature'])
                    
                    f.write("\n")
            
            # Write comparison if multiple models
            if 'comparison' in results:
                f.write("🔄 MODEL COMPARISON:\n")
                f.write("-" * 40 + "\n")
                comparison = results['comparison']
                
                f.write(f"Chord Agreement: {'✅ Yes' if comparison['chord_agreement'] else '❌ No'}\n")
                f.write(f"Tablature Similarity: {comparison['tablature_similarity']:.3f}\n")
                
                for note in comparison['notes']:
                    f.write(f"• {note}\n")
                
                f.write("\n")
            
            # Write consensus
            if 'consensus' in results:
                f.write("🎯 CONSENSUS PREDICTION:\n")
                f.write("-" * 40 + "\n")
                consensus = results['consensus']
                
                f.write(f"Chord: {consensus['chord']}\n")
                f.write(f"Confidence: {consensus['confidence']:.3f}\n")
                
                if consensus['tablature']:
                    f.write("\nTablature:\n")
                    self._write_tablature_to_file(f, consensus['tablature'])
    
    def _write_tablature_to_file(self, file, tablature):
        """Write tablature to file in standard format."""
        if isinstance(tablature, dict):
            # Custom model format
            string_names = ['e', 'B', 'G', 'D', 'A', 'E']
            for string_name in string_names:
                fret = tablature.get(string_name.upper())
                if fret is not None:
                    file.write(f"{string_name}|--{fret:2d}--\n")
                else:
                    file.write(f"{string_name}|-----\n")
        else:
            # CREPE format (list of chord positions)
            for chord_pos in tablature:
                file.write(f"Time: {chord_pos['time']:.2f}s\n")
                for note in chord_pos.get('notes', []):
                    file.write(f"  {note['string_name']}: fret {note['fret']}\n")
                file.write("\n")
    
    def _export_json_format(self, results: Dict, output_dir: str, base_name: str):
        """Export results in JSON format."""
        import json
        output_file = os.path.join(output_dir, f"{base_name}_unified_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _export_latex_format(self, results: Dict, output_dir: str, base_name: str):
        """Export results in LaTeX format."""
        # This would integrate with the existing LaTeX export functionality
        # For now, just export as text
        print("⚠️  LaTeX export not yet implemented")
    
    def get_model_status(self) -> Dict:
        """Get status of available models."""
        status = {
            'custom_model': self.custom_transcriber is not None,
            'crepe_model': self.crepe_transcriber is not None,
            'model_type': self.model_type
        }
        
        if self.custom_transcriber:
            status['custom_info'] = self.custom_transcriber.get_model_info()
        
        return status

# Example usage
if __name__ == "__main__":
    # Test the unified tab generator
    print("🎸 Testing Unified Tab Generator")
    print("=" * 50)
    
    # Initialize with custom model only
    generator = TabGenerator(model_type='custom')
    
    # Get status
    status = generator.get_model_status()
    print("Model Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test with a sample file if available
    test_file = "/Users/james/Documents/CS3642/CS3642/project/test_chords/17569__danglada__c-major.wav"
    if os.path.exists(test_file):
        print(f"\n🎵 Testing with: {test_file}")
        results = generator.transcribe_file(test_file)
        
        print(f"Results:")
        print(f"  Models used: {results['models_used']}")
        for model_name, model_results in results['results'].items():
            if 'error' not in model_results:
                print(f"  {model_name}: {model_results.get('chord', 'N/A')} (conf: {model_results.get('confidence', 0):.3f})")
    else:
        print(f"\n⚠️  Test file not found: {test_file}")
