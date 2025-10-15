"""
Custom Audio Preprocessing for Guitar Transcription
Specialized preprocessing pipeline for guitar audio analysis.
"""

import numpy as np
import librosa
import torch
from typing import Tuple, Dict, List
from scipy import signal
from scipy.signal import butter, filtfilt

class GuitarAudioProcessor:
    """Custom audio processor for guitar transcription."""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Guitar-specific parameters
        self.guitar_strings = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # E2-A2-D3-G3-B3-E4
        self.string_names = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
        
        # Frequency ranges for each string (with harmonics)
        self.string_ranges = self._calculate_string_ranges()
        
    def _calculate_string_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Calculate frequency ranges for each guitar string including harmonics."""
        ranges = {}
        
        for i, (string_name, base_freq) in enumerate(zip(self.string_names, self.guitar_strings)):
            # Base frequency ± 50 cents
            min_freq = base_freq * (2 ** (-50/1200))
            max_freq = base_freq * (2 ** (50/1200))
            
            # Include up to 5th harmonic
            harmonic_max = base_freq * 5 * (2 ** (50/1200))
            
            ranges[string_name] = (min_freq, harmonic_max)
        
        return ranges
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for guitar transcription."""
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Apply guitar-specific filtering
        audio = self._apply_guitar_filter(audio)
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # Use harmonic component for chord analysis
        return y_harmonic
    
    def _apply_guitar_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply guitar-specific frequency filtering."""
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 60 / nyquist  # 60 Hz
        high_cutoff = 0.95  # 95% of Nyquist
        
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract guitar-specific spectral features."""
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Constant-Q Transform (better for musical analysis)
        cqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=84, bins_per_octave=12)
        cqt_magnitude = np.abs(cqt)
        
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(S=magnitude, sr=self.sample_rate)
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt_magnitude, sr=self.sample_rate)
        
        # Guitar-specific chroma (emphasize guitar string frequencies)
        guitar_chroma = self._extract_guitar_chroma(magnitude)
        
        # Spectral centroid and rolloff
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        
        return {
            'stft_magnitude': magnitude,
            'stft_phase': phase,
            'cqt_magnitude': cqt_magnitude,
            'chroma': chroma,
            'chroma_cqt': chroma_cqt,
            'guitar_chroma': guitar_chroma,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr
        }
    
    def _extract_guitar_chroma(self, magnitude: np.ndarray) -> np.ndarray:
        """Extract chroma features optimized for guitar strings."""
        # Create guitar-specific chroma template
        guitar_chroma = np.zeros((12, magnitude.shape[1]))
        
        # Map frequencies to chroma bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        for i, freq in enumerate(freqs):
            if freq > 0:
                # Convert frequency to chroma bin
                chroma_bin = int(round(12 * np.log2(freq / 440.0))) % 12
                
                # Weight by guitar string importance
                weight = self._get_guitar_string_weight(freq)
                
                # Add to chroma bin
                guitar_chroma[chroma_bin, :] += magnitude[i, :] * weight
        
        # Normalize
        guitar_chroma = guitar_chroma / (np.sum(guitar_chroma, axis=0, keepdims=True) + 1e-8)
        
        return guitar_chroma
    
    def _get_guitar_string_weight(self, frequency: float) -> float:
        """Get weight for frequency based on guitar string importance."""
        # Check if frequency is close to any guitar string or its harmonics
        max_weight = 0.0
        
        for base_freq in self.guitar_strings:
            # Check fundamental and harmonics
            for harmonic in range(1, 6):  # Up to 5th harmonic
                target_freq = base_freq * harmonic
                
                # Calculate distance in cents
                cents_diff = 1200 * np.log2(frequency / target_freq)
                
                # Weight decreases with distance
                if abs(cents_diff) < 100:  # Within 100 cents
                    weight = np.exp(-(cents_diff / 50) ** 2)  # Gaussian decay
                    max_weight = max(max_weight, weight)
        
        return max_weight
    
    def extract_string_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for each guitar string."""
        string_features = {}
        
        # Extract features for each string
        for i, (string_name, base_freq) in enumerate(zip(self.string_names, self.guitar_strings)):
            # Bandpass filter around string frequency
            string_audio = self._bandpass_filter(audio, base_freq)
            
            # Extract features
            stft = librosa.stft(string_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)
            
            # Energy
            energy = np.sum(magnitude ** 2, axis=0)
            
            string_features[string_name] = {
                'magnitude': magnitude,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'energy': energy
            }
        
        return string_features
    
    def _bandpass_filter(self, audio: np.ndarray, center_freq: float, bandwidth: float = 200) -> np.ndarray:
        """Apply bandpass filter around center frequency."""
        nyquist = self.sample_rate / 2
        
        # Calculate filter bounds
        low_freq = max(center_freq - bandwidth/2, 20) / nyquist
        high_freq = min(center_freq + bandwidth/2, nyquist * 0.95) / nyquist
        
        # Ensure valid frequency range
        if low_freq >= high_freq or low_freq <= 0 or high_freq >= 1:
            # Return original audio if filter parameters are invalid
            return audio
        
        try:
            # Apply filter
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_audio = filtfilt(b, a, audio)
            
            # Check for non-finite values
            if not np.all(np.isfinite(filtered_audio)):
                # Return original audio if filter produces non-finite values
                return audio
            
            return filtered_audio
        except:
            # Return original audio if filter fails
            return audio
    
    def create_training_data(self, audio: np.ndarray, target_chord: str, target_tablature: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Create training data for the model."""
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)
        
        # Extract features
        spectral_features = self.extract_spectral_features(processed_audio)
        string_features = self.extract_string_features(processed_audio)
        
        # Create input tensor
        # Use guitar chroma as main input
        input_tensor = torch.tensor(spectral_features['guitar_chroma'], dtype=torch.float32)
        
        # Create target tensors
        chord_target = self._encode_chord(target_chord)
        tablature_target = self._encode_tablature(target_tablature)
        
        return {
            'input': input_tensor,
            'chord_target': chord_target,
            'tablature_target': tablature_target,
            'features': spectral_features,
            'string_features': string_features
        }
    
    def _encode_chord(self, chord_name: str) -> torch.Tensor:
        """Encode chord name as one-hot vector."""
        chord_names = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
        ]
        
        if chord_name in chord_names:
            idx = chord_names.index(chord_name)
            target = torch.zeros(len(chord_names))
            target[idx] = 1.0
            return target
        else:
            # Unknown chord
            return torch.zeros(len(chord_names))
    
    def _encode_tablature(self, tablature: Dict[str, int]) -> torch.Tensor:
        """Encode tablature as tensor."""
        # 6 strings, 25 frets (0-24)
        tab_tensor = torch.zeros(6, 25)
        
        string_names = ['E', 'A', 'D', 'G', 'B', 'e']
        
        for i, string_name in enumerate(string_names):
            if string_name in tablature and tablature[string_name] is not None:
                fret = tablature[string_name]
                if 0 <= fret <= 24:
                    tab_tensor[i, fret] = 1.0
        
        return tab_tensor

# Example usage
if __name__ == "__main__":
    # Create processor
    processor = GuitarAudioProcessor()
    
    # Load example audio
    audio, sr = librosa.load('../project/test_chords/17569__danglada__c-major.wav', sr=22050)
    
    # Preprocess
    processed_audio = processor.preprocess_audio(audio)
    
    # Extract features
    spectral_features = processor.extract_spectral_features(processed_audio)
    string_features = processor.extract_string_features(processed_audio)
    
    print("Spectral features:")
    for key, value in spectral_features.items():
        print(f"  {key}: {value.shape}")
    
    print("\nString features:")
    for string_name, features in string_features.items():
        print(f"  {string_name}: {len(features)} features")
    
    # Create training data
    target_chord = "C"
    target_tablature = {'E': 3, 'A': 3, 'D': 0, 'G': 0, 'B': 1, 'e': 0}
    
    training_data = processor.create_training_data(processed_audio, target_chord, target_tablature)
    
    print(f"\nTraining data shapes:")
    print(f"  Input: {training_data['input'].shape}")
    print(f"  Chord target: {training_data['chord_target'].shape}")
    print(f"  Tablature target: {training_data['tablature_target'].shape}")
