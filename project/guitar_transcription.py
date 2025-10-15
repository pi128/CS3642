"""
AI-Driven Guitar Transcription System
From Audio to Frets: Automatic Music Transcription and Tablature Generation

This module implements the core pipeline for transcribing guitar audio into
playable tablature, following the methodology outlined in the project proposal.
"""

import numpy as np
import librosa
import soundfile as sf
import crepe
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GuitarTranscription:
    """
    Main class for guitar audio transcription to tablature.
    
    Pipeline:
    1. Audio preprocessing (normalization, spectrograms, chroma features)
    2. Onset detection and pitch tracking
    3. Chord recognition
    4. Tablature mapping
    5. Export to MIDI/tab
    """
    
    def __init__(self, sample_rate=22050):
        """Initialize the transcription system."""
        self.sample_rate = sample_rate
        self.guitar_strings = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # E2, A2, D3, G3, B3, E4
        self.string_names = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
        
    def load_audio(self, file_path):
        """Load and preprocess audio file."""
        print(f"Loading audio from {file_path}...")
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        print(f"Audio loaded: {len(audio)/sr:.2f}s, {sr}Hz")
        return audio, sr
    
    def extract_features(self, audio, sr):
        """Extract audio features for transcription."""
        print("Extracting audio features...")
        
        # Compute spectrograms
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Constant-Q Transform (better for musical analysis)
        cqt = librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12)
        cqt_magnitude = np.abs(cqt)
        
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(S=magnitude, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt_magnitude, sr=sr)
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y_percussive, sr=sr, 
                                                 hop_length=512, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        features = {
            'stft': stft,
            'magnitude': magnitude,
            'cqt': cqt,
            'cqt_magnitude': cqt_magnitude,
            'chroma': chroma,
            'chroma_cqt': chroma_cqt,
            'harmonic': y_harmonic,
            'percussive': y_percussive,
            'onset_frames': onset_frames,
            'onset_times': onset_times
        }
        
        print(f"Features extracted: {chroma.shape[1]} frames, {len(onset_times)} onsets detected")
        return features
    
    def estimate_pitch(self, audio, sr):
        """Estimate pitch using CREPE."""
        print("Estimating pitch with CREPE...")
        
        # Use CREPE for pitch estimation
        time, frequency, confidence, activation = crepe.predict(audio, sr, 
                                                               model_capacity='full',
                                                               viterbi=True)
        
        # Filter out low confidence estimates
        confidence_threshold = 0.3
        valid_mask = confidence > confidence_threshold
        filtered_frequency = frequency.copy()
        filtered_frequency[~valid_mask] = 0
        
        print(f"Pitch estimated: {np.sum(valid_mask)}/{len(frequency)} frames with confidence > {confidence_threshold}")
        
        return {
            'time': time,
            'frequency': frequency,
            'filtered_frequency': filtered_frequency,
            'confidence': confidence,
            'activation': activation
        }
    
    def detect_chords(self, chroma, onset_frames, sr):
        """Detect chords using chroma features."""
        print("Detecting chords...")
        
        # Simple chord templates (major, minor, 7th)
        chord_templates = self._create_chord_templates()
        
        # Analyze chroma at onset frames
        chord_sequence = []
        chord_times = []
        
        for onset_frame in onset_frames:
            # Get chroma vector at onset
            chroma_vector = chroma[:, onset_frame]
            
            # Find best matching chord
            best_chord = self._match_chord_template(chroma_vector, chord_templates)
            chord_sequence.append(best_chord)
            chord_times.append(librosa.frames_to_time(onset_frame, sr=sr, hop_length=512))
        
        print(f"Chords detected: {len(chord_sequence)} chord changes")
        return chord_sequence, chord_times
    
    def _create_chord_templates(self):
        """Create chord templates for major, minor, and 7th chords."""
        # Chroma order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],      # C-E-G
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],     # C#-F-G#
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],      # D-F#-A
            'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],     # D#-G-A#
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],      # E-G#-B
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],      # F-A-C
            'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],     # F#-A#-C#
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],      # G-B-D
            'G#': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],     # G#-C-D#
            'A': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],      # A-C-E
            'A#': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],     # A#-C#-F
            'B': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],      # B-D-F#
        }
        
        # Add minor chords
        for root in list(templates.keys()):
            major_template = np.array(templates[root])
            minor_template = major_template.copy()
            # Flatten the third (move 4 semitones up to 3)
            minor_template = np.roll(minor_template, -1)  # Simplified minor transformation
            templates[f"{root}m"] = minor_template.tolist()
        
        return templates
    
    def _match_chord_template(self, chroma_vector, templates):
        """Match chroma vector to best chord template."""
        best_chord = 'N'  # No chord
        best_score = 0
        
        for chord, template in templates.items():
            # Normalize both vectors
            chroma_norm = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-8)
            template_norm = np.array(template) / (np.linalg.norm(template) + 1e-8)
            
            # Cosine similarity
            score = np.dot(chroma_norm, template_norm)
            
            if score > best_score and score > 0.3:  # Threshold for chord detection
                best_score = score
                best_chord = chord
        
        return best_chord
    
    def map_to_tablature(self, frequency, time, confidence, onset_times):
        """Map pitch estimates to guitar tablature."""
        print("Mapping to guitar tablature...")
        
        tab_notes = []
        
        for i, (freq, conf, t) in enumerate(zip(frequency, confidence, time)):
            if conf > 0.3 and np.isscalar(freq) and freq > 0:  # Valid pitch
                # Find closest guitar string
                string_idx, fret = self._freq_to_string_fret(freq)
                
                if string_idx is not None:
                    tab_notes.append({
                        'time': t,
                        'frequency': freq,
                        'confidence': conf,
                        'string': string_idx,
                        'fret': fret,
                        'string_name': self.string_names[string_idx]
                    })
        
        # Group notes by onset times
        tab_sequence = self._group_notes_by_onsets(tab_notes, onset_times)
        
        print(f"Tablature mapped: {len(tab_notes)} notes, {len(tab_sequence)} chord positions")
        return tab_sequence
    
    def _freq_to_string_fret(self, frequency):
        """Convert frequency to string and fret number."""
        # Find closest string
        string_diffs = [abs(frequency - string_freq) for string_freq in self.guitar_strings]
        closest_string = np.argmin(string_diffs)
        
        # Calculate fret number (assuming 24 frets)
        string_freq = self.guitar_strings[closest_string]
        fret = int(round(12 * np.log2(frequency / string_freq)))
        
        # Check if fret is within reasonable range (0-24)
        if 0 <= fret <= 24:
            return closest_string, fret
        else:
            return None, None
    
    def _group_notes_by_onsets(self, tab_notes, onset_times):
        """Group tab notes by onset times to form chord positions."""
        if len(tab_notes) == 0 or len(onset_times) == 0:
            return []
        
        onset_tolerance = 0.1  # 100ms tolerance
        grouped_notes = []
        
        for onset_time in onset_times:
            chord_notes = []
            for note in tab_notes:
                time_diff = abs(float(note['time']) - float(onset_time))
                if time_diff <= onset_tolerance:
                    chord_notes.append(note)
            
            if len(chord_notes) > 0:
                grouped_notes.append({
                    'time': onset_time,
                    'notes': chord_notes
                })
        
        return grouped_notes
    
    def export_tablature(self, tab_sequence, output_file):
        """Export tablature to text file."""
        print(f"Exporting tablature to {output_file}...")
        
        with open(output_file, 'w') as f:
            f.write("🎸 GUITAR TABLATURE TRANSCRIPTION 🎸\n")
            f.write("=" * 60 + "\n")
            f.write("Standard Tuning: E2-A2-D3-G3-B3-E4\n")
            f.write("=" * 60 + "\n\n")
            
            for i, chord_pos in enumerate(tab_sequence):
                f.write(f"📍 Position {i+1} - Time: {chord_pos['time']:.2f}s\n")
                f.write("-" * 40 + "\n")
                
                # Create tablature representation with proper formatting
                tab_lines = ['e|', 'B|', 'G|', 'D|', 'A|', 'E|']
                string_names = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
                
                # Initialize all strings with dashes
                for j in range(len(tab_lines)):
                    tab_lines[j] += "---"
                
                # Add notes to appropriate strings
                for note in chord_pos['notes']:
                    string_idx = note['string']
                    fret = note['fret']
                    confidence = note['confidence']
                    
                    # Format fret number with confidence indicator
                    if confidence > 0.8:
                        fret_str = f"{fret:2d}"  # High confidence
                    elif confidence > 0.6:
                        fret_str = f"{fret:2d}"  # Medium confidence
                    else:
                        fret_str = f"{fret:2d}"  # Low confidence
                    
                    tab_lines[string_idx] += fret_str
                
                # Fill remaining space with dashes
                for j, line in enumerate(tab_lines):
                    remaining = 15 - len(line)
                    if remaining > 0:
                        line += "-" * remaining
                    tab_lines[j] = line
                
                # Write tablature in proper guitar order (high E to low E)
                for j, (line, string_name) in enumerate(zip(reversed(tab_lines), reversed(string_names))):
                    f.write(f"{string_name}: {line}\n")
                
                # Add note details
                f.write("\n📝 Notes detected:\n")
                for note in chord_pos['notes']:
                    f.write(f"  • {note['string_name']} string, fret {note['fret']} "
                           f"(freq: {note['frequency']:.1f}Hz, conf: {note['confidence']:.2f})\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Tablature exported to {output_file}")
    
    def export_latex_tablature(self, tab_sequence, output_file, title="Guitar Transcription"):
        """Export tablature to professional LaTeX format."""
        try:
            from professional_tabs import create_professional_guitar_tabs
            print(f"Creating professional LaTeX tablature: {output_file}")
            return create_professional_guitar_tabs(tab_sequence, output_file, title)
        except ImportError:
            print("⚠️ LaTeX tablature module not available. Using text format.")
            return self.export_tablature(tab_sequence, output_file)
    
    def visualize_results(self, audio, features, pitch_data, tab_sequence, output_file=None):
        """Create visualization of transcription results."""
        print("Creating visualization...")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. Waveform
        time_axis = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        axes[0].plot(time_axis, audio, alpha=0.7)
        axes[0].set_title('Audio Waveform')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Chroma features
        librosa.display.specshow(features['chroma'], sr=self.sample_rate, 
                                hop_length=512, x_axis='time', ax=axes[1])
        axes[1].set_title('Chroma Features')
        axes[1].set_ylabel('Pitch Class')
        
        # 3. Pitch tracking
        valid_pitch = pitch_data['filtered_frequency'] > 0
        axes[2].plot(pitch_data['time'], pitch_data['frequency'], alpha=0.3, label='Raw')
        axes[2].plot(pitch_data['time'][valid_pitch], 
                    pitch_data['filtered_frequency'][valid_pitch], 
                    'r-', linewidth=2, label='Filtered')
        axes[2].set_title('Pitch Tracking')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Tablature visualization
        axes[3].set_title('Detected Notes')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('String')
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for chord_pos in tab_sequence:
            for note in chord_pos['notes']:
                axes[3].scatter(note['time'], note['string'], 
                              c=colors[note['string']], s=100, alpha=0.7)
        
        axes[3].set_yticks(range(6))
        axes[3].set_yticklabels(self.string_names)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
    
    def transcribe_audio(self, audio_data, sample_rate, output_dir="./output"):
        """Transcribe audio data directly."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting guitar transcription pipeline...")
        print("=" * 50)
        
        # Use provided audio data
        audio, sr = audio_data, sample_rate
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Estimate pitch
        pitch_data = self.estimate_pitch(audio, sr)
        
        # Detect chords
        chord_sequence, chord_times = self.detect_chords(features['chroma'], 
                                                        features['onset_frames'], sr)
        
        # Map to tablature
        tab_sequence = self.map_to_tablature(pitch_data['frequency'], 
                                           pitch_data['time'], 
                                           pitch_data['confidence'],
                                           features['onset_times'])
        
        # Export results
        base_name = "transcription"
        tab_file = os.path.join(output_dir, f"{base_name}_tab.txt")
        viz_file = os.path.join(output_dir, f"{base_name}_visualization.png")
        
        self.export_tablature(tab_sequence, tab_file)
        self.visualize_results(audio, features, pitch_data, tab_sequence, viz_file)
        
        print("=" * 50)
        print("Transcription complete!")
        print(f"Results saved to: {output_dir}")
        
        return {
            'audio': audio,
            'features': features,
            'pitch_data': pitch_data,
            'chord_sequence': chord_sequence,
            'tab_sequence': tab_sequence,
            'chord_count': len(chord_sequence),
            'note_count': len(tab_sequence),
            'duration': len(audio) / sr
        }


def main():
    """Example usage of the guitar transcription system."""
    # Initialize transcription system
    transcriber = GuitarTranscription()
    
    # Example: transcribe a test audio file
    # Note: You'll need to provide an actual audio file
    audio_file = "test.wav"  # Replace with your audio file
    
    if os.path.exists(audio_file):
        results = transcriber.transcribe(audio_file)
        print("Transcription completed successfully!")
    else:
        print(f"Audio file {audio_file} not found. Please provide a valid audio file.")


if __name__ == "__main__":
    import os
    main()
