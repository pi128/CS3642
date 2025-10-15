"""
MIDI Export Module for Guitar Transcription
Converts transcription results to MIDI format for playback and further processing.
"""

import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
import os

class MIDIExporter:
    """Export guitar transcription results to MIDI format."""
    
    def __init__(self, tempo=120):
        """Initialize MIDI exporter with default tempo."""
        self.tempo = tempo
        self.ticks_per_beat = 480  # Standard MIDI resolution
        
    def export_tab_to_midi(self, tab_sequence, output_file, track_name="Guitar Transcription"):
        """Export tablature sequence to MIDI file."""
        print(f"Exporting MIDI to {output_file}...")
        
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set track name
        track.append(Message('track_name', name=track_name))
        
        # Set tempo
        track.append(Message('set_tempo', tempo=int(mido.bpm2tempo(self.tempo))))
        
        # Set instrument (acoustic guitar)
        track.append(Message('program_change', channel=0, program=24))  # Acoustic Guitar (nylon)
        
        # Convert tab sequence to MIDI events
        current_time = 0
        for chord_pos in tab_sequence:
            chord_time = chord_pos['time']
            
            # Calculate time delta in ticks
            time_delta = int((chord_time - current_time) * self.ticks_per_beat * self.tempo / 60)
            
            # Add note-on events for all notes in the chord
            for note in chord_pos['notes']:
                midi_note = self._freq_to_midi_note(note['frequency'])
                velocity = int(note['confidence'] * 127)  # Map confidence to velocity
                
                track.append(Message('note_on', channel=0, note=midi_note, 
                                   velocity=velocity, time=time_delta))
                time_delta = 0  # Only first note gets the time delta
            
            current_time = chord_time
        
        # Add note-off events (simplified - all notes off at end)
        track.append(Message('note_off', channel=0, note=60, velocity=0, time=0))
        
        # Save MIDI file
        mid.save(output_file)
        print(f"MIDI file saved: {output_file}")
        
    def _freq_to_midi_note(self, frequency):
        """Convert frequency to MIDI note number."""
        if frequency <= 0:
            return 60  # Middle C as default
        
        # MIDI note = 12 * log2(freq / 440) + 69
        midi_note = int(round(12 * np.log2(frequency / 440.0) + 69))
        
        # Clamp to valid MIDI range (0-127)
        return max(0, min(127, midi_note))
    
    def export_chord_sequence_to_midi(self, chord_sequence, chord_times, output_file):
        """Export chord sequence to MIDI file."""
        print(f"Exporting chord sequence to MIDI: {output_file}...")
        
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set track name
        track.append(Message('track_name', name="Chord Progression"))
        
        # Set tempo
        track.append(Message('set_tempo', tempo=int(mido.bpm2tempo(self.tempo))))
        
        # Set instrument
        track.append(Message('program_change', channel=0, program=24))
        
        # Chord to MIDI note mapping
        chord_notes = {
            'C': [60, 64, 67],      # C-E-G
            'C#': [61, 65, 68],     # C#-F-G#
            'D': [62, 66, 69],      # D-F#-A
            'D#': [63, 67, 70],     # D#-G-A#
            'E': [64, 68, 71],      # E-G#-B
            'F': [65, 69, 72],      # F-A-C
            'F#': [66, 70, 73],     # F#-A#-C#
            'G': [67, 71, 74],      # G-B-D
            'G#': [68, 72, 75],     # G#-C-D#
            'A': [69, 73, 76],      # A-C-E
            'A#': [70, 74, 77],     # A#-C#-F
            'B': [71, 75, 78],      # B-D-F#
            'N': []                 # No chord
        }
        
        # Add minor chords
        for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
            major_notes = chord_notes[root]
            minor_notes = [major_notes[0], major_notes[1] - 1, major_notes[2]]  # Flatten third
            chord_notes[f"{root}m"] = minor_notes
        
        current_time = 0
        for chord, chord_time in zip(chord_sequence, chord_times):
            if chord in chord_notes and chord_notes[chord]:
                # Calculate time delta
                time_delta = int((chord_time - current_time) * self.ticks_per_beat * self.tempo / 60)
                
                # Add chord notes
                for i, note in enumerate(chord_notes[chord]):
                    track.append(Message('note_on', channel=0, note=note, 
                                       velocity=80, time=time_delta if i == 0 else 0))
                
                current_time = chord_time
        
        # End of track
        track.append(Message('end_of_track', time=0))
        
        # Save MIDI file
        mid.save(output_file)
        print(f"Chord MIDI file saved: {output_file}")


def create_test_audio():
    """Create a simple test audio file with guitar-like tones."""
    import librosa
    import soundfile as sf
    
    print("Creating test audio file...")
    
    # Generate simple chord progression: C - Am - F - G
    sample_rate = 22050
    duration = 4.0  # 4 seconds per chord
    total_duration = 16.0  # 16 seconds total
    
    # Chord frequencies (fundamental + harmonics)
    chords = {
        'C': [261.63, 329.63, 392.00],      # C-E-G
        'Am': [220.00, 261.63, 329.63],     # A-C-E
        'F': [174.61, 220.00, 261.63],      # F-A-C
        'G': [196.00, 246.94, 293.66]       # G-B-D
    }
    
    audio = np.zeros(int(total_duration * sample_rate))
    
    for i, (chord_name, frequencies) in enumerate(chords.items()):
        start_time = i * duration
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + duration) * sample_rate)
        
        # Generate chord with attack and decay
        chord_audio = np.zeros(end_sample - start_sample)
        
        for freq in frequencies:
            # Generate sine wave with harmonics
            t = np.linspace(0, duration, end_sample - start_sample)
            wave = np.sin(2 * np.pi * freq * t)
            
            # Add harmonics for more guitar-like sound
            wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)  # Octave
            wave += 0.25 * np.sin(2 * np.pi * freq * 3 * t)  # Fifth
            
            # Apply envelope (attack and decay)
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
            wave *= envelope
            
            chord_audio += wave
        
        # Normalize and add to main audio
        chord_audio = chord_audio / np.max(np.abs(chord_audio)) * 0.3
        audio[start_sample:end_sample] = chord_audio
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise
    
    # Save test audio
    output_file = "test_guitar.wav"
    sf.write(output_file, audio, sample_rate)
    print(f"Test audio created: {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Create test audio and export to MIDI
    test_file = create_test_audio()
    
    # Test MIDI export
    exporter = MIDIExporter()
    
    # Create dummy tab sequence for testing
    dummy_tab = [
        {'time': 0.0, 'notes': [
            {'frequency': 261.63, 'confidence': 0.8, 'string': 0, 'fret': 0},
            {'frequency': 329.63, 'confidence': 0.7, 'string': 1, 'fret': 0},
            {'frequency': 392.00, 'confidence': 0.9, 'string': 2, 'fret': 0}
        ]},
        {'time': 4.0, 'notes': [
            {'frequency': 220.00, 'confidence': 0.8, 'string': 0, 'fret': 0},
            {'frequency': 261.63, 'confidence': 0.7, 'string': 1, 'fret': 0},
            {'frequency': 329.63, 'confidence': 0.9, 'string': 2, 'fret': 0}
        ]}
    ]
    
    exporter.export_tab_to_midi(dummy_tab, "test_output.mid")
    print("MIDI export test completed!")
