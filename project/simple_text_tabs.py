#!/usr/bin/env python3
"""
Simple text-based guitar tablature generator that creates real, playable tabs.
"""

import os
import numpy as np

class SimpleTextTabs:
    """Simple text-based guitar tablature generator."""
    
    def __init__(self):
        self.string_names = ['E', 'A', 'D', 'G', 'B', 'e']  # Low E to High e
        
    def create_text_tabs(self, tab_sequence, output_file, title="Guitar Transcription", artist="AI System"):
        """Create simple text-based guitar tablature."""
        measures = self._group_into_measures(tab_sequence)
        chords = self._extract_chords(tab_sequence)
        
        with open(output_file, 'w') as f:
            # Header
            f.write(f"{title} by {artist}\n")
            f.write("=" * 50 + "\n\n")
            
            # Chord progression
            if chords:
                chord_names = [chord[0] if isinstance(chord, tuple) else str(chord) for chord in chords]
                f.write(f"Chord Progression: {' - '.join(chord_names)}\n\n")
            
            # Create tablature
            f.write("Guitar Tablature:\n")
            f.write("-" * 50 + "\n")
            
            # Create tab lines for each string
            for i, string_name in enumerate(self.string_names):
                f.write(f"{string_name}|")
                
                # Add notes for each measure
                for measure in measures[:8]:  # Limit to 8 measures
                    # Find notes on this string in this measure
                    notes_on_string = [note for note in measure if note['string'] == i]
                    
                    if notes_on_string:
                        fret = notes_on_string[0]['fret']
                        f.write(f"--{fret}--")
                    else:
                        f.write("-----")
                
                f.write("|\n")
            
            f.write("-" * 50 + "\n\n")
            
            # Analysis
            f.write(f"Analysis:\n")
            f.write(f"- Total Measures: {len(measures)}\n")
            f.write(f"- Total Notes: {sum(len(m) for m in measures)}\n")
            f.write(f"- Chords Detected: {len(chords)}\n")
            f.write(f"- Tuning: Standard (E A D G B e)\n")
        
        print(f"✅ Text tabs created: {output_file}")
    
    def _group_into_measures(self, tab_sequence, measure_duration=2.0):
        """Group notes into measures."""
        if not tab_sequence:
            return []
        
        # Flatten and group notes by time
        all_notes = []
        for chord_group in tab_sequence:
            if 'notes' in chord_group:
                all_notes.extend(chord_group['notes'])
            else:
                all_notes.append(chord_group)
        
        time_groups = {}
        for note in all_notes:
            time_key = round(note['time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)
        
        measures = []
        current_measure = []
        measure_duration = 2.0
        
        for time_key in sorted(time_groups.keys()):
            if len(current_measure) > 0 and time_key - current_measure[0]['time'] > measure_duration:
                measures.append(current_measure)
                current_measure = []
            
            current_measure.extend(time_groups[time_key])
        
        if current_measure:
            measures.append(current_measure)
        
        return measures
    
    def _extract_chords(self, tab_sequence):
        """Extract chord information."""
        chords = []
        seen_chords = set()
        
        all_notes = []
        for chord_group in tab_sequence:
            if 'notes' in chord_group:
                all_notes.extend(chord_group['notes'])
            else:
                all_notes.append(chord_group)
        
        time_groups = {}
        for note in all_notes:
            time_key = round(note['time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)
        
        for time_key in sorted(time_groups.keys()):
            notes = time_groups[time_key]
            if len(notes) >= 3:
                chord_name, confidence = self._identify_chord(notes)
                if chord_name and chord_name not in seen_chords:
                    chords.append((chord_name, confidence))
                    seen_chords.add(chord_name)
        
        return chords
    
    def _identify_chord(self, notes):
        """Identify chord from notes."""
        note_names = []
        for note in notes:
            freq = note['frequency']
            note_name = self._frequency_to_note_name(freq)
            if note_name:
                note_names.append(note_name)
        
        if len(note_names) >= 3:
            root = note_names[0]
            base_confidence = min(len(notes) / 6.0, 1.0)
            
            if 'C' in note_names and 'E' in note_names and 'G' in note_names:
                return "C", base_confidence + 0.2
            elif 'G' in note_names and 'B' in note_names and 'D' in note_names:
                return "G", base_confidence + 0.2
            elif 'A' in note_names and 'C' in note_names and 'E' in note_names:
                return "Am", base_confidence + 0.2
            elif 'F' in note_names and 'A' in note_names and 'C' in note_names:
                return "F", base_confidence + 0.2
            elif 'D' in note_names and 'F#' in note_names and 'A' in note_names:
                return "D", base_confidence + 0.2
            elif 'E' in note_names and 'G' in note_names and 'B' in note_names:
                return "Em", base_confidence + 0.2
            else:
                return f"{root}M", base_confidence
        
        return None, 0.0
    
    def _frequency_to_note_name(self, frequency):
        """Convert frequency to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        semitones = int(round(12 * np.log2(frequency / 440.0)))
        note_index = (semitones + 9) % 12
        return note_names[note_index]

def create_test_data():
    """Create test tablature data."""
    # Create realistic guitar chord progression data
    test_data = []
    
    # C Major chord
    c_chord = [
        {'time': 0.0, 'frequency': 261.63, 'confidence': 0.9, 'string': 5, 'fret': 3},  # C
        {'time': 0.0, 'frequency': 329.63, 'confidence': 0.8, 'string': 4, 'fret': 3},  # E
        {'time': 0.0, 'frequency': 392.00, 'confidence': 0.9, 'string': 3, 'fret': 0},  # G
        {'time': 0.0, 'frequency': 523.25, 'confidence': 0.7, 'string': 2, 'fret': 1},  # C
        {'time': 0.0, 'frequency': 659.25, 'confidence': 0.8, 'string': 1, 'fret': 0},  # E
    ]
    
    # G Major chord
    g_chord = [
        {'time': 2.0, 'frequency': 196.00, 'confidence': 0.9, 'string': 5, 'fret': 3},  # G
        {'time': 2.0, 'frequency': 246.94, 'confidence': 0.8, 'string': 4, 'fret': 0},  # B
        {'time': 2.0, 'frequency': 293.66, 'confidence': 0.9, 'string': 3, 'fret': 0},  # D
        {'time': 2.0, 'frequency': 392.00, 'confidence': 0.7, 'string': 2, 'fret': 0},  # G
        {'time': 2.0, 'frequency': 493.88, 'confidence': 0.8, 'string': 1, 'fret': 3},  # B
        {'time': 2.0, 'frequency': 659.25, 'confidence': 0.7, 'string': 0, 'fret': 3},  # G
    ]
    
    # A Minor chord
    am_chord = [
        {'time': 4.0, 'frequency': 220.00, 'confidence': 0.9, 'string': 5, 'fret': 0},  # A
        {'time': 4.0, 'frequency': 261.63, 'confidence': 0.8, 'string': 4, 'fret': 2},  # C
        {'time': 4.0, 'frequency': 329.63, 'confidence': 0.9, 'string': 3, 'fret': 2},  # E
        {'time': 4.0, 'frequency': 440.00, 'confidence': 0.7, 'string': 2, 'fret': 0},  # A
        {'time': 4.0, 'frequency': 523.25, 'confidence': 0.8, 'string': 1, 'fret': 0},  # C
        {'time': 4.0, 'frequency': 659.25, 'confidence': 0.7, 'string': 0, 'fret': 0},  # E
    ]
    
    # F Major chord
    f_chord = [
        {'time': 6.0, 'frequency': 174.61, 'confidence': 0.9, 'string': 5, 'fret': 1},  # F
        {'time': 6.0, 'frequency': 220.00, 'confidence': 0.8, 'string': 4, 'fret': 0},  # A
        {'time': 6.0, 'frequency': 261.63, 'confidence': 0.9, 'string': 3, 'fret': 2},  # C
        {'time': 6.0, 'frequency': 349.23, 'confidence': 0.7, 'string': 2, 'fret': 1},  # F
        {'time': 6.0, 'frequency': 440.00, 'confidence': 0.8, 'string': 1, 'fret': 0},  # A
        {'time': 6.0, 'frequency': 523.25, 'confidence': 0.7, 'string': 0, 'fret': 1},  # C
    ]
    
    # D Major chord
    d_chord = [
        {'time': 8.0, 'frequency': 146.83, 'confidence': 0.9, 'string': 5, 'fret': 0},  # D
        {'time': 8.0, 'frequency': 185.00, 'confidence': 0.8, 'string': 4, 'fret': 2},  # F#
        {'time': 8.0, 'frequency': 220.00, 'confidence': 0.9, 'string': 3, 'fret': 2},  # A
        {'time': 8.0, 'frequency': 293.66, 'confidence': 0.7, 'string': 2, 'fret': 0},  # D
        {'time': 8.0, 'frequency': 369.99, 'confidence': 0.8, 'string': 1, 'fret': 2},  # F#
        {'time': 8.0, 'frequency': 440.00, 'confidence': 0.7, 'string': 0, 'fret': 0},  # A
    ]
    
    # E Minor chord
    em_chord = [
        {'time': 10.0, 'frequency': 164.81, 'confidence': 0.9, 'string': 5, 'fret': 0},  # E
        {'time': 10.0, 'frequency': 196.00, 'confidence': 0.8, 'string': 4, 'fret': 2},  # G
        {'time': 10.0, 'frequency': 246.94, 'confidence': 0.9, 'string': 3, 'fret': 2},  # B
        {'time': 10.0, 'frequency': 329.63, 'confidence': 0.7, 'string': 2, 'fret': 0},  # E
        {'time': 10.0, 'frequency': 392.00, 'confidence': 0.8, 'string': 1, 'fret': 0},  # G
        {'time': 10.0, 'frequency': 493.88, 'confidence': 0.7, 'string': 0, 'fret': 0},  # B
    ]
    
    # Combine all chords
    all_chords = [c_chord, g_chord, am_chord, f_chord, d_chord, em_chord]
    
    # Create tab sequence format
    tab_sequence = []
    for chord_notes in all_chords:
        tab_sequence.append({
            'time': chord_notes[0]['time'],
            'notes': chord_notes
        })
    
    return tab_sequence

def test_text_tabs():
    """Test text-based guitar tablature generation."""
    print("🎸 Testing Text-Based Guitar Tablature")
    print("=" * 50)
    
    # Create test data
    tab_sequence = create_test_data()
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize text tablature generator
    text_tabs = SimpleTextTabs()
    
    print("\n🎨 Creating Text-Based Guitar Tabs...")
    
    # Create text-based guitar tabs
    output_file = os.path.join(output_dir, "simple_text_tabs.txt")
    text_tabs.create_text_tabs(
        tab_sequence, output_file,
        "C-G-Am-F-D-Em Progression", "AI Transcription Test"
    )
    
    print("\n" + "=" * 50)
    print("🎉 Text-based guitar tabs created!")
    print(f"📁 Output file: {output_file}")
    print("\n✨ This creates simple, readable guitar tablature!")

if __name__ == "__main__":
    test_text_tabs()
