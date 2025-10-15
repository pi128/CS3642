#!/usr/bin/env python3
"""
🎸 Simple Guitar Transcriber
Command-line interface for guitar transcription with improved tablature.
"""

import sys
import os
import argparse
from guitar_transcription import GuitarTranscription
from clean_chord_tabs import CleanChordTabs
import soundfile as sf

def main():
    """Main function for command-line guitar transcription."""
    parser = argparse.ArgumentParser(description='🎸 AI Guitar Tablature Generator')
    parser.add_argument('audio_file', help='Path to audio file (.wav, .mp3, .flac, etc.)')
    parser.add_argument('-t', '--title', default='My Guitar Song', help='Song title')
    parser.add_argument('-a', '--artist', default='Unknown Artist', help='Artist name')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file '{args.audio_file}' not found!")
        sys.exit(1)
    
    print("🎸 AI Guitar Tablature Generator")
    print("=" * 50)
    print(f"📁 Audio file: {args.audio_file}")
    print(f"🎵 Title: {args.title}")
    print(f"👤 Artist: {args.artist}")
    print(f"📂 Output: {args.output}")
    print("=" * 50)
    
    try:
        # Load audio
        print("🎵 Loading audio...")
        audio, sr = sf.read(args.audio_file)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        print(f"✅ Audio loaded: {len(audio)/sr:.1f}s, {sr}Hz")
        
        # Transcribe
        print("🎸 Transcribing guitar...")
        transcription = GuitarTranscription()
        result = transcription.transcribe_audio(audio, sr, args.output)
        
        print(f"✅ Transcription complete!")
        print(f"📊 {result['chord_count']} chord positions detected")
        print(f"🎵 {result['note_count']} total notes")
        
        # Generate professional tabs
        print("📝 Generating professional tablature...")
        tabs_gen = CleanChordTabs()
        
        # Create official output filename
        tabs_file = os.path.join(args.output, "official_guitar_tabs.tex")
        
        tabs_gen.create_professional_tabs(
            result['tab_sequence'], 
            tabs_file, 
            args.title,
            args.artist
        )
        
        print("=" * 50)
        print("🎉 SUCCESS! Official professional tablature generated!")
        print(f"📝 LaTeX tabs: {tabs_file.replace('.tex', '_latex.tex')}")
        print(f"🖼️ Visual tabs: {tabs_file.replace('.tex', '_visual.png')}")
        print()
        print("🔧 To create PDF:")
        print(f"   1. cd {args.output}")
        print(f"   2. pdflatex official_guitar_tabs_latex.tex")
        print(f"   3. open official_guitar_tabs_latex.pdf")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
