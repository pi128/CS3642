"""
Import Audio Files for Guitar Transcription
Simple script to help you import and transcribe your own guitar recordings.
"""

import os
import shutil
import sys

def import_audio_file(audio_path, new_name=None):
    """
    Import an audio file into the project directory.
    
    Args:
        audio_path (str): Path to your audio file
        new_name (str): Optional new name for the file
    """
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found!")
        return False
    
    # Get file extension
    _, ext = os.path.splitext(audio_path)
    ext = ext.lower()
    
    # Check if it's a supported audio format
    supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
    if ext not in supported_formats:
        print(f"❌ Error: Unsupported audio format '{ext}'")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return False
    
    # Create new filename
    if new_name is None:
        new_name = os.path.basename(audio_path)
    else:
        if not new_name.endswith(ext):
            new_name += ext
    
    # Copy file to project directory
    destination = os.path.join(os.getcwd(), new_name)
    
    try:
        shutil.copy2(audio_path, destination)
        print(f"✅ Audio file imported: {new_name}")
        print(f"📁 Location: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error importing file: {e}")
        return False

def main():
    """Main function for importing audio files."""
    print("🎸 IMPORT AUDIO FILES FOR GUITAR TRANSCRIPTION")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        new_name = sys.argv[2] if len(sys.argv) > 2 else None
        import_audio_file(audio_path, new_name)
    else:
        print("\n💡 How to import your guitar recordings:")
        print("   1. Drag and drop your audio file into this terminal")
        print("   2. Or run: python import_audio.py /path/to/your/file.wav")
        print("   3. Or run: python import_audio.py /path/to/your/file.wav new_name")
        
        print(f"\n📝 Supported audio formats:")
        print(f"   • .wav (recommended for best quality)")
        print(f"   • .mp3")
        print(f"   • .flac")
        print(f"   • .m4a")
        print(f"   • .ogg")
        print(f"   • .aiff")
        
        print(f"\n🎯 After importing:")
        print(f"   1. Run: python transcribe_guitar.py your_file.wav")
        print(f"   2. Check the 'output' folder for results")
        
        print(f"\n📁 Current directory: {os.getcwd()}")
        print(f"📁 Audio files already here:")
        
        # List existing audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
        audio_files = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(file)
        
        if audio_files:
            for file in audio_files:
                print(f"   • {file}")
        else:
            print("   (No audio files found)")

if __name__ == "__main__":
    main()
