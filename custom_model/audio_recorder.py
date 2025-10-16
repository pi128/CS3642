"""
Audio Recorder Module
Handles real-time audio capture for guitar transcription.
"""

import os
import sys
import numpy as np
import threading
import time
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available. Audio recording will not work.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("⚠️  SoundFile not available. Audio saving will not work.")

class AudioRecorder:
    """Real-time audio recorder for guitar transcription."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format: int = None):
        """Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate for recording (default: 22050)
            channels: Number of audio channels (default: 1 for mono)
            chunk_size: Size of audio chunks (default: 1024)
            format: Audio format (default: pyaudio.paFloat32)
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available. Please install it: pip install pyaudio")
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format or pyaudio.paFloat32
        
        # Audio recording state
        self.is_recording = False
        self.is_paused = False
        self.audio_stream = None
        self.audio_data = []
        self.recording_thread = None
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.level_monitor_callback = None
        self.silence_detection_callback = None
        
        # PyAudio instance
        self.pyaudio_instance = None
        
        # Audio level monitoring
        self.current_level = 0.0
        self.level_history = []
        self.silence_threshold = 0.01  # Threshold for silence detection
        self.silence_duration = 0.0
        self.max_silence_duration = 2.0  # Max silence before auto-stop
        
        # Recording statistics
        self.recording_start_time = None
        self.total_recording_time = 0.0
        
    def initialize(self) -> bool:
        """Initialize the audio system."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Get default input device
            default_device = self._get_default_input_device()
            if default_device is None:
                print("❌ No audio input device found")
                return False
            
            print(f"✅ Audio system initialized")
            print(f"   Device: {default_device['name']}")
            print(f"   Sample rate: {self.sample_rate}")
            print(f"   Channels: {self.channels}")
            print(f"   Chunk size: {self.chunk_size}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize audio system: {e}")
            return False
    
    def _get_default_input_device(self) -> Optional[Dict]:
        """Get the default input device information."""
        try:
            default_device_index = self.pyaudio_instance.get_default_input_device_info()['index']
            device_info = self.pyaudio_instance.get_device_info_by_index(default_device_index)
            
            return {
                'index': device_info['index'],
                'name': device_info['name'],
                'max_input_channels': device_info['maxInputChannels'],
                'default_sample_rate': device_info['defaultSampleRate']
            }
        except Exception as e:
            print(f"❌ Error getting default input device: {e}")
            return None
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available audio input devices."""
        if not self.pyaudio_instance:
            return []
        
        devices = []
        device_count = self.pyaudio_instance.get_device_count()
        
        for i in range(device_count):
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': device_info['index'],
                        'name': device_info['name'],
                        'max_input_channels': device_info['maxInputChannels'],
                        'default_sample_rate': device_info['defaultSampleRate']
                    })
            except Exception:
                continue
        
        return devices
    
    def set_input_device(self, device_index: int) -> bool:
        """Set the input device by index."""
        try:
            device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
            if device_info['maxInputChannels'] > 0:
                print(f"✅ Input device set to: {device_info['name']}")
                return True
            else:
                print(f"❌ Device {device_index} has no input channels")
                return False
        except Exception as e:
            print(f"❌ Error setting input device: {e}")
            return False
    
    def start_recording(self, output_file: str = None) -> bool:
        """Start recording audio."""
        if self.is_recording:
            print("⚠️  Recording already in progress")
            return False
        
        if not self.pyaudio_instance:
            print("❌ Audio system not initialized")
            return False
        
        try:
            # Create audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Reset recording state
            self.audio_data = []
            self.is_recording = True
            self.is_paused = False
            self.recording_start_time = time.time()
            self.total_recording_time = 0.0
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            print(f"🎤 Recording started...")
            if output_file:
                print(f"   Output file: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data."""
        if not self.is_recording:
            print("⚠️  No recording in progress")
            return np.array([])
        
        print("🛑 Stopping recording...")
        
        # Stop recording
        self.is_recording = False
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        # Convert audio data to numpy array
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data)
            duration = len(audio_array) / self.sample_rate
            print(f"✅ Recording stopped. Duration: {duration:.2f}s")
            return audio_array
        else:
            print("⚠️  No audio data recorded")
            return np.array([])
    
    def pause_recording(self):
        """Pause the current recording."""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            print("⏸️  Recording paused")
    
    def resume_recording(self):
        """Resume the paused recording."""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            print("▶️  Recording resumed")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback function."""
        if self.is_recording and not self.is_paused:
            # Convert audio data to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to queue for processing
            if not self.audio_queue.full():
                self.audio_queue.put(audio_chunk)
        
        return (None, pyaudio.paContinue)
    
    def _recording_loop(self):
        """Main recording loop."""
        while self.is_recording:
            try:
                # Get audio chunk from queue
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Add to recording data
                    self.audio_data.append(audio_chunk)
                    
                    # Monitor audio level
                    self._monitor_audio_level(audio_chunk)
                    
                    # Check for silence
                    self._check_silence()
                    
                    # Call level monitor callback
                    if self.level_monitor_callback:
                        self.level_monitor_callback(self.current_level)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in recording loop: {e}")
                break
        
        # Clear remaining items in queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def _monitor_audio_level(self, audio_chunk: np.ndarray):
        """Monitor audio level for visualization."""
        # Calculate RMS level
        self.current_level = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Add to history
        self.level_history.append(self.current_level)
        
        # Keep only recent history (last 100 samples)
        if len(self.level_history) > 100:
            self.level_history.pop(0)
    
    def _check_silence(self):
        """Check for silence and trigger auto-stop if needed."""
        if self.current_level < self.silence_threshold:
            self.silence_duration += self.chunk_size / self.sample_rate
        else:
            self.silence_duration = 0.0
        
        # Auto-stop if silence too long
        if self.silence_duration > self.max_silence_duration:
            if self.silence_detection_callback:
                self.silence_detection_callback(self.silence_duration)
    
    def set_level_monitor_callback(self, callback: Callable[[float], None]):
        """Set callback for audio level monitoring."""
        self.level_monitor_callback = callback
    
    def set_silence_detection_callback(self, callback: Callable[[float], None]):
        """Set callback for silence detection."""
        self.silence_detection_callback = callback
    
    def set_silence_threshold(self, threshold: float):
        """Set silence detection threshold."""
        self.silence_threshold = threshold
        print(f"🔇 Silence threshold set to: {threshold:.3f}")
    
    def set_max_silence_duration(self, duration: float):
        """Set maximum silence duration before auto-stop."""
        self.max_silence_duration = duration
        print(f"⏱️  Max silence duration set to: {duration:.1f}s")
    
    def get_current_level(self) -> float:
        """Get current audio level."""
        return self.current_level
    
    def get_level_history(self) -> List[float]:
        """Get audio level history."""
        return self.level_history.copy()
    
    def get_recording_stats(self) -> Dict:
        """Get recording statistics."""
        current_time = time.time()
        recording_time = 0.0
        
        if self.is_recording and self.recording_start_time:
            recording_time = current_time - self.recording_start_time
        
        return {
            'is_recording': self.is_recording,
            'is_paused': self.is_paused,
            'recording_time': recording_time,
            'total_recording_time': self.total_recording_time,
            'current_level': self.current_level,
            'silence_duration': self.silence_duration,
            'audio_data_length': len(self.audio_data),
            'estimated_duration': sum(len(chunk) for chunk in self.audio_data) / self.sample_rate
        }
    
    def save_audio(self, audio_data: np.ndarray, filename: str) -> bool:
        """Save audio data to file."""
        if not SOUNDFILE_AVAILABLE:
            print("❌ SoundFile not available for saving audio")
            return False
        
        try:
            sf.write(filename, audio_data, self.sample_rate)
            print(f"✅ Audio saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return False
    
    def load_audio(self, filename: str) -> Optional[np.ndarray]:
        """Load audio from file."""
        if not SOUNDFILE_AVAILABLE:
            print("❌ SoundFile not available for loading audio")
            return None
        
        try:
            audio_data, sample_rate = sf.read(filename)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            print(f"✅ Audio loaded from: {filename}")
            print(f"   Duration: {len(audio_data) / self.sample_rate:.2f}s")
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return None
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.is_recording:
            self.stop_recording()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        print("🧹 Audio recorder cleaned up")

# Audio level monitoring utilities
class AudioLevelMonitor:
    """Utility class for monitoring audio levels."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.levels = []
    
    def update(self, level: float):
        """Update with new audio level."""
        self.levels.append(level)
        if len(self.levels) > self.window_size:
            self.levels.pop(0)
    
    def get_average_level(self) -> float:
        """Get average level over the window."""
        return np.mean(self.levels) if self.levels else 0.0
    
    def get_peak_level(self) -> float:
        """Get peak level over the window."""
        return np.max(self.levels) if self.levels else 0.0
    
    def is_silent(self, threshold: float = 0.01) -> bool:
        """Check if audio is silent."""
        return self.get_average_level() < threshold

# Example usage and testing
if __name__ == "__main__":
    print("🎤 Testing Audio Recorder")
    print("=" * 50)
    
    if not PYAUDIO_AVAILABLE:
        print("❌ PyAudio not available. Cannot test audio recording.")
        exit(1)
    
    # Create recorder
    recorder = AudioRecorder()
    
    # Initialize
    if not recorder.initialize():
        print("❌ Failed to initialize audio recorder")
        exit(1)
    
    # List available devices
    devices = recorder.get_available_devices()
    print(f"\n📱 Available audio devices ({len(devices)}):")
    for device in devices:
        print(f"  {device['index']}: {device['name']} ({device['max_input_channels']} channels)")
    
    # Set up level monitoring
    level_monitor = AudioLevelMonitor()
    
    def level_callback(level):
        level_monitor.update(level)
        # Print level every 10 updates
        if len(level_monitor.levels) % 10 == 0:
            print(f"🎵 Level: {level:.3f} (avg: {level_monitor.get_average_level():.3f})")
    
    def silence_callback(duration):
        print(f"🔇 Silence detected for {duration:.1f}s")
    
    recorder.set_level_monitor_callback(level_callback)
    recorder.set_silence_detection_callback(silence_callback)
    
    # Test recording
    print(f"\n🎤 Starting 5-second test recording...")
    print("   Speak or play guitar into your microphone...")
    
    if recorder.start_recording():
        time.sleep(5)  # Record for 5 seconds
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            print(f"✅ Recording completed: {len(audio_data)} samples")
            
            # Save audio
            output_file = "test_recording.wav"
            if recorder.save_audio(audio_data, output_file):
                print(f"✅ Audio saved to: {output_file}")
            
            # Load and verify
            loaded_audio = recorder.load_audio(output_file)
            if loaded_audio is not None:
                print(f"✅ Audio verification: {len(loaded_audio)} samples loaded")
        
        # Clean up
        recorder.cleanup()
        
        # Final stats
        stats = recorder.get_recording_stats()
        print(f"\n📊 Final stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    else:
        print("❌ Failed to start recording")
        recorder.cleanup()
